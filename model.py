import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import logging
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, XLMRobertaModel
from collections import OrderedDict
import random
import torch.utils.checkpoint as checkpoint
import torch.quantization as quant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

world_size = 1
rank = 0

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 131072
    initial_seq_len: int = 4096
    context_scaling_factor: float = 2.0
    vocab_size: int = 250002
    dim: int = 6144
    inter_dim: int = 24576
    moe_inter_dim: int = 3072
    n_layers: int = 96
    n_dense_layers: int = 4
    n_heads: int = 64
    n_kv_heads: int = 16
    n_routed_experts: int = 80
    n_shared_experts: int = 2
    n_activated_experts: int = 8
    score_func: str = "softmax"
    route_scale: float = 1.5
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    head_dim: int = 96
    rope_theta: float = 50000.0
    yarn_scaling_factor: float = 1.0
    yarn_beta: float = 0.1
    chunk_size: int = 4096
    summary_size: int = 128
    max_input_tokens: int = 200000
    window_size: int = 512
    global_tokens: int = 128
    attn_reg_weight: float = 0.005
    consistency_loss_weight: float = 0.05
    summarizer_n_layers: int = 4
    summarizer_n_heads: int = 16
    summarizer_ffn_multiplier: int = 4
    memory_bank_max_chunks: int = 300
    router_supervised_weight: float = 0.05
    balance_weight: float = 0.3
    entropy_weight: float = 0.1
    z_loss_weight: float = 1e-4
    dropout: float = 0.1
    rl_reward_correct: float = 25.0
    rl_reward_incorrect: float = -35.0
    use_full_attention_first_n_layers: int = 1
    use_layer_norm: bool = False
    use_swiglu: bool = True
    lora_rank: int = 128

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))
        self.fake_quant = quant.FakeQuantize.with_meta(
            torch.per_tensor_affine, torch.qint8, 0, -128, 127
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fake_quant(self.weight)
        return F.embedding(x, weight)

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, lora_rank: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(out_features, lora_rank, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(lora_rank, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.fake_quant_weight = quant.FakeQuantize.with_meta(
            torch.per_tensor_affine, torch.qint8, 0, -128, 127
        )
        self.fake_quant_output = quant.FakeQuantize.with_meta(
            torch.per_tensor_affine, torch.qint8, 0, -128, 127
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fake_quant_weight(self.weight)
        y = F.linear(x, weight, self.bias)
        lora_adjust = F.linear(F.linear(x, self.lora_B), self.lora_A)
        return self.fake_quant_output(y + lora_adjust)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)

def precompute_yarn_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, scaling_factor: float = 1.0, beta: float = 0.1) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    scaled_freqs = freqs * scaling_factor
    freqs = (1 - beta) * freqs + beta * scaled_freqs
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_.size(1), 1, x_.size(-1))
    x_rotated = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_rotated.to(x.dtype)

class FullAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.wq = LoRALinear(args.dim, self.n_heads * self.head_dim, args.q_lora_rank)
        self.wk = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank)
        self.wv = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank)
        self.wo = LoRALinear(self.n_heads * self.head_dim, args.dim, args.lora_rank)
        self.scale = (self.head_dim ** -0.5)
        self.freqs_cis = precompute_yarn_freqs_cis(
            self.head_dim // 2, args.max_seq_len, args.rope_theta, args.yarn_scaling_factor, args.yarn_beta
        )
        self.kv_cache = None
        self.attn_reg_weight = args.attn_reg_weight
        self.dropout = nn.Dropout(args.dropout)

    def compute_attention_reward(self, scores: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        focus = scores.var(dim=-1).mean()
        return torch.where(focus > 0.1, torch.tensor(self.args.rl_reward_correct), torch.tensor(self.args.rl_reward_incorrect))

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, torch.Tensor]:
        bsz, seqlen, _ = x.size()
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        q_rot, q_no_rot = torch.split(q, [self.head_dim // 2, self.head_dim // 2], dim=-1)
        k_rot, k_no_rot = torch.split(k, [self.head_dim // 2, self.head_dim // 2], dim=-1)
        q_rot = apply_rotary_emb(q_rot, self.freqs_cis[start_pos:start_pos + seqlen].to(x.device))
        k_rot = apply_rotary_emb(k_rot, self.freqs_cis[start_pos:start_pos + seqlen].to(x.device))
        q = torch.cat([q_rot, q_no_rot], dim=-1)
        k = torch.cat([k_rot, k_no_rot], dim=-1)

        if self.kv_cache is None or start_pos == 0:
            self.kv_cache = (k, v)
        else:
            self.kv_cache = (torch.cat([self.kv_cache[0], k], dim=1), torch.cat([self.kv_cache[1], v], dim=1))

        k, v = self.kv_cache

        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.scale
        if mask is not None:
            scores = scores + mask  # Mask now includes both causal and padding masks
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores = self.dropout(scores)
        attn_reg_loss = self.attn_reg_weight * (scores.var(dim=-1).mean())
        out = torch.einsum("bsht,bthd->bshd", scores, v).flatten(2)
        attn_reward = self.compute_attention_reward(scores, x)
        return self.wo(out), attn_reg_loss.item(), attn_reward

class SparseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.wq = LoRALinear(args.dim, self.n_heads * self.head_dim, args.q_lora_rank)
        self.wk = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank)
        self.wv = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, args.kv_lora_rank)
        self.wo = LoRALinear(self.n_heads * self.head_dim, args.dim, args.lora_rank)
        self.scale = (self.head_dim ** -0.5)
        self.freqs_cis = precompute_yarn_freqs_cis(
            self.head_dim // 2, args.max_seq_len, args.rope_theta, args.yarn_scaling_factor, args.yarn_beta
        )
        self.kv_cache = None
        self.attn_reg_weight = args.attn_reg_weight
        self.window_size = args.window_size
        self.global_tokens = args.global_tokens
        self.policy_network = nn.Sequential(
            nn.Linear(args.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(args.dropout)

    def get_attention_mask(self, x: torch.Tensor, seqlen: int, k_seqlen: int) -> torch.Tensor:
        adjustments = self.policy_network(x.mean(dim=1))
        window_adjust = adjustments[:, 0] * 100
        global_adjust = adjustments[:, 1] * 10
        window_size = (self.window_size + window_adjust).clamp(min=128, max=1024).int()
        global_tokens = (self.global_tokens + global_adjust).clamp(min=16, max=256).int()

        global_mask = torch.zeros(x.size(0), self.n_heads, seqlen, k_seqlen, device=x.device, dtype=x.dtype)
        global_mask[:, :, :, :global_tokens.max()] = 1
        local_mask = torch.zeros(x.size(0), self.n_heads, seqlen, k_seqlen, device=x.device, dtype=x.dtype)
        for b in range(x.size(0)):
            for i in range(seqlen):
                start = max(0, i - window_size[b] // 2)
                end = min(k_seqlen, i + window_size[b] // 2 + 1)
                local_mask[b, :, i, start:end] = 1
        return global_mask + local_mask, window_size, global_tokens

    def compute_attention_reward(self, scores: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        focus = scores.var(dim=-1).mean()
        return torch.where(focus > 0.1, torch.tensor(self.args.rl_reward_correct), torch.tensor(self.args.rl_reward_incorrect))

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, torch.Tensor]:
        bsz, seqlen, _ = x.size()
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        q_rot, q_no_rot = torch.split(q, [self.head_dim // 2, self.head_dim // 2], dim=-1)
        k_rot, k_no_rot = torch.split(k, [self.head_dim // 2, self.head_dim // 2], dim=-1)
        q_rot = apply_rotary_emb(q_rot, self.freqs_cis[start_pos:start_pos + seqlen].to(x.device))
        k_rot = apply_rotary_emb(k_rot, self.freqs_cis[start_pos:start_pos + seqlen].to(x.device))
        q = torch.cat([q_rot, q_no_rot], dim=-1)
        k = torch.cat([k_rot, k_no_rot], dim=-1)

        if self.kv_cache is None or start_pos == 0:
            self.kv_cache = (k, v)
        else:
            self.kv_cache = (torch.cat([self.kv_cache[0], k], dim=1), torch.cat([self.kv_cache[1], v], dim=1))

        k, v = self.kv_cache

        attention_mask, _, _ = self.get_attention_mask(x, seqlen, k.size(1))
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.scale
        scores = scores * attention_mask + (1 - attention_mask) * float("-inf")
        if mask is not None:
            scores = scores + mask  # Combine with provided mask
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores = self.dropout(scores)
        attn_reg_loss = self.attn_reg_weight * (scores.var(dim=-1).mean())
        out = torch.einsum("bsht,bthd->bshd", scores, v).flatten(2)
        attn_reward = self.compute_attention_reward(scores, x)
        return self.wo(out), attn_reg_loss.item(), attn_reward

class TransformerSummarizer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.chunk_size = args.chunk_size
        self.summary_size = args.summary_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.dim,
                nhead=args.summarizer_n_heads,
                dim_feedforward=args.dim * args.summarizer_ffn_multiplier,
                dropout=args.dropout,
                batch_first=True
            ),
            num_layers=args.summarizer_n_layers
        )
        self.attention_pool = nn.MultiheadAttention(args.dim, args.summarizer_n_heads, batch_first=True, dropout=args.dropout)
        self.summary_projector = LoRALinear(args.dim, args.dim, args.lora_rank, dtype=torch.bfloat16)
        self.policy_network = nn.Sequential(
            nn.Linear(args.dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.summarizer_n_heads),
            nn.Softmax(dim=-1)
        )

    def compute_summary_reward(self, summary: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(summary.mean(dim=1), x.mean(dim=1), dim=-1).mean()
        return torch.where(similarity > 0.8, torch.tensor(self.args.rl_reward_correct), torch.tensor(self.args.rl_reward_incorrect))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.size()
        x = self.encoder(x)
        query = torch.zeros(bsz, 1, dim, device=x.device, dtype=x.dtype)
        policy_weights = self.policy_network(x.mean(dim=1)).unsqueeze(1)
        summary, _ = self.attention_pool(query, x, x)
        summary = summary.squeeze(1).view(bsz, self.summary_size, dim // self.summary_size)
        summary = self.summary_projector(summary)
        reward = self.compute_summary_reward(summary, x)
        return summary, reward

class DynamicMemoryBank(nn.Module):
    def __init__(self, dim: int, summary_size: int, max_chunks: int):
        super().__init__()
        self.memory = OrderedDict()
        self.max_chunks = max_chunks
        self.summary_size = summary_size
        self.dim = dim
        self.persistent_keys = set()

    def add(self, summary: torch.Tensor, key: int, persistent: bool = False):
        if key in self.memory:
            self.memory.pop(key)
        if len(self.memory) >= self.max_chunks and key not in self.persistent_keys:
            for k in list(self.memory.keys()):
                if k not in self.persistent_keys:
                    self.memory.pop(k)
                    break
        self.memory[key] = summary.detach()
        self.memory.move_to_end(key)
        if persistent:
            self.persistent_keys.add(key)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        if not self.memory:
            return torch.zeros(1, self.summary_size, self.dim, device=query.device, dtype=query.dtype)
        summaries = torch.stack(list(self.memory.values()))
        query_norm = query.mean(dim=1) / (query.mean(dim=1).norm(dim=-1, keepdim=True) + 1e-6)
        summaries_norm = summaries.mean(dim=1) / (summaries.mean(dim=1).norm(dim=-1, keepdim=True) + 1e-6)
        scores = F.cosine_similarity(query_norm.unsqueeze(0), summaries_norm, dim=-1)
        topk_indices = torch.topk(scores, min(k, len(self.memory)), dim=0)[1]
        retrieved = summaries[topk_indices]
        weights = F.softmax(scores[topk_indices], dim=0).unsqueeze(-1).unsqueeze(-1)
        return (retrieved * weights).sum(dim=0)

    def get_memory_usage(self) -> float:
        num_entries = len(self.memory)
        entry_size = self.summary_size * self.dim * 2
        total_bytes = num_entries * entry_size
        return total_bytes / (1024 * 1024)

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, lora_rank: int, use_swiglu: bool = False, dropout: float = 0.1):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.w1 = LoRALinear(dim, inter_dim, lora_rank)
        self.w2 = LoRALinear(inter_dim, dim, lora_rank)
        self.w3 = LoRALinear(dim, inter_dim, lora_rank)
        self.dropout = nn.Dropout(dropout)
        self.fake_quant_output = quant.FakeQuantize.with_meta(
            torch.per_tensor_affine, torch.qint8, 0, -128, 127
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gate = torch.sigmoid(self.w3(x))
            out = self.w2(self.dropout(self.w1(x) * gate))
        else:
            out = self.w2(self.dropout(F.gelu(self.w1(x)) * self.w3(x)))
        return self.fake_quant_output(out)

class DynamicRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_experts = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.scorer = nn.Linear(args.dim, args.n_routed_experts, dtype=torch.bfloat16)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.balance_weight = args.balance_weight
        self.entropy_weight = args.entropy_weight
        self.z_loss_weight = args.z_loss_weight
        self.step = 0
        self.total_steps = 1_000_000
        self.supervised_weight = args.router_supervised_weight
        self.policy_network = nn.Sequential(
            nn.Linear(args.dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.n_routed_experts),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(args.dropout)

    def update_temperature(self):
        progress = self.step / self.total_steps
        self.temp.data = torch.tensor(0.5 + 0.5 * math.cos(math.pi * progress), dtype=torch.bfloat16)
        self.step += 1

    def compute_router_reward(self, indices: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is None:
            return torch.tensor(0.0, device=indices.device)
        correct = (indices == labels.unsqueeze(-1)).any(dim=-1).float()
        return torch.where(correct.bool(), torch.tensor(self.args.rl_reward_correct), torch.tensor(self.args.rl_reward_incorrect)).mean()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.update_temperature()
        policy_scores = self.policy_network(x.mean(dim=1))
        base_scores = self.scorer(x.mean(dim=1)) / self.temp.clamp(min=0.1)
        scores = (base_scores + policy_scores) / 2
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        scores = self.dropout(scores)

        entropy = -(scores * torch.log(scores + 1e-6)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy
        z_loss = self.z_loss_weight * (torch.logsumexp(base_scores, dim=-1) ** 2).mean()

        weights, indices = torch.topk(scores, self.topk, dim=-1)
        weights *= self.temp.clamp(min=0.1)
        expert_usage = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        expert_usage.scatter_add_(0, indices.flatten(), weights.flatten())
        usage_mean = expert_usage.mean()
        usage_fraction = expert_usage / (usage_mean + 1e-6)
        balance_loss = self.balance_weight * usage_fraction.var() + entropy_loss + z_loss

        supervised_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            supervised_loss = F.cross_entropy(scores, labels) * self.supervised_weight

        reward = self.compute_router_reward(indices, labels)
        return weights, indices, expert_usage, balance_loss, supervised_loss, reward

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, lora_rank: int, use_swiglu: bool = False, dropout: float = 0.1):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.w1 = LoRALinear(dim, inter_dim, lora_rank)
        self.w2 = LoRALinear(inter_dim, dim, lora_rank)
        self.w3 = LoRALinear(dim, inter_dim, lora_rank)
        self.dropout = nn.Dropout(dropout)
        self.fake_quant_output = quant.FakeQuantize.with_meta(
            torch.per_tensor_affine, torch.qint8, 0, -128, 127
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            gate = torch.sigmoid(self.w3(x))
            out = self.w2(self.dropout(self.w1(x) * gate))
        else:
            out = self.w2(self.dropout(F.gelu(self.w1(x)) * self.w3(x)))
        return self.fake_quant_output(out)

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.router = DynamicRouter(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, args.lora_rank, args.use_swiglu, args.dropout)
            for _ in range(args.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.moe_inter_dim, args.lora_rank, args.use_swiglu, args.dropout)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, usage, balance_loss, supervised_loss, router_reward = self.router(x, labels)
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel() > 0:
                y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape), usage, balance_loss, supervised_loss, router_reward

class LanguageAdapter(nn.Module):
    def __init__(self, dim: int, lora_rank: int):
        super().__init__()
        self.down = LoRALinear(dim, dim // 4, lora_rank)
        self.up = LoRALinear(dim // 4, dim, lora_rank)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(F.gelu(self.down(x))))

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = FullAttention(args) if layer_id < args.use_full_attention_first_n_layers else SparseAttention(args)
        self.ffn = MLP(args.dim, args.inter_dim, args.lora_rank, args.use_swiglu, args.dropout) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = LayerNorm(args.dim) if args.use_layer_norm else RMSNorm(args.dim)
        self.ffn_norm = LayerNorm(args.dim) if args.use_layer_norm else RMSNorm(args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.hindi_adapter = LanguageAdapter(args.dim, args.lora_rank) if layer_id % 2 == 0 else None
        self.memory_gate = nn.Linear(args.dim * 2, args.dim, dtype=torch.bfloat16)
        self.memory_gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], memory: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, languages: Optional[List[str]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        def checkpointed_forward(x, start_pos, mask, memory, labels, languages):
            if memory is not None:
                combined = torch.cat([x, memory], dim=-1)
                gate = self.memory_gate_act(self.memory_gate(combined))
                attn_input = x + gate * memory
            else:
                attn_input = x

            h = self.attn_norm(attn_input)
            x_attn, attn_reg_loss, attn_reward = self.attn(h, start_pos, mask)
            x = x + self.dropout(x_attn)
            if self.hindi_adapter is not None and languages is not None:
                hindi_mask = torch.tensor([1 if lang == "hindi" else 0 for lang in languages], device=x.device, dtype=torch.bfloat16).unsqueeze(-1).unsqueeze(-1)
                x = x + hindi_mask * self.hindi_adapter(x)
            h = self.ffn_norm(x)
            if isinstance(self.ffn, MoE):
                x_ffn, usage, balance_loss, supervised_loss, router_reward = self.ffn(h, labels)
                x = x + self.dropout(x_ffn)
                return x, usage, balance_loss + attn_reg_loss, supervised_loss, router_reward + attn_reward
            x_ffn = self.ffn(h)
            x = x + self.dropout(x_ffn)
            return x, None, attn_reg_loss, torch.tensor(0.0, device=x.device), attn_reward

        return checkpoint.checkpoint(checkpointed_forward, x, start_pos, mask, memory, labels, languages)

class XenArcAIi1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        try:
            self.device = xm.xla_device()
        except Exception as e:
            logger.warning(f"XLA device not available: {str(e)}. Falling back to CPU/GPU.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._initialize_tokenizer()
        if self.tokenizer is None:
            raise ValueError("Tokenizer initialization failed.")
        assert self.tokenizer.vocab_size == self.args.vocab_size, f"Tokenizer vocab size ({self.tokenizer.vocab_size}) does not match model vocab size ({self.args.vocab_size})"

        self.token_embedding = ParallelEmbedding(self.args.vocab_size, self.args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = LayerNorm(args.dim) if args.use_layer_norm else RMSNorm(args.dim)
        self.output_layer = LoRALinear(args.dim, args.vocab_size, args.lora_rank, bias=False)
        self.context_projector = LoRALinear(args.dim, args.dim, args.lora_rank, dtype=torch.bfloat16)
        self.summarizer = TransformerSummarizer(args)
        self.memory_bank = DynamicMemoryBank(args.dim, args.summary_size, args.memory_bank_max_chunks)
        self.cluster_to_label = None
        self.model_version = "1.0.0"

        self._initialize_with_xlmr()
        self.to(self.device)

        total_params, active_params = self.count_parameters()
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Active parameters (considering MoE): {active_params:,}")

    def _initialize_with_xlmr(self):
        try:
            xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-large")
            logger.info("Loaded XLM-RoBERTa weights for initialization.")
            xlmr_state_dict = xlmr.state_dict()
            our_state_dict = self.state_dict()
            if "embeddings.word_embeddings.weight" in xlmr_state_dict:
                xlmr_emb = xlmr_state_dict["embeddings.word_embeddings.weight"]
                if xlmr_emb.size(1) <= self.args.dim:
                    our_state_dict["token_embedding.weight"][:xlmr_emb.size(0), :xlmr_emb.size(1)] = xlmr_emb
            self.load_state_dict(our_state_dict, strict=False)
            logger.info("Initialized model with XLM-RoBERTa weights where compatible.")
        except Exception as e:
            logger.warning(f"Failed to load XLM-RoBERTa weights: {str(e)}. Proceeding with random initialization.")

    def count_parameters(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        active_params = 0
        active_params += sum(p.numel() for p in self.token_embedding.parameters())
        active_params += sum(p.numel() for p in self.output_layer.parameters())
        active_params += sum(p.numel() for p in self.context_projector.parameters())
        active_params += sum(p.numel() for p in self.summarizer.parameters())
        active_params += sum(p.numel() for p in self.memory_bank.parameters())
        active_params += sum(p.numel() for p in self.norm.parameters())

        for layer in self.layers:
            active_params += sum(p.numel() for p in layer.attn.parameters())
            active_params += sum(p.numel() for p in layer.attn_norm.parameters())
            active_params += sum(p.numel() for p in layer.ffn_norm.parameters())
            if layer.hindi_adapter is not None:
                active_params += sum(p.numel() for p in layer.hindi_adapter.parameters())
            if isinstance(layer.ffn, MoE):
                active_params += sum(p.numel() for p in layer.ffn.router.parameters())
                active_params += sum(p.numel() for p in layer.ffn.shared_experts.parameters())
                total_expert_params = sum(sum(p.numel() for p in expert.parameters()) for expert in layer.ffn.experts)
                active_expert_params = (total_expert_params / self.args.n_routed_experts) * self.args.n_activated_experts
                active_params += int(active_expert_params)
            else:
                active_params += sum(p.numel() for p in layer.ffn.parameters())

        return total_params, active_params

    def _initialize_tokenizer(self):
        tokenizer_path = "custom_xlmr_tokenizer"
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}.")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}. Falling back to xlm-roberta-large.")
            try:
                tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
                if tokenizer.vocab_size != self.args.vocab_size:
                    tokenizer.add_tokens([f"<extra_token_{i}>" for i in range(self.args.vocab_size - tokenizer.vocab_size)])
                tokenizer.save_pretrained(tokenizer_path)
                logger.info(f"Initialized and saved tokenizer to {tokenizer_path}")
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer: {str(e)}")
                return None
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def process_long_input(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        chunks = [input_ids[:, i:i + self.args.chunk_size] for i in range(0, input_ids.size(1), self.args.chunk_size)]
        if chunks[-1].size(1) < self.args.chunk_size:
            padding = torch.full(
                (input_ids.size(0), self.args.chunk_size - chunks[-1].size(1)),
                self.tokenizer.pad_token_id,
                dtype=input_ids.dtype,
                device=self.device
            )
            chunks[-1] = torch.cat([chunks[-1], padding], dim=1)
        return chunks

    def summarize_chunks(self, chunks: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        summaries = []
        total_reward = torch.tensor(0.0, device=self.device)
        for idx, chunk in enumerate(chunks):
            chunk = chunk.to(self.device)
            emb = self.token_embedding(chunk)
            summary, reward = self.summarizer(emb)
            self.memory_bank.add(summary, key=idx, persistent=(idx == 0))
            summaries.append(summary)
            total_reward += reward
        if rank == 0:
            logger.info(f"Memory bank usage: {self.memory_bank.get_memory_usage():.2f} MB")
        return torch.cat(summaries, dim=1), total_reward / len(chunks)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, start_pos: int = 0, router_labels: Optional[torch.Tensor] = None, languages: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if router_labels is not None:
            router_labels = router_labels.to(self.device)
        bsz, total_len = input_ids.size()

        if total_len > self.args.max_seq_len:
            chunks = self.process_long_input(input_ids)
            memory, summary_reward = self.summarize_chunks(chunks)
            active_ids = chunks[-1]
            # Adjust attention_mask for the last chunk if provided
            if attention_mask is not None:
                active_mask = attention_mask[:, -self.args.chunk_size:]
            else:
                active_mask = None
        else:
            active_ids = input_ids
            active_mask = attention_mask
            memory = None
            summary_reward = torch.tensor(0.0, device=self.device)

        x = self.token_embedding(active_ids)
        # Create causal mask and combine with padding mask if provided
        mask = torch.full((bsz, active_ids.size(1), active_ids.size(1)), float('-inf'), device=self.device)
        mask = torch.triu(mask, diagonal=1)  # Causal mask
        if active_mask is not None:
            # Convert padding mask (1s and 0s) to attention mask (-inf where padded)
            padding_mask = (1 - active_mask).bool().unsqueeze(1).unsqueeze(1) * float('-inf')
            mask = mask + padding_mask  # Combine causal and padding masks

        total_usage = None
        total_balance_loss = torch.tensor(0.0, device=self.device)
        total_supervised_loss = torch.tensor(0.0, device=self.device)
        total_router_reward = torch.tensor(0.0, device=self.device)
        total_attn_reward = torch.tensor(0.0, device=self.device)
        total_attn_reg_loss = torch.tensor(0.0, device=self.device)

        if memory is not None:
            memory = self.context_projector(memory)

        for i, layer in enumerate(self.layers):
            x, usage, balance_loss, supervised_loss, reward = layer(x, start_pos, mask, memory=memory, labels=router_labels, languages=languages)
            total_balance_loss += balance_loss
            total_supervised_loss += supervised_loss
            if usage is not None:
                total_usage = total_usage + usage if total_usage is not None else usage
            if isinstance(layer.ffn, MoE):
                total_router_reward += reward
            else:
                total_attn_reward += reward
            total_attn_reg_loss += balance_loss  # Accumulate attn_reg_loss from attention layers

        x = self.norm(x)
        logits = self.output_layer(x)
        return logits, total_usage, total_balance_loss, total_supervised_loss, total_router_reward, total_attn_reward

    def save(self, path: str):
        state_dict = self.state_dict()
        total_params = len(state_dict)
        params_per_file = total_params // 4 + (1 if total_params % 4 else 0)

        for i in range(4):
            start_idx = i * params_per_file
            end_idx = min((i + 1) * params_per_file, total_params)
            part_dict = dict(list(state_dict.items())[start_idx:end_idx])
            save_file(part_dict, f"{path}_part_{i+1}.safetensors")
            logger.info(f"Saved model part {i+1} to {path}_part_{i+1}.safetensors")