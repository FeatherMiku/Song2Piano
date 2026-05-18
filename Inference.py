import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Args:
    # Paths
    input_file: str = "testdata/input/001_01.jsonl"
    output_path: str = "testdata/pred_output"
    ckpt_path: str = "testdata/short_best_ok.pt"
    files_on: bool = False
    files_path: str = "testdata/inputs"

    # Runtime
    seed: int = 39
    device: str = "1,2"  # auto | cpu | cuda | cuda:0 | cuda:0,1 or 0,1
    batch_size: int = 4
    num_workers: int = 8
    pin_memory: bool = True

    # Inference safety and decode
    use_amp: bool = False
    amp_dtype: str = "bf16"  # fp16 | bf16 | fp32
    max_input_len: int = 32768
    gen_max_len: int = 32768
    temperature: float = 1.0
    top_k: int = 0
    decode_log_every: int = 256

    # Tokens
    pad_id: int = 1024
    bos_id: int = 1025
    eos_id: int = 1026

    # Model architecture (must match training)
    vocab_size: int = 1027
    d_model: int = 256
    ff_mult: int = 4
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.0
    bucket_size: int = 64
    n_hashes: int = 4
    ff_chunk_size: int = 256
    compressed_mem_len: int = 768
    cross_attn_query_chunk_size: int = 128
    cross_attn_min_chunk_size: int = 32
    max_len: int = 32768


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_input_files(args: Args) -> List[Path]:
    if bool(args.files_on):
        base_dir = Path(args.files_path)
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"files_path is not a valid directory: {base_dir}")
        files = sorted(p for p in base_dir.glob("*.jsonl") if p.is_file())
        if len(files) == 0:
            raise FileNotFoundError(f"No jsonl files found in files_path: {base_dir}")
        return files

    input_path = Path(args.input_file)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"input_file is not a valid file: {input_path}")
    return [input_path]


def _extract_int_sequence(raw_obj) -> List[int]:
    if isinstance(raw_obj, list) and len(raw_obj) == 1 and isinstance(raw_obj[0], list):
        raw_obj = raw_obj[0]
    if not isinstance(raw_obj, list):
        raise ValueError("Expected a list or nested single list.")

    seq: List[int] = []
    for x in raw_obj:
        if isinstance(x, bool):
            raise ValueError("Boolean token is not allowed.")
        if isinstance(x, int):
            seq.append(int(x))
        else:
            raise ValueError(f"Non-integer token found: {type(x)}")
    return seq


def read_jsonl_array_line(path: Path) -> List[List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) == 0:
        raise ValueError(f"Input file is empty: {path}")

    sequences: List[List[int]] = []
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
            seq = _extract_int_sequence(obj)
            if len(seq) == 0:
                raise ValueError("Empty sequence is not allowed.")
        except Exception as ex:
            raise ValueError(f"Parse error at {path}, line {i + 1}: {ex}") from ex
        sequences.append(seq)
    return sequences


class InferenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_input_len: int, pad_id: int):
        self.sequences = sequences
        self.max_input_len = max_input_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        src = self.sequences[idx]
        src_tokens = src[: self.max_input_len]
        src_mask = [1] * len(src_tokens)
        return {
            "src_tokens": src_tokens,
            "src_mask": src_mask,
        }


def collate_infer_batch(batch: List[Dict[str, object]], pad_id: int) -> Dict[str, object]:
    src_list = [item["src_tokens"] for item in batch]

    batch_max_len = max(len(s) for s in src_list)
    batch_size = len(batch)

    src_tokens = torch.full((batch_size, batch_max_len), int(pad_id), dtype=torch.long)
    src_mask = torch.zeros((batch_size, batch_max_len), dtype=torch.bool)
    for i, seq in enumerate(src_list):
        n = len(seq)
        if n == 0:
            continue
        src_tokens[i, :n] = torch.as_tensor(seq, dtype=torch.long)
        src_mask[i, :n] = True

    return {
        "src_tokens": src_tokens,
        "src_mask": src_mask,
    }


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, max_distance: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(n_heads, 2 * max_distance + 1))
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel = (k_pos - q_pos).clamp(-self.max_distance, self.max_distance) + self.max_distance
        return self.bias[:, rel]


class ChunkedFFN(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.fc1 = nn.Linear(d_model, d_model * ff_mult)
        self.fc2 = nn.Linear(d_model * ff_mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, l, _ = x.shape
        outs = []
        for start in range(0, l, self.chunk_size):
            end = min(start + self.chunk_size, l)
            chunk = x[:, start:end, :]
            chunk = self.fc2(self.drop(F.gelu(self.fc1(chunk))))
            outs.append(chunk)
        return torch.cat(outs, dim=1)


class ReversibleResidualBlock(nn.Module):
    def __init__(self, f_block: nn.Module, g_block: nn.Module):
        super().__init__()
        self.f = f_block
        self.g = g_block

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = x1 + self.f(x2, mask)
        y2 = x2 + self.g(y1, mask)
        return y1, y2


class ReversibleSequence(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        for block in self.blocks:
            x1, x2 = block(x1, x2, mask)
        return torch.cat([x1, x2], dim=-1)


def _round_to_bucket_multiple(x: int, bucket: int) -> int:
    return ((x + bucket - 1) // bucket) * bucket


class LSHSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bucket_size: int,
        n_hashes: int,
        dropout: float,
        causal: bool,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.causal = causal

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rel_bias = RelativePositionBias(n_heads=n_heads)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _hash_buckets(self, q: torch.Tensor) -> torch.Tensor:
        b, h, _, d = q.shape
        rand = torch.randn(self.n_hashes, d, device=q.device, dtype=q.dtype)
        proj = torch.einsum("bhld,rd->bhlr", q, rand)
        buckets = torch.argmax(proj, dim=-1)
        if buckets.shape[0] != b or buckets.shape[1] != h:
            raise RuntimeError("Unexpected bucket shape")
        return buckets

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self._shape(q)
        k = self._shape(k)
        v = self._shape(v)

        buckets = self._hash_buckets(q)

        outputs = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(self.head_dim)
        rel_bias_all = self.rel_bias(self.bucket_size, self.bucket_size, x.device)
        mask_fill_value = float(torch.finfo(q.dtype).min)
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(self.bucket_size, self.bucket_size, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        for bi in range(b):
            for hi in range(self.n_heads):
                order = torch.argsort(buckets[bi, hi], dim=0)
                qh = q[bi, hi, order, :]
                kh = k[bi, hi, order, :]
                vh = v[bi, hi, order, :]

                q_len = qh.size(0)
                padded = _round_to_bucket_multiple(q_len, self.bucket_size)
                if padded != q_len:
                    pad_sz = padded - q_len
                    qh = F.pad(qh, (0, 0, 0, pad_sz))
                    kh = F.pad(kh, (0, 0, 0, pad_sz))
                    vh = F.pad(vh, (0, 0, 0, pad_sz))

                qh = qh.view(-1, self.bucket_size, self.head_dim)
                kh = kh.view(-1, self.bucket_size, self.head_dim)
                vh = vh.view(-1, self.bucket_size, self.head_dim)

                logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
                logits = logits + rel_bias_all[hi].unsqueeze(0)

                valid = attn_mask[bi, order]
                if padded != q_len:
                    valid = F.pad(valid, (0, padded - q_len), value=False)
                valid = valid.view(-1, self.bucket_size)
                logits = logits.masked_fill(~valid[:, None, :], mask_fill_value)

                if causal_mask is not None:
                    logits = logits.masked_fill(causal_mask.unsqueeze(0), mask_fill_value)

                probs = F.softmax(logits, dim=-1)
                probs = self.drop(probs)
                yh = torch.matmul(probs, vh).reshape(-1, self.head_dim)[:q_len]

                inv_order = torch.empty_like(order)
                inv_order[order] = torch.arange(order.numel(), device=order.device)
                yh = yh[inv_order]
                outputs[bi, hi] = yh

        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(b, l, self.d_model)
        return self.proj(outputs)


class ReformerSubLayer(nn.Module):
    def __init__(
        self,
        d_half: int,
        n_heads: int,
        bucket_size: int,
        n_hashes: int,
        dropout: float,
        causal: bool,
        ff_mult: int,
        ff_chunk_size: int,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_half)
        self.attn = LSHSelfAttention(
            d_model=d_half,
            n_heads=max(1, n_heads // 2),
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            dropout=dropout,
            causal=causal,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_half)
        self.ffn = ChunkedFFN(d_half, ff_mult=ff_mult, dropout=dropout, chunk_size=ff_chunk_size)
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_attn(self.attn(self.norm_attn(x), mask))
        x = x + self.drop_ffn(self.ffn(self.norm_ffn(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, query_chunk_size: int, min_chunk_size: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.min_chunk_size = max(1, int(min_chunk_size))

    def _run_chunked_mha(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_len = q.size(1)
        chunk_size = min(self.query_chunk_size, total_len)

        while True:
            chunks = []
            try:
                for start in range(0, total_len, chunk_size):
                    end = min(start + chunk_size, total_len)
                    y_chunk, _ = self.mha(
                        query=q[:, start:end, :],
                        key=memory,
                        value=memory,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                    )
                    chunks.append(y_chunk)
                return torch.cat(chunks, dim=1)
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if chunk_size <= self.min_chunk_size:
                    raise
                new_chunk = max(self.min_chunk_size, chunk_size // 2)
                if new_chunk == chunk_size:
                    raise
                chunk_size = new_chunk
                print(f"[OOM] CrossAttention reduce query_chunk_size to {chunk_size}")

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        key_padding_mask = ~memory_key_padding_mask
        y = self._run_chunked_mha(x_norm, memory, key_padding_mask)
        return x + self.drop(y)


class MemoryCompressor(nn.Module):
    def __init__(self, compressed_len: int):
        super().__init__()
        self.compressed_len = max(1, int(compressed_len))

    def forward(self, memory: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, d = memory.shape
        out_len = min(self.compressed_len, l)

        if out_len == l:
            return memory, src_mask.bool()

        edges = torch.linspace(0, l, steps=out_len + 1, device=memory.device)
        seg_start = torch.floor(edges[:-1]).to(dtype=torch.long)
        seg_end = torch.floor(edges[1:]).to(dtype=torch.long)
        seg_end = torch.maximum(seg_end, seg_start + 1).clamp(max=l)

        mem_t = memory.transpose(1, 2).contiguous()
        mem_prefix = torch.cat(
            [torch.zeros((b, d, 1), device=memory.device, dtype=memory.dtype), mem_t.cumsum(dim=-1)],
            dim=-1,
        )
        seg_sum = mem_prefix.index_select(dim=2, index=seg_end) - mem_prefix.index_select(dim=2, index=seg_start)
        seg_len = (seg_end - seg_start).clamp(min=1).to(dtype=memory.dtype).view(1, 1, out_len)
        compressed = (seg_sum / seg_len).transpose(1, 2).contiguous()

        src_mask_i = src_mask.to(dtype=torch.int32)
        mask_prefix = torch.cat(
            [torch.zeros((b, 1), device=memory.device, dtype=torch.int32), src_mask_i.cumsum(dim=1)],
            dim=1,
        )
        compressed_mask = (
            mask_prefix.index_select(dim=1, index=seg_end) - mask_prefix.index_select(dim=1, index=seg_start)
        ) > 0
        if compressed_mask.shape != (b, out_len):
            raise RuntimeError("Compressed mask shape mismatch")
        return compressed, compressed_mask


class ReformerEncoder(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        if args.d_model % 2 != 0:
            raise ValueError("d_model must be even for reversible split")
        d_half = args.d_model // 2
        blocks = []
        for _ in range(args.n_layers):
            f = ReformerSubLayer(
                d_half=d_half,
                n_heads=args.n_heads,
                bucket_size=args.bucket_size,
                n_hashes=args.n_hashes,
                dropout=args.dropout,
                causal=False,
                ff_mult=args.ff_mult,
                ff_chunk_size=args.ff_chunk_size,
            )
            g = ReformerSubLayer(
                d_half=d_half,
                n_heads=args.n_heads,
                bucket_size=args.bucket_size,
                n_hashes=args.n_hashes,
                dropout=args.dropout,
                causal=False,
                ff_mult=args.ff_mult,
                ff_chunk_size=args.ff_chunk_size,
            )
            blocks.append(ReversibleResidualBlock(f, g))
        self.rev = ReversibleSequence(nn.ModuleList(blocks))
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.rev(x, src_mask)
        return self.norm(h)


class ReformerDecoderLayer(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.self_norm = nn.LayerNorm(args.d_model)
        self.self_attn = LSHSelfAttention(
            d_model=args.d_model,
            n_heads=args.n_heads,
            bucket_size=args.bucket_size,
            n_hashes=args.n_hashes,
            dropout=args.dropout,
            causal=True,
        )
        self.self_drop = nn.Dropout(args.dropout)

        self.cross_attn = CrossAttention(
            args.d_model,
            args.n_heads,
            args.dropout,
            query_chunk_size=args.cross_attn_query_chunk_size,
            min_chunk_size=args.cross_attn_min_chunk_size,
        )

        self.ff_norm = nn.LayerNorm(args.d_model)
        self.ff = ChunkedFFN(args.d_model, args.ff_mult, args.dropout, args.ff_chunk_size)
        self.ff_drop = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        compressed_memory: torch.Tensor,
        compressed_src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_drop(self.self_attn(self.self_norm(x), tgt_mask))
        x = self.cross_attn(x, compressed_memory, compressed_src_mask)
        x = x + self.ff_drop(self.ff(self.ff_norm(x)))
        return x


class ReformerDecoder(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.layers = nn.ModuleList([ReformerDecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.d_model)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        compressed_memory: torch.Tensor,
        compressed_src_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, src_mask, compressed_memory, compressed_src_mask)
        return self.norm(x)


class ReformerSeq2Seq(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_emb = nn.Embedding(args.max_len, args.d_model)
        self.drop = nn.Dropout(args.dropout)
        self.encoder = ReformerEncoder(args)
        self.memory_compressor = MemoryCompressor(args.compressed_mem_len)
        self.decoder = ReformerDecoder(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device).unsqueeze(0).expand(b, l)
        return self.drop(self.tok_emb(tokens) + self.pos_emb(pos))

    def encode(self, src_tokens: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_h = self.embed(src_tokens)
        memory = self.encoder(src_h, src_mask)
        compressed_memory, compressed_src_mask = self.memory_compressor(memory, src_mask)
        return memory, compressed_memory, compressed_src_mask

    def decode_with_memory(
        self,
        tgt_input_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        compressed_memory: torch.Tensor,
        compressed_src_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_h = self.embed(tgt_input_tokens)
        dec_h = self.decoder(
            tgt_h,
            tgt_mask,
            memory,
            src_mask,
            compressed_memory,
            compressed_src_mask,
        )
        logits = self.lm_head(dec_h)
        return logits

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_input_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory, compressed_memory, compressed_src_mask = self.encode(src_tokens, src_mask)
        return self.decode_with_memory(
            tgt_input_tokens,
            tgt_mask,
            memory,
            src_mask,
            compressed_memory,
            compressed_src_mask,
        )


def choose_device(device_str: str) -> torch.device:
    if "," in str(device_str):
        gpu_ids = resolve_gpu_ids_from_device(device_str)
        if len(gpu_ids) == 0:
            return torch.device("cpu")
        return torch.device(f"cuda:{gpu_ids[0]}")

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu")

    if requested.type == "cuda" and requested.index is not None:
        if requested.index >= torch.cuda.device_count():
            print(
                f"[WARN] Requested CUDA index {requested.index} is out of range "
                f"(count={torch.cuda.device_count()}). Falling back to cuda:0."
            )
            return torch.device("cuda:0")
    return requested


def resolve_gpu_ids_from_device(device_str: str) -> List[int]:
    if not torch.cuda.is_available():
        return []

    lowered = str(device_str).strip().lower()
    visible_count = torch.cuda.device_count()

    if lowered in {"cpu"}:
        return []
    if lowered in {"auto", "cuda"}:
        return list(range(visible_count))

    tokens = [t.strip() for t in lowered.split(",") if t.strip() != ""]
    if len(tokens) == 0:
        raise ValueError("device is empty. Use cpu/auto/cuda/cuda:0 or comma-separated gpu ids.")

    gpu_ids: List[int] = []
    for tok in tokens:
        if tok.startswith("cuda:"):
            tok = tok.split(":", 1)[1].strip()
        if tok == "":
            raise ValueError(f"Invalid device token: '{device_str}'")
        if not tok.isdigit():
            raise ValueError(
                f"Unsupported device token '{tok}'. Use formats like cuda:0,cuda:1 or 0,1."
            )
        gpu_id = int(tok)
        if gpu_id < 0 or gpu_id >= visible_count:
            raise ValueError(
                f"GPU id {gpu_id} is out of range for visible CUDA devices (count={visible_count})."
            )
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    return gpu_ids


def choose_amp_dtype(device: torch.device, amp_dtype: str):
    amp_dtype = amp_dtype.lower()
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "fp32":
        return torch.float32
    raise ValueError("amp_dtype must be one of: fp16, bf16, fp32")


def update_model_args_from_checkpoint(args: Args, ckpt_obj: dict) -> None:
    ckpt_args = ckpt_obj.get("args", None)
    if not isinstance(ckpt_args, dict):
        return

    arch_keys = [
        "vocab_size",
        "d_model",
        "ff_mult",
        "n_heads",
        "n_layers",
        "dropout",
        "bucket_size",
        "n_hashes",
        "ff_chunk_size",
        "compressed_mem_len",
        "cross_attn_query_chunk_size",
        "cross_attn_min_chunk_size",
        "max_len",
        "pad_id",
        "bos_id",
        "eos_id",
    ]
    for key in arch_keys:
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])


def extract_state_dict(ckpt_obj: dict) -> dict:
    if "model_state" in ckpt_obj and isinstance(ckpt_obj["model_state"], dict):
        state = ckpt_obj["model_state"]
    elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        state = ckpt_obj["model_state_dict"]
    elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        state = ckpt_obj["model"]
    elif all(isinstance(k, str) for k in ckpt_obj.keys()):
        state = ckpt_obj
    else:
        raise ValueError("Cannot find model state in checkpoint.")

    has_module_prefix = any(k.startswith("module.") for k in state.keys())
    if has_module_prefix:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_best_model(ckpt_path: Path, args: Args, device: torch.device) -> ReformerSeq2Seq:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Unsupported checkpoint format.")

    update_model_args_from_checkpoint(args, ckpt_obj)
    model = ReformerSeq2Seq(args)
    state = extract_state_dict(ckpt_obj)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected[:8]}")

    model.to(device)
    model.eval()
    return model


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    vals, _ = torch.topk(logits, k, dim=-1)
    min_vals = vals[..., -1, None]
    return torch.where(logits < min_vals, torch.full_like(logits, -1e9), logits)


def maybe_cuda_autocast(enabled: bool, amp_dtype):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def greedy_decode(
    model: ReformerSeq2Seq,
    src_tokens: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    temperature: float,
    top_k: int,
    use_amp: bool,
    amp_dtype,
    decode_log_every: int,
) -> torch.Tensor:
    batch_size = src_tokens.size(0)
    device = src_tokens.device

    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    autocast_enabled = use_amp and (device.type == "cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    decode_t0 = time.time()

    with torch.no_grad():
        with maybe_cuda_autocast(autocast_enabled, amp_dtype):
            memory, compressed_memory, compressed_src_mask = model.encode(src_tokens=src_tokens, src_mask=src_mask)

        for _ in range(max_len):
            tgt_mask = generated.ne(model.args.pad_id)

            with maybe_cuda_autocast(autocast_enabled, amp_dtype):
                logits = model.decode_with_memory(
                    tgt_input_tokens=generated,
                    tgt_mask=tgt_mask,
                    memory=memory,
                    src_mask=src_mask,
                    compressed_memory=compressed_memory,
                    compressed_src_mask=compressed_src_mask,
                )

            next_logits = logits[:, -1, :]
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            next_logits = next_logits / temperature
            next_logits = top_k_logits(next_logits, top_k)

            next_token = torch.argmax(next_logits, dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)

            finished = finished | next_token.eq(eos_id)
            if decode_log_every > 0 and (generated.size(1) % decode_log_every == 0 or finished.all()):
                elapsed = max(1e-6, time.time() - decode_t0)
                tps = float(generated.numel()) / elapsed
                print(
                    f"[INFO] decode_progress tokens={generated.size(1)-1}/{max_len} "
                    f"finished={int(finished.sum().item())}/{batch_size} "
                    f"tokens_per_sec={tps:.2f}"
                )
            if finished.all():
                break

    return generated[:, 1:]


def warn_if_high_risk(args: Args, device: torch.device) -> None:
    risk_msgs: List[str] = []
    if args.max_input_len > 8192:
        risk_msgs.append(f"max_input_len={args.max_input_len} is high")
    if args.gen_max_len > 4096:
        risk_msgs.append(f"gen_max_len={args.gen_max_len} is high")
    if args.batch_size > 1:
        risk_msgs.append(f"batch_size={args.batch_size} may increase peak memory")
    if args.decode_log_every <= 0:
        risk_msgs.append("decode_log_every<=0 disables in-loop progress logs")
    if device.type == "cpu" and args.max_input_len >= 4096:
        risk_msgs.append("CPU inference with long sequences can be very slow")

    if risk_msgs:
        print("[WARN] Potential memory/time risks:")
        for msg in risk_msgs:
            print(f"  - {msg}")


def parse_args() -> Args:
    defaults = Args()
    parser = argparse.ArgumentParser(description="Standalone inference for Reformer Seq2Seq checkpoint.")

    parser.add_argument("--ckpt_path", type=str, default=defaults.ckpt_path)
    parser.add_argument("--input_file", type=str, default=defaults.input_file)
    parser.add_argument("--output_path", type=str, default=defaults.output_path)
    parser.add_argument("--files_on", type=parse_bool_arg, default=defaults.files_on)
    parser.add_argument("--files_path", type=str, default=defaults.files_path)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--pin_memory", action="store_true", default=defaults.pin_memory)

    parser.add_argument("--use_amp", action="store_true", default=defaults.use_amp)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")
    parser.add_argument("--amp_dtype", type=str, default=defaults.amp_dtype)
    parser.add_argument("--max_input_len", type=int, default=defaults.max_input_len)
    parser.add_argument("--gen_max_len", type=int, default=defaults.gen_max_len)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top_k", type=int, default=defaults.top_k)
    parser.add_argument("--decode_log_every", type=int, default=defaults.decode_log_every)

    parser.add_argument("--pad_id", type=int, default=defaults.pad_id)
    parser.add_argument("--bos_id", type=int, default=defaults.bos_id)
    parser.add_argument("--eos_id", type=int, default=defaults.eos_id)

    # Optional overrides for architecture (normally restored from ckpt args)
    parser.add_argument("--vocab_size", type=int, default=defaults.vocab_size)
    parser.add_argument("--d_model", type=int, default=defaults.d_model)
    parser.add_argument("--ff_mult", type=int, default=defaults.ff_mult)
    parser.add_argument("--n_heads", type=int, default=defaults.n_heads)
    parser.add_argument("--n_layers", type=int, default=defaults.n_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--bucket_size", type=int, default=defaults.bucket_size)
    parser.add_argument("--n_hashes", type=int, default=defaults.n_hashes)
    parser.add_argument("--ff_chunk_size", type=int, default=defaults.ff_chunk_size)
    parser.add_argument("--compressed_mem_len", type=int, default=defaults.compressed_mem_len)
    parser.add_argument("--cross_attn_query_chunk_size", type=int, default=defaults.cross_attn_query_chunk_size)
    parser.add_argument("--cross_attn_min_chunk_size", type=int, default=defaults.cross_attn_min_chunk_size)
    parser.add_argument("--max_len", type=int, default=defaults.max_len)

    parsed = parser.parse_args()
    return Args(**vars(parsed))


def run_inference_on_file(
    model: ReformerSeq2Seq,
    args: Args,
    device: torch.device,
    amp_dtype,
    input_path: Path,
    output_file: Path,
) -> int:
    sequences = read_jsonl_array_line(input_path)

    ds = InferenceDataset(sequences=sequences, max_input_len=args.max_input_len, pad_id=args.pad_id)
    effective_pin_memory = bool(args.pin_memory and device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=(args.num_workers > 0),
        collate_fn=partial(collate_infer_batch, pad_id=args.pad_id),
    )

    n_samples = 0
    decode_time_total = 0.0
    with output_file.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, total=len(loader), desc=f"Inference:{input_path.name}", unit="batch"):
            src_tokens = batch["src_tokens"].to(device, non_blocking=effective_pin_memory)
            src_mask = batch["src_mask"].to(device, non_blocking=effective_pin_memory)

            batch_decode_t0 = time.time()
            pred_tokens = greedy_decode(
                model=model,
                src_tokens=src_tokens,
                src_mask=src_mask,
                max_len=args.gen_max_len,
                bos_id=args.bos_id,
                eos_id=args.eos_id,
                temperature=args.temperature,
                top_k=args.top_k,
                use_amp=args.use_amp,
                amp_dtype=amp_dtype,
                decode_log_every=args.decode_log_every,
            )
            batch_decode_dt = time.time() - batch_decode_t0
            decode_time_total += batch_decode_dt

            pred_tokens_cpu = pred_tokens.detach().cpu().tolist()
            for pred_ids in pred_tokens_cpu:
                if args.eos_id in pred_ids:
                    eos_pos = pred_ids.index(args.eos_id)
                    pred_ids = pred_ids[:eos_pos]
                f.write(json.dumps([pred_ids], ensure_ascii=False) + "\n")
                n_samples += 1

            token_count = int(pred_tokens.ne(args.pad_id).sum().item())
            print(
                f"[INFO] batch_done file={input_path.name} samples={len(pred_tokens_cpu)} "
                f"pred_tokens={token_count} decode_time={batch_decode_dt:.2f}s"
            )

    if n_samples > 0:
        print(f"[INFO] file={input_path.name} avg_decode_sec_per_sample={decode_time_total / n_samples:.2f}")
    print(f"[INFO] file_done file={input_path.name} samples={n_samples} output={output_file}")
    return n_samples


def split_files_round_robin(input_files: List[Path], n_shards: int) -> List[List[Path]]:
    shards: List[List[Path]] = [[] for _ in range(n_shards)]
    for i, p in enumerate(input_files):
        shards[i % n_shards].append(p)
    return shards


def multi_gpu_file_worker(
    worker_idx: int,
    gpu_id: int,
    assigned_files: List[str],
    args_dict: Dict,
    result_queue,
) -> None:
    try:
        args = Args(**args_dict)
        args.device = f"cuda:{gpu_id}"
        set_seed(int(args.seed) + int(worker_idx))

        device = choose_device(args.device)
        amp_dtype = choose_amp_dtype(device, args.amp_dtype)
        model = load_best_model(ckpt_path=Path(args.ckpt_path), args=args, device=device)

        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        local_samples = 0
        local_files = 0
        for file_str in assigned_files:
            input_path = Path(file_str)
            output_file = output_dir / input_path.name
            local_samples += run_inference_on_file(
                model=model,
                args=args,
                device=device,
                amp_dtype=amp_dtype,
                input_path=input_path,
                output_file=output_file,
            )
            local_files += 1

        result_queue.put(
            {
                "ok": True,
                "worker_idx": int(worker_idx),
                "gpu_id": int(gpu_id),
                "files": int(local_files),
                "samples": int(local_samples),
            }
        )
    except Exception as ex:
        result_queue.put(
            {
                "ok": False,
                "worker_idx": int(worker_idx),
                "gpu_id": int(gpu_id),
                "error": str(ex),
                "traceback": traceback.format_exc(),
            }
        )


def run_multi_gpu_file_inference(args: Args, input_files: List[Path], gpu_ids: List[int]) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU file inference requires CUDA, but CUDA is unavailable.")

    worker_count = min(len(input_files), len(gpu_ids))
    if worker_count <= 1:
        raise RuntimeError("Multi-GPU file inference requires at least 2 available GPUs and 2 input files.")

    file_shards = split_files_round_robin(input_files, worker_count)
    mp_ctx = torch.multiprocessing.get_context("spawn")
    queue = mp_ctx.Queue()
    procs = []

    for worker_idx in range(worker_count):
        gpu_id = int(gpu_ids[worker_idx])
        shard = [str(p) for p in file_shards[worker_idx]]
        proc = mp_ctx.Process(
            target=multi_gpu_file_worker,
            args=(worker_idx, gpu_id, shard, asdict(args), queue),
        )
        proc.start()
        procs.append(proc)

    results = []
    for _ in range(worker_count):
        results.append(queue.get())

    for proc in procs:
        proc.join()

    failures = [r for r in results if not bool(r.get("ok", False))]
    if len(failures) > 0:
        f0 = failures[0]
        raise RuntimeError(
            "Multi-GPU worker failed: "
            f"worker={f0.get('worker_idx')} gpu={f0.get('gpu_id')} error={f0.get('error')}\n"
            f"traceback:\n{f0.get('traceback', '')}"
        )

    total_samples = int(sum(int(r.get("samples", 0)) for r in results))
    for r in sorted(results, key=lambda x: int(x.get("worker_idx", 0))):
        print(
            f"[INFO] worker_done worker={r.get('worker_idx')} gpu={r.get('gpu_id')} "
            f"files={r.get('files')} samples={r.get('samples')}"
        )
    return total_samples


def run_inference(args: Args) -> Path:
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = resolve_input_files(args)
    selected_gpu_ids = resolve_gpu_ids_from_device(args.device)

    use_multi_gpu_files = bool(
        args.files_on
        and len(input_files) > 1
        and len(selected_gpu_ids) > 1
    )

    if len(selected_gpu_ids) > 0:
        device = torch.device(f"cuda:{selected_gpu_ids[0]}")
    else:
        device = choose_device(args.device)
    amp_dtype = choose_amp_dtype(device, args.amp_dtype)

    warn_if_high_risk(args, device)

    print("[INFO] Runtime args:")
    print(json.dumps(asdict(args), indent=2, ensure_ascii=False))
    print(f"[INFO] Device: {device}")
    print(f"[INFO] selected_gpu_ids={selected_gpu_ids}")
    print(f"[INFO] DataLoader workers={args.num_workers}, batch_size={args.batch_size}")
    print(f"[INFO] files_on={bool(args.files_on)}, num_inputs={len(input_files)}")
    print(
        f"[INFO] visible_cuda_devices={torch.cuda.device_count()} "
        f"multi_gpu_file_mode={use_multi_gpu_files}"
    )

    t0 = time.time()
    n_samples = 0
    last_output_path: Path = output_dir / input_files[-1].name

    try:
        if use_multi_gpu_files:
            n_samples = run_multi_gpu_file_inference(args=args, input_files=input_files, gpu_ids=selected_gpu_ids)
        else:
            model = load_best_model(ckpt_path=ckpt_path, args=args, device=device)
            for input_path in input_files:
                this_output = output_dir / input_path.name
                last_output_path = this_output
                n_samples += run_inference_on_file(
                    model=model,
                    args=args,
                    device=device,
                    amp_dtype=amp_dtype,
                    input_path=input_path,
                    output_file=this_output,
                )
    except RuntimeError as ex:
        if "out of memory" in str(ex).lower() and device.type == "cuda":
            raise RuntimeError(
                "CUDA OOM during inference. Try lowering --gen_max_len, --max_input_len, or --batch_size, "
                "or use --amp_dtype bf16/fp16."
            ) from ex
        raise

    elapsed = time.time() - t0
    print(f"[INFO] Done. files={len(input_files)}, samples={n_samples}, elapsed={elapsed:.2f}s")
    print(f"[INFO] Output directory: {output_dir}")
    return last_output_path


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
