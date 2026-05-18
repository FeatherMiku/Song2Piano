import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import argparse
import json
import math
import time
import random
import socket
import traceback
from functools import partial
from datetime import timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# 设置随机种子，统一 Python/NumPy/PyTorch 的随机性以保证复现。
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 统计模型中参与训练的参数总量，用于评估模型规模。
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class Args:
    # Paths
    data_root: str = "data"  # 数据根目录，存放输入输出与切分后的数据文件
    input_dir: str = "input"  # 输入序列子目录名，用于拼接 data_root 后定位输入数据
    output_dir: str = "output"  # 输出序列子目录名，用于拼接 data_root 后定位目标数据
    ckpt_dir: str = "checkpoints7"  # 模型权重与训练状态保存目录
    enable_finetune: bool = False  # 是否启用二次训练（从已有 checkpoint 继续训练）
    finetune_model_path: str = r"testdata/testbest2.pt"  # 二次训练时加载的初始模型路径

    # Data
    train_ratio: float = 0.8  # 训练集划分比例
    val_ratio: float = 0.1  # 验证集划分比例
    test_ratio: float = 0.1  # 测试集划分比例
    max_len: int = 32768  # 序列最大长度，超过会被截断以控制显存和计算量
    batch_size: int = 4  # 每个 step 的样本数（每卡/每进程）
    num_workers: int = 16  # DataLoader 子进程数量，影响数据读取吞吐
    use_dynamic_padding: bool = True  # 是否按 batch 内最大长度动态补齐，减少无效 padding
    use_length_bucketing: bool = True  # 是否按长度分桶组 batch，降低长度差异带来的浪费
    bucket_size_multiplier: int = 50  # 分桶粒度倍率，值越大同桶长度范围越宽

    # Tokens
    pad_id: int = 1024  # PAD 标记 ID，用于补齐短序列
    bos_id: int = 1025  # BOS 标记 ID，表示序列起始
    eos_id: int = 1026  # EOS 标记 ID，表示序列结束
    vocab_size: int = 1027  # 词表大小，用于 embedding 与输出层维度

    # Model
    d_model: int = 256  # 模型隐藏维度，决定表示能力与计算开销
    ff_mult: int = 4  # 前馈层扩张倍数，FFN 维度约为 d_model * ff_mult
    n_heads: int = 8  # 多头注意力头数，影响注意力分解粒度
    n_layers: int = 6  # 编解码堆叠层数，决定模型深度
    dropout: float = 0.0  # dropout 概率，用于正则化防止过拟合
    bucket_size: int = 64  # Reformer LSH 注意力桶大小
    n_hashes: int = 4  # Reformer LSH 哈希轮数，影响召回与速度
    ff_chunk_size: int = 256  # 前馈网络分块计算大小，减小峰值显存
    compressed_mem_len: int = 768  # 压缩记忆长度，提供更长上下文信息
    cross_attn_query_chunk_size: int = 128  # 交叉注意力 query 分块大小，平衡速度与显存
    cross_attn_min_chunk_size: int = 32  # 交叉注意力最小分块大小，避免分块过小导致低效

    # Optimization
    epochs: int = 200  # 训练总轮数
    lr: float = 1e-3  # 初始学习率
    weight_decay: float = 0.01  # 权重衰减系数，抑制过拟合
    grad_clip: float = 1.0  # 梯度裁剪阈值，防止梯度爆炸
    label_smoothing: float = 0.0  # 标签平滑系数，提升泛化并缓解过拟合
    early_stop_patience: int = 20  # 早停耐心值，验证集长期无提升时停止训练
    max_oom_retries_per_batch: int = 2  # 单个 batch 触发 OOM 后允许重试的最大次数
    oom_retry_shrink_factor: float = 0.75  # OOM 重试时序列长度缩减比例
    oom_min_seq_len: int = 256  # OOM 重试可缩减到的最小序列长度下限
    skip_batch_on_oom: bool = True  # 多次 OOM 后是否跳过该 batch 继续训练

    # Runtime
    seed: int = 39  # 随机种子，保证可复现性
    use_amp: bool = False  # 是否启用自动混合精度训练
    amp_dtype: str = "bf16"  # AMP 使用的数据类型（如 bf16/fp16）
    enable_tf32: bool = True  # 是否启用 TF32 以提升 Ampere+ GPU 矩阵运算速度
    persistent_workers: bool = True  # DataLoader worker 是否常驻，减少每轮重建开销
    prefetch_factor: int = 4  # 每个 worker 预取 batch 数，提升数据管线吞吐
    validate_batch_contract: bool = False  # 是否对 batch 数据契约做额外校验（调试用）
    use_gradient_checkpointing: bool = True  # 是否启用梯度检查点以省显存
    gpu_ids: str = "0, 1, 2, 3"  # 可见 GPU 列表（逗号分隔）
    use_ddp: bool = True  # 是否启用 DDP 分布式训练
    ddp_backend: str = "nccl"  # DDP 通信后端（GPU 常用 nccl）
    ddp_timeout_minutes: int = 30  # DDP 进程组初始化与通信超时分钟数
    ddp_find_unused_parameters: bool = False  # DDP 是否查找未参与反传的参数
    ddp_broadcast_buffers: bool = False  # DDP 是否在进程间广播 buffers
    ddp_blocking_wait: bool = True  # NCCL 异常时是否阻塞等待，便于定位问题
    ddp_disable_ib: bool = True  # 是否禁用 InfiniBand 通信通道（网络兼容性开关）
    ddp_disable_p2p: bool = True  # 是否禁用 GPU 间 P2P 通信（兼容性开关）
    ddp_preflight: bool = True  # 是否在正式训练前做 DDP 预检
    ddp_preflight_timeout_minutes: int = 1  # DDP 预检超时时间（分钟）
    ddp_auto_fallback_gloo: bool = True  # NCCL 不可用时是否自动回退到 gloo
    save_every: int = 1  # 按 epoch 保存模型的间隔
    step_save_interval: int = 200  # 按 step 保存模型的间隔
    log_every: int = 20  # 训练日志打印间隔（step）


# 解析命令行中的布尔字符串参数，兼容 true/false、1/0 等写法。
def parse_bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# 从命令行构建训练配置，并处理微调开关与路径的合法性校验。
def build_args_from_cli() -> Args:
    parser = argparse.ArgumentParser(description="Train ReformerSeq2Seq model")
    parser.add_argument(
        "--enable_finetune",
        type=parse_bool_arg,
        default=None,
        help="Enable secondary training from an existing checkpoint (true/false).",
    )
    parser.add_argument(
        "--finetune_model_path",
        type=str,
        default=None,
        help="Path to a checkpoint generated by this training script.",
    )
    cli_args = parser.parse_args()

    args = Args()
    if cli_args.enable_finetune is not None:
        args.enable_finetune = bool(cli_args.enable_finetune)
    if cli_args.finetune_model_path is not None:
        args.finetune_model_path = str(cli_args.finetune_model_path).strip()

    if args.enable_finetune:
        if args.finetune_model_path == "":
            raise ValueError("enable_finetune=True requires finetune_model_path to be provided.")
    elif args.finetune_model_path:
        print("[INFO] finetune_model_path is set but enable_finetune is false, path will be ignored.")

    return args


# 规范并写入 CUDA_VISIBLE_DEVICES，限制当前进程可见的 GPU。
def configure_visible_gpus(gpu_ids: str) -> None:
    # Normalize user input like "0, 1" -> "0,1" to avoid ambiguous device parsing.
    parsed = [x.strip() for x in str(gpu_ids).split(",") if x.strip() != ""]
    if len(parsed) == 0:
        raise ValueError("gpu_ids is empty. Please provide at least one GPU id, e.g. '0' or '0,1'.")
    normalized = ",".join(parsed)
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized


# 根据环境返回训练设备，优先使用 CUDA。
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 配置 CUDA 运行时性能选项（TF32、cuDNN benchmark 等）。
def configure_runtime_performance(args: Args) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.enable_tf32)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


# 根据配置选择 autocast 的混合精度数据类型。
def get_autocast_dtype(args: Args) -> torch.dtype:
    amp_dtype = str(getattr(args, "amp_dtype", "bf16")).lower()
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    return torch.float32


# 判断分布式进程组是否已正确初始化。
def ddp_is_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# 按配置设置 NCCL 相关环境变量，提升 DDP 稳定性与兼容性。
def configure_ddp_backend_env(args: Args) -> None:
    # Follow modern torch env names and single-node defaults used by robust projects.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1" if bool(args.ddp_blocking_wait) else "0")
    if bool(args.ddp_disable_ib):
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if bool(args.ddp_disable_p2p):
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")


# 初始化分布式进程组，按 backend/rank/world_size 建立通信。
def setup_process_group(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=max(1, int(timeout_minutes))),
        init_method="env://",
    )


# 在本机申请一个空闲端口，供 DDP MASTER_PORT 使用。
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


# 安全销毁分布式进程组，避免残留通信状态。
def cleanup_process_group() -> None:
    if ddp_is_ready():
        torch.distributed.destroy_process_group()


# 判断当前进程是否主进程（单卡时恒为主进程）。
def is_main_process() -> bool:
    if not ddp_is_ready():
        return True
    return torch.distributed.get_rank() == 0


# 在单卡或多卡场景下汇总 loss/acc/steps，返回全局均值指标。
def reduce_mean_stats(loss_sum: float, acc_sum: float, steps: int, device: torch.device) -> Dict[str, float]:
    if not ddp_is_ready():
        denom = max(1, steps)
        return {"loss": float(loss_sum / denom), "acc": float(acc_sum / denom)}

    stats = torch.tensor(
        [float(loss_sum), float(acc_sum), float(steps)],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    denom = max(1.0, float(stats[2].item()))
    return {
        "loss": float(stats[0].item() / denom),
        "acc": float(stats[1].item() / denom),
    }


# Data contract checks and loading

# 将原始 JSON 对象规范为整数序列，并校验结构与元素类型。
def _extract_int_sequence(raw_obj) -> List[int]:
    if isinstance(raw_obj, list) and len(raw_obj) == 1 and isinstance(raw_obj[0], list):
        raw_obj = raw_obj[0]
    if not isinstance(raw_obj, list):
        raise ValueError("Expected a list or nested single list.")

    seq = []
    for x in raw_obj:
        if isinstance(x, bool):
            raise ValueError("Boolean token is not allowed.")
        if isinstance(x, (int, np.integer)):
            seq.append(int(x))
        else:
            raise ValueError(f"Non-integer token found: {type(x)}")
    return seq


# 读取 jsonl 文件每一行的数组数据并解析为整数序列列表。
def read_jsonl_array_line(path: Path) -> List[List[int]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sequences = []
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
            seq = _extract_int_sequence(obj)
        except Exception as ex:
            raise ValueError(f"Parse error at {path}, line {i + 1}: {ex}")
        sequences.append(seq)
    return sequences


# 收集输入/输出目录同名的 jsonl 文件对，作为配对样本来源。
def _paired_jsonl_files(args: Args) -> List[Tuple[str, Path, Path]]:
    data_root = Path(args.data_root)
    in_dir = data_root / args.input_dir
    out_dir = data_root / args.output_dir

    if not in_dir.exists() or not out_dir.exists():
        raise FileNotFoundError(f"Missing data dirs: {in_dir} or {out_dir}")

    in_files = sorted([p for p in in_dir.glob("*.jsonl")])
    out_files = sorted([p for p in out_dir.glob("*.jsonl")])
    in_map = {p.name: p for p in in_files}
    out_map = {p.name: p for p in out_files}
    common = sorted(set(in_map.keys()) & set(out_map.keys()))

    print("=== Data pairing debug ===")
    print(f"input files: {len(in_map)}")
    print(f"output files: {len(out_map)}")
    print(f"matched files: {len(common)}")

    if len(common) == 0:
        raise RuntimeError("No paired jsonl filenames found in input/output.")

    return [(name, in_map[name], out_map[name]) for name in common]


# 读取文件中的非空行并去除首尾空白。
def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# 统计所有配对文件中的样本总数，并校验输入输出行数一致。
def _count_paired_samples(paired_files: List[Tuple[str, Path, Path]]) -> int:
    total = 0
    for i, (name, in_path, out_path) in enumerate(paired_files, start=1):
        in_lines = _read_nonempty_lines(in_path)
        out_lines = _read_nonempty_lines(out_path)
        if len(in_lines) != len(out_lines):
            raise ValueError(f"Line count mismatch for {name}: {len(in_lines)} vs {len(out_lines)}")
        total += len(in_lines)
        if i % 200 == 0:
            print(f"Counting samples progress: {i}/{len(paired_files)} files")
    return total


# 逐条迭代配对样本，按文件与行号解析出输入/输出整数序列。
def _iter_paired_samples(
    paired_files: List[Tuple[str, Path, Path]],
) -> Iterator[Tuple[str, List[int], List[int]]]:
    for i, (name, in_path, out_path) in enumerate(paired_files, start=1):
        in_lines = _read_nonempty_lines(in_path)
        out_lines = _read_nonempty_lines(out_path)
        if len(in_lines) != len(out_lines):
            raise ValueError(f"Line count mismatch for {name}: {len(in_lines)} vs {len(out_lines)}")

        for line_idx, (in_ln, out_ln) in enumerate(zip(in_lines, out_lines), start=1):
            try:
                s_in = _extract_int_sequence(json.loads(in_ln))
                s_out = _extract_int_sequence(json.loads(out_ln))
            except Exception as ex:
                raise ValueError(f"Parse error at {name}, line {line_idx}: {ex}")
            yield name, s_in, s_out

        if i % 200 == 0:
            print(f"Parsing samples progress: {i}/{len(paired_files)} files")


# 返回 train/val/test 三个 H5 切分文件的标准路径。
def get_h5_split_paths(args: Args) -> Dict[str, Path]:
    data_root = Path(args.data_root)
    return {
        "train": data_root / "train.h5",
        "val": data_root / "val.h5",
        "test": data_root / "test.h5",
    }


# 将配对 jsonl 数据随机切分并写入 train/val/test 三个 H5 数据集。
def build_h5_splits(args: Args) -> Tuple[Dict[str, Path], Dict[str, float]]:
    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    paired_files = _paired_jsonl_files(args)
    total_samples = _count_paired_samples(paired_files)
    if total_samples == 0:
        raise RuntimeError("No samples found in paired jsonl files.")

    idx = list(range(total_samples))
    rng = random.Random(args.seed)
    rng.shuffle(idx)

    n_train = int(total_samples * args.train_ratio)
    n_val = int(total_samples * args.val_ratio)
    n_test = total_samples - n_train - n_val

    split_assignment = np.full(total_samples, 2, dtype=np.uint8)
    split_assignment[np.array(idx[:n_train], dtype=np.int64)] = 0
    split_assignment[np.array(idx[n_train : n_train + n_val], dtype=np.int64)] = 1

    split_counts = {
        "train": int((split_assignment == 0).sum()),
        "val": int((split_assignment == 1).sum()),
        "test": int((split_assignment == 2).sum()),
    }
    if split_counts["train"] != n_train or split_counts["val"] != n_val or split_counts["test"] != n_test:
        raise RuntimeError("Unexpected split count mismatch while assigning train/val/test.")

    h5_paths = get_h5_split_paths(args)
    for p in h5_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    vlen_dtype = h5py.vlen_dtype(np.dtype("int32"))
    with h5py.File(h5_paths["train"], "w") as f_train, h5py.File(h5_paths["val"], "w") as f_val, h5py.File(
        h5_paths["test"], "w"
    ) as f_test:
        train_src = f_train.create_dataset("src", shape=(split_counts["train"],), dtype=vlen_dtype)
        train_tgt = f_train.create_dataset("tgt", shape=(split_counts["train"],), dtype=vlen_dtype)
        val_src = f_val.create_dataset("src", shape=(split_counts["val"],), dtype=vlen_dtype)
        val_tgt = f_val.create_dataset("tgt", shape=(split_counts["val"],), dtype=vlen_dtype)
        test_src = f_test.create_dataset("src", shape=(split_counts["test"],), dtype=vlen_dtype)
        test_tgt = f_test.create_dataset("tgt", shape=(split_counts["test"],), dtype=vlen_dtype)

        counters = {"train": 0, "val": 0, "test": 0}
        total_in_tokens = 0
        total_out_tokens = 0
        max_in_len = 0
        max_out_len = 0
        global_min = 10**9
        global_max = -10**9

        for sample_idx, (name, s_in, s_out) in enumerate(_iter_paired_samples(paired_files)):
            if len(s_in) == 0 or len(s_out) == 0:
                raise ValueError(f"Empty sequence found in {name}.")

            local_min = min(min(s_in), min(s_out))
            local_max = max(max(s_in), max(s_out))
            if local_min < 0 or local_max >= args.vocab_size:
                raise ValueError(
                    f"Token out of range in {name}. min={local_min}, max={local_max}, vocab={args.vocab_size}"
                )

            total_in_tokens += len(s_in)
            total_out_tokens += len(s_out)
            max_in_len = max(max_in_len, len(s_in))
            max_out_len = max(max_out_len, len(s_out))
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

            split_id = int(split_assignment[sample_idx])
            if split_id == 0:
                pos = counters["train"]
                train_src[pos] = np.asarray(s_in, dtype=np.int32)
                train_tgt[pos] = np.asarray(s_out, dtype=np.int32)
                counters["train"] += 1
            elif split_id == 1:
                pos = counters["val"]
                val_src[pos] = np.asarray(s_in, dtype=np.int32)
                val_tgt[pos] = np.asarray(s_out, dtype=np.int32)
                counters["val"] += 1
            else:
                pos = counters["test"]
                test_src[pos] = np.asarray(s_in, dtype=np.int32)
                test_tgt[pos] = np.asarray(s_out, dtype=np.int32)
                counters["test"] += 1

            if (sample_idx + 1) % 10000 == 0:
                print(f"Writing H5 progress: {sample_idx + 1}/{total_samples} samples")

        for split_name in ["train", "val", "test"]:
            if counters[split_name] != split_counts[split_name]:
                raise RuntimeError(
                    f"H5 write count mismatch for {split_name}: {counters[split_name]} vs {split_counts[split_name]}"
                )

        for f_h5, split_name in [(f_train, "train"), (f_val, "val"), (f_test, "test")]:
            f_h5.attrs["num_samples"] = split_counts[split_name]
            f_h5.attrs["seed"] = args.seed
            f_h5.attrs["vocab_size"] = args.vocab_size

    stats = {
        "num_files": float(len(paired_files)),
        "num_samples": float(total_samples),
        "total_in_tokens": float(total_in_tokens),
        "total_out_tokens": float(total_out_tokens),
        "max_in_len": float(max_in_len),
        "max_out_len": float(max_out_len),
        "global_min_token": float(global_min),
        "global_max_token": float(global_max),
        "train_samples": float(split_counts["train"]),
        "val_samples": float(split_counts["val"]),
        "test_samples": float(split_counts["test"]),
    }
    return h5_paths, stats


# 检查 train/val/test 三个 H5 切分文件是否全部存在。
def h5_splits_exist(args: Args) -> bool:
    h5_paths = get_h5_split_paths(args)
    return all(p.exists() for p in h5_paths.values())


# 读取已有 H5 切分文件并汇总各 split 样本数量。
def summarize_h5_splits(args: Args) -> Dict[str, float]:
    h5_paths = get_h5_split_paths(args)
    split_sizes = {}
    for split_name, path in h5_paths.items():
        with h5py.File(path, "r") as f:
            if "src" not in f or "tgt" not in f:
                raise KeyError(f"Invalid h5 split file (missing src/tgt): {path}")
            if f["src"].shape[0] != f["tgt"].shape[0]:
                raise ValueError(f"Invalid h5 split file (src/tgt size mismatch): {path}")
            split_sizes[split_name] = f["src"].shape[0]
    return {
        "train_samples": float(split_sizes["train"]),
        "val_samples": float(split_sizes["val"]),
        "test_samples": float(split_sizes["test"]),
        "num_samples": float(split_sizes["train"] + split_sizes["val"] + split_sizes["test"]),
    }


# 若 H5 切分已存在则复用，否则从原始 jsonl 重新构建。
def ensure_h5_splits(args: Args) -> Tuple[Dict[str, Path], Dict[str, float]]:
    h5_paths = get_h5_split_paths(args)
    if h5_splits_exist(args):
        print("Found existing H5 splits. Reusing train/val/test h5 files.")
        return h5_paths, summarize_h5_splits(args)

    print("H5 splits not found. Building train/val/test h5 from jsonl files...")
    return build_h5_splits(args)


# 直接从 jsonl 收集并校验全部样本对，同时统计数据分布信息。
def collect_pairs(args: Args) -> Tuple[List[Tuple[str, List[int], List[int]]], Dict[str, float]]:
    data_root = Path(args.data_root)
    in_dir = data_root / args.input_dir
    out_dir = data_root / args.output_dir

    if not in_dir.exists() or not out_dir.exists():
        raise FileNotFoundError(f"Missing data dirs: {in_dir} or {out_dir}")

    in_files = sorted([p for p in in_dir.glob("*.jsonl")])
    out_files = sorted([p for p in out_dir.glob("*.jsonl")])
    in_map = {p.name: p for p in in_files}
    out_map = {p.name: p for p in out_files}
    common = sorted(set(in_map.keys()) & set(out_map.keys()))

    print("=== Data pairing debug ===")
    print(f"input files: {list(in_map.keys())}")
    print(f"output files: {list(out_map.keys())}")
    print(f"matched files: {len(common)}")

    if len(common) == 0:
        raise RuntimeError("No paired jsonl filenames found in input/output.")

    pairs = []
    total_in_tokens = 0
    total_out_tokens = 0
    global_min = 10**9
    global_max = -10**9
    max_in_len = 0
    max_out_len = 0

    for name in common:
        in_seqs = read_jsonl_array_line(in_map[name])
        out_seqs = read_jsonl_array_line(out_map[name])
        if len(in_seqs) != len(out_seqs):
            raise ValueError(f"Line count mismatch for {name}: {len(in_seqs)} vs {len(out_seqs)}")

        for s_in, s_out in zip(in_seqs, out_seqs):
            if len(s_in) == 0 or len(s_out) == 0:
                raise ValueError(f"Empty sequence found in {name}.")

            total_in_tokens += len(s_in)
            total_out_tokens += len(s_out)
            max_in_len = max(max_in_len, len(s_in))
            max_out_len = max(max_out_len, len(s_out))
            local_min = min(min(s_in), min(s_out))
            local_max = max(max(s_in), max(s_out))
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

            if local_min < 0 or local_max >= args.vocab_size:
                raise ValueError(
                    f"Token out of range in {name}. min={local_min}, max={local_max}, vocab={args.vocab_size}"
                )

            pairs.append((name, s_in, s_out))

    stats = {
        "num_files": float(len(common)),
        "num_samples": float(len(pairs)),
        "total_in_tokens": float(total_in_tokens),
        "total_out_tokens": float(total_out_tokens),
        "max_in_len": float(max_in_len),
        "max_out_len": float(max_out_len),
        "global_min_token": float(global_min),
        "global_max_token": float(global_max),
    }
    return pairs, stats


# 按给定比例与随机种子将样本对切分为 train/val/test。
def split_pairs(
    pairs: List[Tuple[str, List[int], List[int]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val : n_train + n_val + n_test]

    train_set = [pairs[i] for i in train_idx]
    val_set = [pairs[i] for i in val_idx]
    test_set = [pairs[i] for i in test_idx]
    return train_set, val_set, test_set


# Dataset and dataloaders

# 将序列截断到最大长度或用 pad_id 补齐到固定长度。
def pad_or_truncate(seq: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))


class Seq2SeqTokenDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, List[int], List[int]]],
        max_len: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ):
        self.samples = samples
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.samples)

    def get_sequence_lengths(self) -> List[int]:
        return [min(self.max_len, len(src)) for _, src, _ in self.samples]

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        _, src, tgt = self.samples[idx]

        src_seq = src[: self.max_len]
        tgt_for_label = tgt[: self.max_len - 1] + [self.eos_id]
        tgt_input_seq = [self.bos_id] + tgt_for_label[:-1]

        return {
            "src_tokens": src_seq,
            "tgt_input_tokens": tgt_input_seq,
            "tgt_labels": tgt_for_label,
        }


class H5Seq2SeqTokenDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        max_len: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
    ):
        self.h5_path = str(h5_path)
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self._h5_file: Optional[h5py.File] = None
        self._src = None
        self._tgt = None

        with h5py.File(self.h5_path, "r") as f:
            if "src" not in f or "tgt" not in f:
                raise KeyError(f"Invalid h5 split file (missing src/tgt): {self.h5_path}")
            if f["src"].shape[0] != f["tgt"].shape[0]:
                raise ValueError(f"Invalid h5 split file (src/tgt size mismatch): {self.h5_path}")
            self.length = int(f["src"].shape[0])

    def _ensure_open(self) -> None:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._src = self._h5_file["src"]
            self._tgt = self._h5_file["tgt"]

    def __len__(self) -> int:
        return self.length

    def get_sequence_lengths(self) -> List[int]:
        self._ensure_open()
        return [min(self.max_len, len(self._src[i])) for i in range(self.length)]

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        self._ensure_open()
        src = np.asarray(self._src[idx], dtype=np.int64)
        tgt = np.asarray(self._tgt[idx], dtype=np.int64)

        src_seq = src[: self.max_len]
        tgt_core = tgt[: self.max_len - 1]
        tgt_for_label = np.concatenate([tgt_core, np.asarray([self.eos_id], dtype=np.int64)], axis=0)
        tgt_input_seq = np.concatenate([np.asarray([self.bos_id], dtype=np.int64), tgt_for_label[:-1]], axis=0)

        return {
            "src_tokens": src_seq,
            "tgt_input_tokens": tgt_input_seq,
            "tgt_labels": tgt_for_label,
        }

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# 对一个 batch 做结构、形状、类型与 token 范围校验，确保训练输入合法。
def assert_batch_contract(batch: Dict[str, torch.Tensor], args: Args) -> None:
    required = ["src_tokens", "src_mask", "tgt_input_tokens", "tgt_labels", "tgt_mask"]
    for key in required:
        if key not in batch:
            raise KeyError(f"Missing key in batch: {key}")

    if batch["src_tokens"].ndim != 2 or batch["src_mask"].ndim != 2:
        raise ValueError("src_tokens/src_mask must be 2D tensors")
    if batch["tgt_input_tokens"].ndim != 2 or batch["tgt_labels"].ndim != 2 or batch["tgt_mask"].ndim != 2:
        raise ValueError("tgt_input_tokens/tgt_labels/tgt_mask must be 2D tensors")

    b = batch["src_tokens"].shape[0]
    src_l = batch["src_tokens"].shape[1]
    tgt_l = batch["tgt_input_tokens"].shape[1]

    if src_l > args.max_len:
        raise ValueError(f"src_tokens length exceeds max_len: {src_l} > {args.max_len}")
    if tgt_l > args.max_len:
        raise ValueError(f"tgt_tokens length exceeds max_len: {tgt_l} > {args.max_len}")

    if batch["src_mask"].shape != (b, src_l):
        raise ValueError("src_mask shape mismatch")
    if batch["tgt_input_tokens"].shape != (b, tgt_l):
        raise ValueError("tgt_input_tokens shape mismatch")
    if batch["tgt_labels"].shape != (b, tgt_l):
        raise ValueError("tgt_labels shape mismatch")
    if batch["tgt_mask"].shape != (b, tgt_l):
        raise ValueError("tgt_mask shape mismatch")
    if batch["src_tokens"].dtype != torch.long:
        raise TypeError("src_tokens must be torch.long")
    if batch["tgt_input_tokens"].dtype != torch.long:
        raise TypeError("tgt_input_tokens must be torch.long")
    if batch["tgt_labels"].dtype != torch.long:
        raise TypeError("tgt_labels must be torch.long")

    token_min = min(
        int(batch["src_tokens"].min().item()),
        int(batch["tgt_input_tokens"].min().item()),
        int(batch["tgt_labels"].min().item()),
    )
    token_max = max(
        int(batch["src_tokens"].max().item()),
        int(batch["tgt_input_tokens"].max().item()),
        int(batch["tgt_labels"].max().item()),
    )
    if token_min < 0 or token_max >= args.vocab_size:
        raise ValueError(f"Token out of bounds in batch: min={token_min}, max={token_max}")


# 将样本列表动态补齐并组装为张量 batch（含 tokens 与 mask）。
def collate_dynamic_batch(
    samples: List[Dict[str, List[int]]],
    pad_id: int,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    src_list = [s["src_tokens"] for s in samples]
    tgt_in_list = [s["tgt_input_tokens"] for s in samples]
    tgt_lbl_list = [s["tgt_labels"] for s in samples]

    src_len = min(max_len, max(len(s) for s in src_list))
    tgt_len = min(max_len, max(len(s) for s in tgt_in_list))
    batch_size = len(samples)

    src_tokens = torch.full((batch_size, src_len), pad_id, dtype=torch.long)
    src_mask = torch.zeros((batch_size, src_len), dtype=torch.bool)
    tgt_input_tokens = torch.full((batch_size, tgt_len), pad_id, dtype=torch.long)
    tgt_labels = torch.full((batch_size, tgt_len), pad_id, dtype=torch.long)
    tgt_mask = torch.zeros((batch_size, tgt_len), dtype=torch.bool)

    for i, (src, tgt_in, tgt_lbl) in enumerate(zip(src_list, tgt_in_list, tgt_lbl_list)):
        src_trim = torch.as_tensor(src[:src_len], dtype=torch.long)
        tgt_in_trim = torch.as_tensor(tgt_in[:tgt_len], dtype=torch.long)
        tgt_lbl_trim = torch.as_tensor(tgt_lbl[:tgt_len], dtype=torch.long)

        src_n = src_trim.numel()
        tgt_n = tgt_in_trim.numel()

        src_tokens[i, :src_n] = src_trim
        src_mask[i, :src_n] = True
        tgt_input_tokens[i, :tgt_n] = tgt_in_trim
        tgt_labels[i, :tgt_n] = tgt_lbl_trim
        tgt_mask[i, :tgt_n] = True

    return {
        "src_tokens": src_tokens,
        "src_mask": src_mask,
        "tgt_input_tokens": tgt_input_tokens,
        "tgt_labels": tgt_labels,
        "tgt_mask": tgt_mask,
    }


# 构建训练/验证/测试 DataLoader，支持 H5 数据源、长度分桶与 DDP 采样。
def make_dataloaders(
    args: Args,
    train_pairs: Optional[List[Tuple[str, List[int], List[int]]]] = None,
    val_pairs: Optional[List[Tuple[str, List[int], List[int]]]] = None,
    test_pairs: Optional[List[Tuple[str, List[int], List[int]]]] = None,
    train_h5_path: Optional[Path] = None,
    val_h5_path: Optional[Path] = None,
    test_h5_path: Optional[Path] = None,
    rank: int = 0,
    world_size: int = 1,
):
    class LengthBucketBatchSampler(BatchSampler):
        def __init__(
            self,
            lengths: List[int],
            batch_size: int,
            drop_last: bool,
            shuffle: bool,
            bucket_size_multiplier: int,
        ):
            self.lengths = np.asarray(lengths, dtype=np.int32)
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.bucket_size = max(batch_size, batch_size * max(1, bucket_size_multiplier))

        def __iter__(self):
            n = len(self.lengths)
            indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(indices)

            pooled = []
            for start in range(0, n, self.bucket_size):
                bucket = indices[start : start + self.bucket_size]
                order = np.argsort(self.lengths[bucket], kind="stable")
                pooled.extend(bucket[order].tolist())

            batches = [pooled[i : i + self.batch_size] for i in range(0, len(pooled), self.batch_size)]
            if self.drop_last and len(batches) > 0 and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

        def __len__(self) -> int:
            n = len(self.lengths)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size

    if train_h5_path is not None and val_h5_path is not None and test_h5_path is not None:
        train_ds = H5Seq2SeqTokenDataset(train_h5_path, args.max_len, args.pad_id, args.bos_id, args.eos_id)
        val_ds = H5Seq2SeqTokenDataset(val_h5_path, args.max_len, args.pad_id, args.bos_id, args.eos_id)
        test_ds = H5Seq2SeqTokenDataset(test_h5_path, args.max_len, args.pad_id, args.bos_id, args.eos_id)
    else:
        if train_pairs is None or val_pairs is None or test_pairs is None:
            raise ValueError("Either h5 paths or all pair lists must be provided.")
        train_ds = Seq2SeqTokenDataset(train_pairs, args.max_len, args.pad_id, args.bos_id, args.eos_id)
        val_ds = Seq2SeqTokenDataset(val_pairs, args.max_len, args.pad_id, args.bos_id, args.eos_id)
        test_ds = Seq2SeqTokenDataset(test_pairs, args.max_len, args.pad_id, args.bos_id, args.eos_id)

    train_sampler = None
    val_sampler = None
    test_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    collate_fn = partial(collate_dynamic_batch, pad_id=args.pad_id, max_len=args.max_len)
    effective_num_workers = int(args.num_workers)
    if world_size > 1:
        effective_num_workers = max(1, int(args.num_workers) // int(world_size))

    loader_extra_kwargs = {}
    if effective_num_workers > 0:
        if args.persistent_workers:
            loader_extra_kwargs["persistent_workers"] = True
        if args.prefetch_factor > 0:
            loader_extra_kwargs["prefetch_factor"] = args.prefetch_factor

    if world_size == 1 and args.use_length_bucketing:
        train_lengths = train_ds.get_sequence_lengths()
        train_batch_sampler = LengthBucketBatchSampler(
            lengths=train_lengths,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            bucket_size_multiplier=args.bucket_size_multiplier,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=effective_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            **loader_extra_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=effective_num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_fn,
            **loader_extra_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        **loader_extra_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        **loader_extra_kwargs,
    )

    return train_loader, val_loader, test_loader, train_sampler


# Reformer-style modules

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


# 将长度向上取整到 bucket 的整数倍，便于分桶注意力对齐。
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


# Encoder-decoder model (Reformer-style)

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
        self.use_gradient_checkpointing = bool(args.use_gradient_checkpointing)

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
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer,
                    x,
                    tgt_mask,
                    memory,
                    src_mask,
                    compressed_memory,
                    compressed_src_mask,
                    use_reentrant=False,
                )
            else:
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

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_input_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_h = self.embed(src_tokens)
        memory = self.encoder(src_h, src_mask)
        compressed_memory, compressed_src_mask = self.memory_compressor(memory, src_mask)

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


# Loss, metrics, optimization helpers

# 计算带可选标签平滑的交叉熵损失，并忽略 pad 位置。
def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    smoothing: float,
) -> torch.Tensor:
    vocab = logits.size(-1)
    logits_flat = logits.view(-1, vocab)
    targets_flat = targets.view(-1)

    if smoothing <= 0.0:
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)

    with torch.no_grad():
        true_dist = torch.zeros_like(logits_flat)
        true_dist.fill_(smoothing / (vocab - 1))
        ignore = targets_flat.eq(pad_id)
        true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)
        true_dist[ignore] = 0

    log_probs = F.log_softmax(logits_flat, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1)
    valid = ~targets_flat.eq(pad_id)
    return loss[valid].mean()


# 计算非 pad token 的准确率，作为训练/验证指标。
@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    pred = logits.argmax(dim=-1)
    valid = targets.ne(pad_id)
    correct = pred.eq(targets) & valid
    denom = valid.sum().item()
    return correct.sum().item() / max(1, denom)


# 将 batch 中各张量搬运到目标设备，使用 non_blocking 提升吞吐。
def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# 发生 OOM 时缩短序列长度并裁剪 batch，生成重试用更小输入。
def shrink_batch_for_oom(batch: Dict[str, torch.Tensor], args: Args) -> Optional[Dict[str, torch.Tensor]]:
    src_l = batch["src_tokens"].shape[1]
    tgt_l = batch["tgt_input_tokens"].shape[1]

    min_len = max(2, int(args.oom_min_seq_len))
    factor = float(args.oom_retry_shrink_factor)
    factor = min(0.95, max(0.1, factor))

    new_src_l = max(min_len, int(src_l * factor))
    new_tgt_l = max(min_len, int(tgt_l * factor))
    if new_src_l >= src_l:
        new_src_l = src_l - 1
    if new_tgt_l >= tgt_l:
        new_tgt_l = tgt_l - 1
    if new_src_l < 2 or new_tgt_l < 2:
        return None

    sliced = dict(batch)
    sliced["src_tokens"] = batch["src_tokens"][:, :new_src_l].contiguous()
    sliced["src_mask"] = batch["src_mask"][:, :new_src_l].contiguous()
    sliced["tgt_input_tokens"] = batch["tgt_input_tokens"][:, :new_tgt_l].contiguous()
    sliced["tgt_labels"] = batch["tgt_labels"][:, :new_tgt_l].contiguous()
    sliced["tgt_mask"] = batch["tgt_mask"][:, :new_tgt_l].contiguous()
    return sliced


# 根据训练参数构建 AdamW 优化器。
def build_optimizer(model: nn.Module, args: Args):
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# 根据总步数构建余弦退火学习率调度器。
def build_scheduler(optimizer, total_steps: int):
    if total_steps <= 0:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


# Train and validation loops

# 执行一个训练 epoch，包含 AMP、OOM 重试、梯度更新与日志回调。
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    args: Args,
    scheduler=None,
    on_step_end=None,
) -> Dict[str, float]:
    model.train()
    log_on_main = is_main_process()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    skipped_oom = 0
    amp_dtype = get_autocast_dtype(args)

    for step, batch in enumerate(loader):
        if args.validate_batch_contract:
            assert_batch_contract(batch, args)
        batch = move_batch_to_device(batch, device)

        oom_retry = 0
        step_done = False
        while not step_done:
            logits = None
            loss = None
            try:
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=(args.use_amp and device.type == "cuda"),
                ):
                    logits = model(
                        src_tokens=batch["src_tokens"],
                        src_mask=batch["src_mask"],
                        tgt_input_tokens=batch["tgt_input_tokens"],
                        tgt_mask=batch["tgt_mask"],
                    )
                    loss = cross_entropy_with_label_smoothing(
                        logits,
                        batch["tgt_labels"],
                        pad_id=args.pad_id,
                        smoothing=args.label_smoothing,
                    )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                acc = token_accuracy(logits.detach(), batch["tgt_labels"], args.pad_id)
                total_loss += loss.item()
                total_acc += acc
                steps += 1
                step_done = True

                if on_step_end is not None:
                    on_step_end(
                        steps,
                        float(loss.item()),
                        float(acc),
                        float(total_loss / max(1, steps)),
                        float(total_acc / max(1, steps)),
                    )

                if step % args.log_every == 0:
                    if log_on_main:
                        print(f"train step={step} loss={loss.item():.4f} acc={acc:.4f}")
            except torch.OutOfMemoryError:
                if logits is not None:
                    del logits
                if loss is not None:
                    del loss
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if oom_retry < int(args.max_oom_retries_per_batch):
                    shrunk = shrink_batch_for_oom(batch, args)
                    if shrunk is not None:
                        oom_retry += 1
                        batch = shrunk
                        if log_on_main:
                            print(
                                f"[OOM] train step={step} retry={oom_retry} "
                                f"with src_len={batch['src_tokens'].shape[1]} tgt_len={batch['tgt_input_tokens'].shape[1]}"
                            )
                        continue

                if args.skip_batch_on_oom:
                    skipped_oom += 1
                    step_done = True
                    if log_on_main:
                        print(f"[OOM] train step={step} skipped after retries={oom_retry}")
                else:
                    raise

    if skipped_oom > 0:
        if log_on_main:
            print(f"[OOM] skipped batches this epoch: {skipped_oom}")

    return reduce_mean_stats(total_loss, total_acc, steps, device)


# 在验证集上评估模型，返回平均 loss 与准确率。
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: Args,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    amp_dtype = get_autocast_dtype(args)

    for batch in loader:
        if args.validate_batch_contract:
            assert_batch_contract(batch, args)
        batch = move_batch_to_device(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(args.use_amp and device.type == "cuda"),
        ):
            logits = model(
                src_tokens=batch["src_tokens"],
                src_mask=batch["src_mask"],
                tgt_input_tokens=batch["tgt_input_tokens"],
                tgt_mask=batch["tgt_mask"],
            )
            loss = cross_entropy_with_label_smoothing(
                logits,
                batch["tgt_labels"],
                pad_id=args.pad_id,
                smoothing=0.0,
            )
        acc = token_accuracy(logits, batch["tgt_labels"], args.pad_id)

        total_loss += loss.item()
        total_acc += acc
        steps += 1

    return reduce_mean_stats(total_loss, total_acc, steps, device)


# 保存训练检查点（模型参数、优化器状态、当前 epoch 与最佳指标）。
def save_checkpoint(model, optimizer, epoch: int, best_val_loss: float, path: Path, args: Args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": asdict(args),
        },
        path,
    )


# 在启用微调时加载已有 checkpoint 权重，支持宽松匹配并打印差异信息。
def maybe_load_finetune_weights(
    model: nn.Module,
    args: Args,
    device: torch.device,
    *,
    verbose: bool = True,
) -> None:
    if not bool(args.enable_finetune):
        return

    ckpt_path = Path(args.finetune_model_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Finetune checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location=device)
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        model_state = payload["model_state"]
        source_epoch = payload.get("epoch")
    elif isinstance(payload, dict):
        model_state = payload
        source_epoch = None
    else:
        raise ValueError(f"Unsupported checkpoint format for finetune: {ckpt_path}")

    load_result = model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"[Finetune] loaded checkpoint: {ckpt_path}")
        if source_epoch is not None:
            print(f"[Finetune] source checkpoint epoch: {source_epoch}")
        if len(load_result.missing_keys) > 0:
            print(f"[Finetune] missing keys: {len(load_result.missing_keys)}")
        if len(load_result.unexpected_keys) > 0:
            print(f"[Finetune] unexpected keys: {len(load_result.unexpected_keys)}")


# 创建本次训练的日志文件路径并确保日志目录存在。
def create_train_log_file() -> Path:
    log_dir = Path.cwd() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")
    return log_dir / f"train_log_{timestamp}.json"


# 将训练日志 payload 序列化写入 JSON 文件。
def write_train_log(log_path: Path, payload: Dict) -> None:
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# 构建训练日志初始结构（创建时间、参数快照、epoch 与 checkpoint 列表）。
def build_initial_train_log_payload(args: Args) -> Dict:
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": asdict(args),
        "epochs": [],
        "checkpoints": [],
    }


# 从磁盘读取日志并补齐缺省字段，若不存在或损坏则初始化新日志结构。
def load_or_init_train_log_payload(log_path: Path, args: Args) -> Dict:
    if log_path.exists():
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if "args" not in payload:
                    payload["args"] = asdict(args)
                if "epochs" not in payload or not isinstance(payload["epochs"], list):
                    payload["epochs"] = []
                if "checkpoints" not in payload or not isinstance(payload["checkpoints"], list):
                    payload["checkpoints"] = []
                if "created_at" not in payload:
                    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return payload
        except Exception:
            pass
    return build_initial_train_log_payload(args)


# 汇总训练结束统计信息（轮次、最佳验证损失、耗时、checkpoint 数等）。
def _build_run_end_summary(payload: Dict, run_duration_sec: Optional[float] = None) -> Dict:
    epochs_raw = payload.get("epochs", [])
    epochs = epochs_raw if isinstance(epochs_raw, list) else []

    summary: Dict[str, float] = {
        "num_logged_epochs": float(len(epochs)),
    }

    if run_duration_sec is not None:
        summary["duration_sec"] = round(float(run_duration_sec), 3)

    if len(epochs) == 0:
        summary["last_epoch"] = 0.0
        ckpts_raw = payload.get("checkpoints", [])
        ckpts = ckpts_raw if isinstance(ckpts_raw, list) else []
        summary["num_logged_checkpoints"] = float(len(ckpts))
        return summary

    last = epochs[-1] if isinstance(epochs[-1], dict) else {}
    if "epoch" in last:
        summary["last_epoch"] = float(last["epoch"])
    else:
        summary["last_epoch"] = float(len(epochs))

    for k in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        if k in last and isinstance(last[k], (int, float)):
            summary[f"last_{k}"] = float(last[k])

    val_losses = [
        float(ep["val_loss"])
        for ep in epochs
        if isinstance(ep, dict) and isinstance(ep.get("val_loss"), (int, float))
    ]
    if len(val_losses) > 0:
        summary["best_val_loss"] = float(min(val_losses))

    if run_duration_sec is None:
        epoch_secs = [
            float(ep["time_sec"])
            for ep in epochs
            if isinstance(ep, dict) and isinstance(ep.get("time_sec"), (int, float))
        ]
        if len(epoch_secs) > 0:
            summary["duration_sec"] = round(float(sum(epoch_secs)), 3)

    ckpts_raw = payload.get("checkpoints", [])
    ckpts = ckpts_raw if isinstance(ckpts_raw, list) else []
    summary["num_logged_checkpoints"] = float(len(ckpts))

    return summary


# 将结束描述与统计摘要拼接成可读的 run_end 明细字符串。
def _compose_run_end_detail(base_detail: str, summary: Dict) -> str:
    parts = [base_detail]

    if "duration_sec" in summary:
        parts.append(f"duration_sec={summary['duration_sec']}")
    if "last_epoch" in summary:
        parts.append(f"last_epoch={int(summary['last_epoch'])}")
    if "best_val_loss" in summary:
        parts.append(f"best_val_loss={summary['best_val_loss']:.6f}")
    if "last_val_loss" in summary:
        parts.append(f"last_val_loss={summary['last_val_loss']:.6f}")
    if "last_train_loss" in summary:
        parts.append(f"last_train_loss={summary['last_train_loss']:.6f}")
    if "last_val_acc" in summary:
        parts.append(f"last_val_acc={summary['last_val_acc']:.6f}")
    if "last_train_acc" in summary:
        parts.append(f"last_train_acc={summary['last_train_acc']:.6f}")
    if "num_logged_checkpoints" in summary:
        parts.append(f"num_logged_checkpoints={int(summary['num_logged_checkpoints'])}")

    return " | ".join(parts)


# 写入训练结束状态到日志（成功/中断/异常），并附带摘要与错误栈信息。
def finalize_train_log(
    log_path: Optional[Path],
    payload: Optional[Dict],
    status: str,
    detail: str,
    error: Optional[BaseException] = None,
    run_duration_sec: Optional[float] = None,
) -> None:
    if log_path is None:
        return

    data = load_or_init_train_log_payload(log_path, Args())
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "epochs" and isinstance(value, list):
                disk_epochs = data.get("epochs", [])
                if not isinstance(disk_epochs, list) or len(value) > len(disk_epochs):
                    data["epochs"] = value
            elif key == "checkpoints" and isinstance(value, list):
                disk_ckpts = data.get("checkpoints", [])
                if not isinstance(disk_ckpts, list) or len(value) > len(disk_ckpts):
                    data["checkpoints"] = value
            elif key != "run_end":
                data[key] = value

    summary = _build_run_end_summary(data, run_duration_sec=run_duration_sec)
    run_end = {
        "status": status,
        "detail": _compose_run_end_detail(detail, summary),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
    }
    if error is not None:
        run_end["error_type"] = type(error).__name__
        run_end["error_message"] = str(error)
        run_end["traceback"] = traceback.format_exc()

    data["run_end"] = run_end
    write_train_log(log_path, data)


# 训练主循环：执行多轮 train/val、保存 step/epoch/best checkpoint、处理早停与日志。
def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: Args,
    train_sampler=None,
    write_log: bool = True,
    log_path: Optional[Path] = None,
    log_payload: Optional[Dict] = None,
) -> Path:
    model.to(device)
    is_ddp = ddp_is_ready()
    main_process = is_main_process()

    if is_ddp and (args.max_oom_retries_per_batch > 0 or args.skip_batch_on_oom):
        if main_process:
            print(
                "[DDP] Disabling per-rank OOM retry/skip to avoid rank desync. "
                "Please reduce batch size or max_len if OOM persists."
            )
        args.max_oom_retries_per_batch = 0
        args.skip_batch_on_oom = False

    optimizer = build_optimizer(model, args)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps)
    scaler_enabled = bool(args.use_amp and device.type == "cuda")
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    best_val = float("inf")
    bad_epochs = 0

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    effective_write_log = bool(write_log and main_process)

    if effective_write_log:
        if log_path is None:
            log_path = create_train_log_file()
        if log_payload is None:
            log_payload = load_or_init_train_log_payload(log_path, args)
        if "epochs" not in log_payload or not isinstance(log_payload["epochs"], list):
            log_payload["epochs"] = []
        if "checkpoints" not in log_payload or not isinstance(log_payload["checkpoints"], list):
            log_payload["checkpoints"] = []
        if "args" not in log_payload:
            log_payload["args"] = asdict(args)
        if "created_at" not in log_payload:
            log_payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    log_records: List[Dict] = log_payload["epochs"] if (effective_write_log and log_path is not None and log_payload is not None) else []
    checkpoint_records: List[Dict] = (
        log_payload["checkpoints"] if (effective_write_log and log_path is not None and log_payload is not None) else []
    )

    def append_checkpoint_log(record: Dict) -> None:
        if not (effective_write_log and log_path is not None and log_payload is not None):
            return
        checkpoint_records.append(record)
        write_train_log(log_path, log_payload)

    if effective_write_log and log_path is not None and log_payload is not None:
        write_train_log(log_path, log_payload)
        print(f"Training log file: {log_path}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        step_ckpt_count = 0

        def maybe_save_step_checkpoint(
            step_in_epoch: int,
            train_step_loss: float,
            train_step_acc: float,
            train_avg_loss: float,
            train_avg_acc: float,
        ) -> None:
            nonlocal step_ckpt_count, best_val
            if args.step_save_interval <= 0:
                return
            if step_in_epoch % args.step_save_interval != 0:
                return

            # Keep all ranks on the same code path: run step validation on every rank.
            step_metrics = validate(model, val_loader, device, args)
            step_val = float(step_metrics["loss"])
            step_acc = float(step_metrics["acc"])
            model.train()

            best_before = float(best_val)
            is_new_best = step_val < best_val
            if step_val < best_val:
                best_val = step_val

            # Only rank0 performs I/O and logging, matching single-card style.
            if not main_process:
                return

            step_ckpt_count += 1
            step_path = ckpt_dir / f"step{epoch:03d}_{step_ckpt_count}.pt"

            # Save the periodic step checkpoint first.
            save_checkpoint(model, optimizer, epoch, best_before, step_path, args)

            # Evaluate this step checkpoint candidate for global best selection.
            if is_new_best:
                save_checkpoint(model, optimizer, epoch, best_val, best_path, args)

            append_checkpoint_log(
                {
                    "save_order": len(checkpoint_records) + 1,
                    "save_type": "step",
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "path": step_path.name,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "step_save_index": step_ckpt_count,
                    "train_step_loss": round(float(train_step_loss), 6),
                    "train_step_acc": round(float(train_step_acc), 6),
                    "train_avg_loss_so_far": round(float(train_avg_loss), 6),
                    "train_avg_acc_so_far": round(float(train_avg_acc), 6),
                    "val_loss": round(step_val, 6),
                    "val_acc": round(step_acc, 6),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "best_val_loss_before": round(best_before, 6) if math.isfinite(best_before) else None,
                    "best_val_loss_after": round(float(best_val), 6),
                    "is_new_best": bool(is_new_best),
                }
            )

            if is_new_best:
                append_checkpoint_log(
                    {
                        "save_order": len(checkpoint_records) + 1,
                        "save_type": "best",
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "path": best_path.name,
                        "epoch": epoch,
                        "trigger": "step",
                        "step_in_epoch": step_in_epoch,
                        "step_save_index": step_ckpt_count,
                        "val_loss": round(step_val, 6),
                        "val_acc": round(step_acc, 6),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "best_val_loss_after": round(float(best_val), 6),
                    }
                )

            print(
                f"saved step checkpoint: {step_path.name} "
                f"(epoch={epoch}, step={step_in_epoch}, idx={step_ckpt_count}, "
                f"train_step_loss={train_step_loss:.4f}, train_step_acc={train_step_acc:.4f}, "
                f"train_avg_loss={train_avg_loss:.4f}, train_avg_acc={train_avg_acc:.4f}, "
                f"val_loss={step_val:.4f}, val_acc={step_acc:.4f}, "
                f"best_before={best_before:.4f}, best_after={best_val:.4f}, is_new_best={is_new_best})"
            )

        t0 = time.time()
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            scheduler=scheduler,
            on_step_end=maybe_save_step_checkpoint,
        )
        va = validate(model, val_loader, device, args)
        dt = time.time() - t0

        if main_process:
            print(
                f"epoch={epoch} time={dt:.1f}s "
                f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
                f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}"
            )

            log_records.append(
                {
                    "epoch": epoch,
                    "time_sec": round(dt, 3),
                    "train_loss": round(float(tr["loss"]), 6),
                    "train_acc": round(float(tr["acc"]), 6),
                    "val_loss": round(float(va["loss"]), 6),
                    "val_acc": round(float(va["acc"]), 6),
                    "best_val_loss_so_far": round(float(min(best_val, va["loss"])), 6),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if effective_write_log and log_path is not None:
                write_train_log(log_path, log_payload)

            if va["loss"] < best_val:
                best_val = va["loss"]
                bad_epochs = 0
                save_checkpoint(model, optimizer, epoch, best_val, best_path, args)
                append_checkpoint_log(
                    {
                        "save_order": len(checkpoint_records) + 1,
                        "save_type": "best",
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "path": best_path.name,
                        "epoch": epoch,
                        "trigger": "epoch",
                        "val_loss": round(float(va["loss"]), 6),
                        "val_acc": round(float(va["acc"]), 6),
                        "train_loss": round(float(tr["loss"]), 6),
                        "train_acc": round(float(tr["acc"]), 6),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "best_val_loss_after": round(float(best_val), 6),
                    }
                )
            else:
                bad_epochs += 1

            if epoch % args.save_every == 0:
                epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
                save_checkpoint(model, optimizer, epoch, best_val, epoch_path, args)
                append_checkpoint_log(
                    {
                        "save_order": len(checkpoint_records) + 1,
                        "save_type": "epoch",
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "path": epoch_path.name,
                        "epoch": epoch,
                        "train_loss": round(float(tr["loss"]), 6),
                        "train_acc": round(float(tr["acc"]), 6),
                        "val_loss": round(float(va["loss"]), 6),
                        "val_acc": round(float(va["acc"]), 6),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "best_val_loss_so_far": round(float(best_val), 6),
                        "is_best_at_save_time": bool(float(va["loss"]) <= float(best_val)),
                    }
                )

        stop_flag = bool(bad_epochs >= args.early_stop_patience)
        if is_ddp:
            ctrl = torch.tensor(
                [float(best_val), float(bad_epochs), 1.0 if stop_flag else 0.0],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.broadcast(ctrl, src=0)
            best_val = float(ctrl[0].item())
            bad_epochs = int(ctrl[1].item())
            stop_flag = bool(int(ctrl[2].item()))

        if stop_flag:
            if main_process:
                print("Early stopping triggered.")
            break

    return best_path


# DDP 子进程入口：初始化分布式环境、构建数据与模型并执行训练。
def ddp_worker(
    rank: int,
    world_size: int,
    args: Args,
    train_h5_path: str,
    val_h5_path: str,
    test_h5_path: str,
    log_path: str,
):
    visible_count = torch.cuda.device_count()
    if rank >= visible_count:
        raise RuntimeError(
            f"Rank {rank} requested cuda:{rank}, but only {visible_count} CUDA devices are visible. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}"
        )
    local_device = torch.device(f"cuda:{rank}")

    try:
        torch.cuda.set_device(rank)
        configure_runtime_performance(args)
        requested_backend = str(args.ddp_backend).lower()
        effective_backend = requested_backend

        if requested_backend == "nccl":
            configure_ddp_backend_env(args)

            if bool(args.ddp_preflight):
                try:
                    setup_process_group(
                        rank=rank,
                        world_size=world_size,
                        backend="nccl",
                        timeout_minutes=args.ddp_preflight_timeout_minutes,
                    )
                    probe = torch.ones(1, device=local_device)
                    torch.distributed.all_reduce(probe, op=torch.distributed.ReduceOp.SUM)
                except Exception as ex:
                    if bool(args.ddp_auto_fallback_gloo):
                        effective_backend = "gloo"
                        if rank == 0:
                            print(f"[DDP] NCCL preflight failed, fallback to gloo backend. reason={ex}")
                    else:
                        raise
                finally:
                    cleanup_process_group()

        setup_process_group(
            rank=rank,
            world_size=world_size,
            backend=effective_backend,
            timeout_minutes=args.ddp_timeout_minutes,
        )
        if rank == 0:
            print(f"[DDP] effective backend: {torch.distributed.get_backend()}")

        # Keep deterministic initialization across ranks.
        set_seed(args.seed)

        tr_loader, va_loader, _, tr_sampler = make_dataloaders(
            args,
            train_h5_path=Path(train_h5_path),
            val_h5_path=Path(val_h5_path),
            test_h5_path=Path(test_h5_path),
            rank=rank,
            world_size=world_size,
        )

        local_model = ReformerSeq2Seq(args).to(local_device)
        maybe_load_finetune_weights(local_model, args, local_device, verbose=(rank == 0))
        ddp_backend = str(torch.distributed.get_backend()).lower()
        if ddp_backend == "nccl":
            local_model = DDP(
                local_model,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=bool(args.ddp_broadcast_buffers),
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
                gradient_as_bucket_view=True,
            )
        else:
            local_model = DDP(
                local_model,
                broadcast_buffers=bool(args.ddp_broadcast_buffers),
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
                gradient_as_bucket_view=False,
            )

        best_path = fit(
            local_model,
            tr_loader,
            va_loader,
            local_device,
            args,
            train_sampler=tr_sampler,
            write_log=(rank == 0),
            log_path=(Path(log_path) if rank == 0 else None),
        )
        if rank == 0:
            print("Best checkpoint:", str(best_path))
    finally:
        cleanup_process_group()


# 统一训练启动器：根据配置选择多卡 DDP 或单进程训练路径。
def launch_training(
    args: Args,
    device: torch.device,
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader],
    train_sampler,
    h5_paths: Dict[str, Path],
    log_path: Optional[Path] = None,
    log_payload: Optional[Dict] = None,
) -> Path:
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.use_ddp and world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        print(f"Launching DDP training with world_size={world_size}")
        torch.multiprocessing.spawn(
            ddp_worker,
            args=(
                world_size,
                args,
                str(h5_paths["train"]),
                str(h5_paths["val"]),
                str(h5_paths["test"]),
                str(log_path) if log_path is not None else str(create_train_log_file()),
            ),
            nprocs=world_size,
            join=True,
        )
        return Path(args.ckpt_dir) / "best.pt"

    print("Running single-process training")
    if train_loader is None or val_loader is None:
        raise ValueError("Single-process training requires train_loader and val_loader.")
    single_model = ReformerSeq2Seq(args).to(device)
    maybe_load_finetune_weights(single_model, args, device, verbose=True)
    local_best = fit(
        single_model,
        train_loader,
        val_loader,
        device,
        args,
        train_sampler=train_sampler,
        write_log=True,
        log_path=log_path,
        log_payload=log_payload,
    )
    return local_best


# 脚本主入口：解析参数、准备数据与环境、启动训练并收尾写入运行日志。
def main() -> None:
    args = build_args_from_cli()
    run_log_path = create_train_log_file()
    run_log_payload = build_initial_train_log_payload(args)
    write_train_log(run_log_path, run_log_payload)
    print(f"Training log file: {run_log_path}")

    set_seed(args.seed)
    run_started_at = time.time()

    try:
        print("PyTorch:", torch.__version__)
        print(json.dumps(asdict(args), indent=2))

        configure_visible_gpus(args.gpu_ids)
        device = get_device()
        print("Visible CUDA devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"))
        print("Device:", device)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("TF32 enabled:", bool(args.enable_tf32))

        h5_paths, data_stats = ensure_h5_splits(args)

        print("Data stats:", json.dumps(data_stats, indent=2))
        print(
            "Split sizes:",
            int(data_stats.get("train_samples", 0.0)),
            int(data_stats.get("val_samples", 0.0)),
            int(data_stats.get("test_samples", 0.0)),
        )

        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
        is_multi_gpu_ddp = bool(args.use_ddp and world_size > 1)

        train_loader = None
        val_loader = None
        train_sampler = None

        if not is_multi_gpu_ddp:
            train_loader, val_loader, test_loader, train_sampler = make_dataloaders(
                args,
                train_h5_path=h5_paths["train"],
                val_h5_path=h5_paths["val"],
                test_h5_path=h5_paths["test"],
            )

            sample_batch = next(iter(train_loader))
            assert_batch_contract(sample_batch, args)
            for k, v in sample_batch.items():
                print(k, tuple(v.shape), v.dtype)

            model = ReformerSeq2Seq(args).to(device)
            print("Trainable params:", count_trainable_params(model))

            with torch.no_grad():
                quick_batch = move_batch_to_device(sample_batch, device)
                quick_logits = model(
                    src_tokens=quick_batch["src_tokens"],
                    src_mask=quick_batch["src_mask"],
                    tgt_input_tokens=quick_batch["tgt_input_tokens"],
                    tgt_mask=quick_batch["tgt_mask"],
                )
            print("quick_logits shape:", tuple(quick_logits.shape))

            del test_loader  # Unused in this training-only script.
        else:
            print("DDP mode: skip single-process dataloader/model warmup before spawn.")

        best_ckpt_path = launch_training(
            args=args,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            h5_paths=h5_paths,
            log_path=run_log_path,
            log_payload=run_log_payload,
        )
        print("Training done. Best checkpoint at:", str(best_ckpt_path))
        finalize_train_log(
            run_log_path,
            run_log_payload,
            status="success",
            detail=f"Training completed normally. Best checkpoint: {best_ckpt_path}",
            run_duration_sec=(time.time() - run_started_at),
        )
    except KeyboardInterrupt as ex:
        print("Training interrupted by user (KeyboardInterrupt).")
        finalize_train_log(
            run_log_path,
            run_log_payload,
            status="interrupted",
            detail="Training was manually interrupted by user.",
            error=ex,
            run_duration_sec=(time.time() - run_started_at),
        )
        raise
    except Exception as ex:
        finalize_train_log(
            run_log_path,
            run_log_payload,
            status="error",
            detail="Training failed due to an unhandled exception.",
            error=ex,
            run_duration_sec=(time.time() - run_started_at),
        )
        raise


if __name__ == "__main__":
    main()
