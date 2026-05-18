import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from audiotools import AudioSignal

from dac.utils import load_model


def normalize_song_volume(song_path: Path, normalized_output_path: Path, target_dBFS: float = -10.0) -> tuple[float, float]:
    """按目标 dBFS 对单首歌曲做音量标准化，并导出为临时 WAV。"""
    try:
        pydub_module = __import__("pydub", fromlist=["AudioSegment"])
        AudioSegment = getattr(pydub_module, "AudioSegment")
    except Exception as exc:
        raise ImportError("未安装 pydub，请先执行 pip install pydub") from exc

    audio = AudioSegment.from_file(song_path)
    current_dBFS = audio.dBFS

    if current_dBFS == float("-inf"):
        # 静音文件保持原样，避免无意义增益。
        normalized_audio = audio
        gain_adjustment = 0.0
    else:
        gain_adjustment = target_dBFS - current_dBFS
        normalized_audio = audio.apply_gain(gain_adjustment)

    normalized_audio.export(normalized_output_path, format="wav")
    normalized_dBFS = normalized_audio.dBFS

    print(f"[预处理] 当前音量: {current_dBFS:.2f} dBFS")
    print(f"[预处理] 增益调整: {gain_adjustment:.2f} dB")
    print(f"[预处理] 调整后音量: {normalized_dBFS:.2f} dBFS")
    print(f"[预处理] 临时音频: {normalized_output_path}")

    return float(current_dBFS), float(normalized_dBFS)


def encode_song_to_initial_jsonl(song_audio_path: Path, initial_jsonl_path: Path) -> dict:
    """参考 2jsonl.py 的 JSONL 模式，把歌曲编码为一条初始 JSONL 数据。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[编码] 使用设备: {device}")

    generator = load_model(
        model_type="44khz",
        model_bitrate="8kbps",
        tag="latest",
        load_path="",
    )
    generator.to(device)
    generator.eval()

    signal = AudioSignal(str(song_audio_path))

    artifact = generator.compress(
        signal,
        win_duration=5.0,
        verbose=False,
        n_quantizers=None,
    )

    codes_np = artifact.codes.cpu().numpy()
    data = {
        "codes": codes_np.tolist(),
        "chunk_length": artifact.chunk_length,
        "original_length": artifact.original_length,
        "input_db": float(artifact.input_db),
        "channels": artifact.channels,
        "sample_rate": artifact.sample_rate,
        "padding": artifact.padding,
        "dac_version": artifact.dac_version,
        "metadata": {
            "original_quantizers": int(codes_np.shape[1]),
            "modified_quantizers": 1,
            "modification_note": "Reduced from 9 to 1 quantizers",
        },
    }

    initial_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(initial_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[编码] 初始 JSONL: {initial_jsonl_path}")
    print(f"[编码] codes 形状: {tuple(codes_np.shape)}")

    return data


def split_initial_jsonl_to_18_parts(initial_data: dict, name: str, data_path: Path) -> list[Path]:
    """参考 cut_jsonl.py + cut_jsonl2data.py，将 codes 拆为 18 份 [1,n]。"""
    if "codes" not in initial_data:
        raise KeyError("初始 JSONL 数据中缺少 'codes' 字段")

    codes = np.asarray(initial_data["codes"])
    if codes.ndim != 3:
        raise ValueError(f"codes 维度错误: {codes.shape}，期望 [2,9,n]")

    if codes.shape[0] != 2:
        raise ValueError(f"codes 第 1 维必须是 2，当前为 {codes.shape[0]}")

    split_count = int(codes.shape[0] * codes.shape[1])
    if split_count != 18:
        raise ValueError(
            f"当前 codes 形状为 {codes.shape}，可拆分数量是 {split_count}，不是目标 18 份。"
        )

    data_path.mkdir(parents=True, exist_ok=True)

    reshaped = codes.reshape(split_count, codes.shape[2])
    out_files: list[Path] = []

    for i in range(split_count):
        # 每份保存为 [1,n]。
        part = [reshaped[i].tolist()]
        out_file = data_path / f"{name}_{i + 1:02d}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False)
            f.write("\n")
        out_files.append(out_file)

    return out_files


def process_song_to_data(song_path: str, data_path: str) -> None:
    song = Path(song_path)
    output_dir = Path(data_path)

    if not song.exists():
        raise FileNotFoundError(f"歌曲文件不存在: {song}")

    name = song.stem
    print(f"[开始] song_path: {song}")
    print(f"[开始] data_path: {output_dir}")
    print(f"[开始] name: {name}")

    with tempfile.TemporaryDirectory(prefix="song2data_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        normalized_song = tmp_root / f"{name}_normalized.wav"
        initial_jsonl = tmp_root / f"{name}_initial.jsonl"

        normalize_song_volume(song, normalized_song, target_dBFS=-10.0)
        initial_data = encode_song_to_initial_jsonl(normalized_song, initial_jsonl)
        output_files = split_initial_jsonl_to_18_parts(initial_data, name, output_dir)

    print("-" * 60)
    print("[完成] 已生成 18 个 JSONL 文件:")
    for p in output_files:
        print(f"  - {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键将歌曲转换为 18 个 JSONL 数据文件")
    parser.add_argument("--song_path", type=str, required=True, help="歌曲文件路径")
    parser.add_argument("--data_path", type=str, required=True, help="18个JSONL输出目录")
    return parser.parse_args()


def main() -> None:
    # 直接修改这个 args 即可运行，不传命令行参数时会使用这里的值。
    args = {
        "song_path": r"W:\code\song2piano\descript-audio-codec\input\dshho.mp3",
        "data_path": r"W:\code\song2piano\descript-audio-codec\input\dshho_data",
    }

    if len(sys.argv) > 1:
        ns = parse_args()
        args = {
            "song_path": ns.song_path,
            "data_path": ns.data_path,
        }

    process_song_to_data(song_path=args["song_path"], data_path=args["data_path"])


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"依赖导入失败: {e}")
        print("请确认已安装依赖: pip install -r requirements.txt")
        print("并确保 ffmpeg 可用且已加入 PATH")
        sys.exit(1)
    except Exception as e:
        print(f"执行失败: {e}")
        sys.exit(1)
