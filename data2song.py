import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from audiotools import AudioSignal
from tqdm import trange

from dac import DACFile
from dac.utils import load_model

# Fixed template values from requirements.
TEMPLATE_CHUNK_LENGTH = 416
TEMPLATE_ORIGINAL_LENGTH = 12779666
TEMPLATE_INPUT_DB = -8.037091255187988
TEMPLATE_CHANNELS = 2
TEMPLATE_SAMPLE_RATE = 44100
TEMPLATE_PADDING = False
TEMPLATE_DAC_VERSION = "1.0.0"


def list_jsonl_files(jsonl_dir: Path) -> List[Path]:
    files = sorted([p for p in jsonl_dir.glob("*.jsonl") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in directory: {jsonl_dir}")
    if len(files) > 9:
        raise ValueError(f"Too many .jsonl files ({len(files)}). Maximum is 9.")
    return files


def _to_1d_sequence(raw: Any, source: Path) -> np.ndarray:
    arr = np.asarray(raw)

    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 1:
        seq = arr[0, 0]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        seq = arr[0]
    elif arr.ndim == 1:
        seq = arr
    else:
        raise ValueError(
            f"Unsupported sequence shape in {source}: {arr.shape}, expected [1,1,x] or [1,x] or [x]"
        )

    if seq.size == 0:
        raise ValueError(f"Empty sequence in file: {source}")

    return seq.astype(np.int64)


def read_jsonl_sequence(jsonl_file: Path) -> np.ndarray:
    lines = [line.strip() for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty jsonl file: {jsonl_file}")

    if len(lines) > 1:
        print(f"[WARN] {jsonl_file.name} has {len(lines)} lines; using the first non-empty line only.")

    parsed = json.loads(lines[0])

    # Compatibility: allow either plain array or object with key 'codes'.
    if isinstance(parsed, dict):
        if "codes" not in parsed:
            raise KeyError(f"Missing 'codes' key in {jsonl_file}")
        parsed = parsed["codes"]

    return _to_1d_sequence(parsed, jsonl_file)


def symmetric_edge_pad(seq: np.ndarray, target_len: int) -> np.ndarray:
    curr = seq.shape[0]
    if curr > target_len:
        raise ValueError(f"Current length {curr} is larger than target length {target_len}")
    if curr == target_len:
        return seq

    diff = target_len - curr
    left = diff // 2
    right = diff - left

    left_pad = np.full((left,), seq[0], dtype=seq.dtype)
    right_pad = np.full((right,), seq[-1], dtype=seq.dtype)
    return np.concatenate([left_pad, seq, right_pad], axis=0)


def build_data_template(sequences: List[np.ndarray]) -> Dict[str, Any]:
    max_len = max(seq.shape[0] for seq in sequences)
    aligned = [symmetric_edge_pad(seq, max_len) for seq in sequences]

    v = len(aligned)
    merged = np.stack(aligned, axis=0)  # [v, n]
    merged = merged[np.newaxis, :, :]   # [1, v, n]
    merged = np.repeat(merged, repeats=2, axis=0)  # [2, v, n]

    data = {
        "codes": merged.tolist(),
        "chunk_length": TEMPLATE_CHUNK_LENGTH,
        "original_length": TEMPLATE_ORIGINAL_LENGTH,
        "input_db": TEMPLATE_INPUT_DB,
        "channels": TEMPLATE_CHANNELS,
        "sample_rate": TEMPLATE_SAMPLE_RATE,
        "padding": TEMPLATE_PADDING,
        "dac_version": TEMPLATE_DAC_VERSION,
        "metadata": {
            "original_quantizers": 9,
            "modified_quantizers": 1,
            "modification_note": "Reduced from 9 to 1 quantizers",
        },
    }

    print("[INFO] Sequence preprocessing done.")
    print(f"[INFO] Unified tensor shape: [2, {v}, {max_len}]")
    return data


def save_dac_from_data(data: Dict[str, Any], out_dac_file: Path) -> Path:
    out_dac_file = out_dac_file.with_suffix(".dac")
    out_dac_file.parent.mkdir(parents=True, exist_ok=True)

    codes_tensor = torch.tensor(data["codes"], dtype=torch.long)
    artifacts = {
        "codes": codes_tensor.cpu().numpy().astype(np.uint16),
        "metadata": {
            "input_db": np.array(data["input_db"], dtype=np.float32),
            "original_length": data["original_length"],
            "sample_rate": data["sample_rate"],
            "chunk_length": data["chunk_length"],
            "channels": data["channels"],
            "padding": data["padding"],
            "dac_version": data["dac_version"],
        },
    }

    with open(out_dac_file, "wb") as f:
        np.save(f, artifacts)

    print("[INFO] DAC saved.")
    print(f"[INFO] DAC path: {out_dac_file}")
    print(
        "[INFO] DAC metadata: "
        f"chunk_length={data['chunk_length']}, original_length={data['original_length']}, "
        f"input_db={data['input_db']}, channels={data['channels']}, sample_rate={data['sample_rate']}, "
        f"padding={data['padding']}, dac_version={data['dac_version']}"
    )
    return out_dac_file


@torch.no_grad()
def decompress_modified(self, obj: DACFile, verbose: bool = True) -> AudioSignal:
    self.eval()

    chunk_length = 416
    min_tail_codes = 32
    input_db = -11
    channels = 2
    sample_rate = 44100
    padding = False

    original_padding = self.padding
    self.padding = padding

    try:
        codes = obj.codes
        original_device = codes.device
        total_codes = codes.shape[-1]

        # Build safe segments so the final tail is not too short for decoder conv kernels.
        segments = []
        start = 0
        while start < total_codes:
            end = min(start + chunk_length, total_codes)
            remaining = total_codes - end
            if 0 < remaining < min_tail_codes:
                end = total_codes
            segments.append((start, end))
            start = end

        if verbose:
            print(
                f"[INFO] Decode segments: {len(segments)} chunk(s), "
                f"chunk_length={chunk_length}, min_tail_codes={min_tail_codes}, total_codes={total_codes}"
            )

        recons = []
        seg_iter = trange(len(segments), desc="Decoding chunks") if verbose else range(len(segments))
        for idx in seg_iter:
            s, e = segments[idx]
            c = codes[..., s:e].to(self.device)
            z = self.quantizer.from_codes(c)[0]
            r = self.decode(z)
            recons.append(r.to(original_device))

        recons = torch.cat(recons, dim=-1)
        recons = AudioSignal(recons, sample_rate)

        resample_fn = recons.resample
        loudness_fn = recons.loudness
        if recons.signal_duration >= 10 * 60 * 60:
            resample_fn = recons.ffmpeg_resample
            loudness_fn = recons.ffmpeg_loudness

        recons.normalize(input_db)
        resample_fn(sample_rate)

        audio_data = recons.audio_data
        flat_audio = audio_data.flatten()
        total_samples = flat_audio.numel()

        if total_samples % channels != 0:
            samples_per_channel = total_samples // channels
            flat_audio = flat_audio[: samples_per_channel * channels]
            print(
                f"[INFO] Audio samples adjusted: {total_samples} -> {samples_per_channel * channels}"
            )
        else:
            samples_per_channel = total_samples // channels

        recons.audio_data = flat_audio.reshape(-1, channels, samples_per_channel)
        loudness_fn()
        return recons
    finally:
        self.padding = original_padding


def decode_dac_to_wav(dac_file: Path, wav_file: Path) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model on device: {device}")

    generator = load_model(
        model_type="44khz",
        model_bitrate="8kbps",
        tag="latest",
        load_path="",
    )
    generator.to(device)
    generator.eval()

    print("[INFO] Using custom decoder with fixed parameters.")
    print("[INFO] params: chunk_length=416, input_db=-11, channels=2, sample_rate=44100, padding=False")
    generator.decompress = partial(decompress_modified, generator)

    artifact = DACFile.load(dac_file)
    recons = generator.decompress(artifact, verbose=True)

    wav_file = wav_file.with_suffix(".wav")
    wav_file.parent.mkdir(parents=True, exist_ok=True)
    recons.write(wav_file)

    print("[INFO] Song generated.")
    print(f"[INFO] WAV path: {wav_file}")
    print(f"[INFO] Output audio shape: {recons.audio_data.shape}, sample_rate={recons.sample_rate}")
    return wav_file


def process(jsonl_path: str, dac_path: str, song_path: str) -> None:
    jsonl_dir = Path(jsonl_path)
    out_dac_dir = Path(dac_path)
    out_song_dir = Path(song_path)

    if not jsonl_dir.exists() or not jsonl_dir.is_dir():
        raise NotADirectoryError(f"jsonl_path is not a valid directory: {jsonl_dir}")

    out_dac_dir.mkdir(parents=True, exist_ok=True)
    out_song_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list_jsonl_files(jsonl_dir)
    name = jsonl_files[0].stem

    print(f"[INFO] Found {len(jsonl_files)} jsonl file(s) in: {jsonl_dir}")
    print(f"[INFO] Output DAC dir: {out_dac_dir}")
    print(f"[INFO] Output song dir: {out_song_dir}")
    print(f"[INFO] Base name (from first jsonl): {name}")

    sequences: List[np.ndarray] = []
    for jf in jsonl_files:
        seq = read_jsonl_sequence(jf)
        sequences.append(seq)
        print(f"[INFO] Loaded {jf.name}, sequence length={seq.shape[0]}")

    data = build_data_template(sequences)

    dac_file = save_dac_from_data(data, out_dac_dir / name)
    decode_dac_to_wav(dac_file, out_song_dir / name)

    print("[INFO] One-click pipeline finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click: jsonl folder -> aligned codes -> dac -> wav"
    )
    parser.add_argument("--jsonl_path", required=True, help="Directory containing jsonl files")
    parser.add_argument("--dac_path", required=True, help="Directory to save generated .dac")
    parser.add_argument("--song_path", required=True, help="Directory to save generated songs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        process(args.jsonl_path, args.dac_path, args.song_path)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    # Example mode: edit these paths and run this file directly.
    # If CLI args are provided, normal argparse flow is used.
    if len(sys.argv) == 1:
        example_args = {
            "jsonl_path": r"W:\code\song2piano\descript-audio-codec\moni",
            "dac_path": r"W:\code\song2piano\descript-audio-codec\output",
            "song_path": r"W:\code\song2piano\descript-audio-codec\reconstructed",
        }

        print("[INFO] Running with built-in example arguments.")
        print(f"[INFO] jsonl_path={example_args['jsonl_path']}")
        print(f"[INFO] dac_path={example_args['dac_path']}")
        print(f"[INFO] song_path={example_args['song_path']}")

        try:
            process(
                example_args["jsonl_path"],
                example_args["dac_path"],
                example_args["song_path"],
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            raise SystemExit(1)
        raise SystemExit(0)

    raise SystemExit(main())
