"""切り出し・診断・de-clip判定・ステム分離（手順1-4）。曲ごとに1回だけ実行する。

例:
  .venv/bin/python prep.py "recordings/20261031_my song.wav"
  .venv/bin/python prep.py live.wav --name song3 --start 612 --end 845   # 前後3秒の余裕を付けて切り出す
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal

from audio_metrics import clip_runs, db, diagnose
from songs import ROOT, STEM_NAMES, TEMPLATE_YAML, Song, slugify

SR = 48000
MODEL = "htdemucs"
SHIFTS = 2
PAD_SEC = 3.0

# ピーク直下 0.1dB に 5 サンプル以上張り付く区間の割合がこれを超えたら「割れが目立つ」とみなす
DECLIP_MIN_RUN = 5
DECLIP_RATIO_PCT = 0.01

# 進み具合の通知先: progress(段階名, 0〜1 の進捗 or None)
Progress = Callable[[str, float | None], None]


def no_progress(stage: str, fraction: float | None) -> None:
    pass


def padded_range(start: float | None, end: float | None, duration: float) -> tuple[float, float]:
    """指定区間の前後に PAD_SEC の余裕を付け、ファイルの範囲内に収める。未指定なら全区間。"""
    lo = 0.0 if start is None else max(0.0, start - PAD_SEC)
    hi = duration if end is None else min(duration, end + PAD_SEC)
    if lo >= hi:
        raise ValueError(f"empty range: start={start} end={end} duration={duration}")
    return lo, hi


def cut_song(source: Path, song: Song, lo: float, hi: float) -> None:
    song.work.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ss", f"{lo:.6f}", "-to", f"{hi:.6f}",
         "-af", f"aresample={SR}:resampler=soxr:precision=28",
         "-c:a", "pcm_s24le", str(song.song_wav)],
        check=True,
    )


def declip_check(source: np.ndarray) -> dict:
    ceiling = float(np.max(np.abs(source))) * 10 ** (-0.1 / 20)
    runs = clip_runs(source, ceiling, DECLIP_MIN_RUN)
    return {**runs, "needed": runs["ratio_pct"] > DECLIP_RATIO_PCT}


def separate(song: Song, progress: Progress = no_progress) -> None:
    """demucs を実行する。stderr はそのまま流しつつ、tqdm の % から進捗を拾う。

    shifts 回ぶん 0→100% の進捗バーが順に出るので、% が戻ったら次の回とみなす。
    """
    # MPS は htdemucs の Conv1d で NotImplementedError になるため CPU で実行する
    proc = subprocess.Popen(
        ["demucs", "-n", MODEL, "--shifts", str(SHIFTS), "--float32", "-d", "cpu",
         "-o", str(song.work / "demucs"), str(song.song_wav)],
        stderr=subprocess.PIPE,
        env={**os.environ, "PATH": f"{ROOT / '.venv/bin'}:{os.environ['PATH']}"},
    )
    shift, last, tail = 0, 0, b""
    for chunk in iter(lambda: proc.stderr.read1(4096), b""):
        sys.stderr.buffer.write(chunk)
        sys.stderr.flush()
        tail = (tail + chunk)[-4000:]
        for pct in re.findall(rb"(\d+)%\|", chunk):
            pct = int(pct)
            if pct < last:
                shift += 1
            last = pct
            progress("分離", min(1.0, (shift * 100 + pct) / (SHIFTS * 100)))
    if proc.wait() != 0:
        raise RuntimeError(f"demucs failed ({proc.returncode}): {tail.decode(errors='replace')[-800:]}")


def resample_stems(song: Song) -> None:
    """demucs の 44.1kHz 出力を song.wav と同じ 48kHz・同じ長さ・同じch数に揃える。"""
    mix, _ = sf.read(song.song_wav, dtype="float64", always_2d=True)
    song.stems.mkdir(exist_ok=True)
    for name in STEM_NAMES:
        stem, sr = sf.read(song.work / "demucs" / MODEL / "song" / f"{name}.wav",
                           dtype="float64", always_2d=True)
        if mix.shape[1] == 1:
            stem = stem.mean(axis=1, keepdims=True)
        if sr != SR:
            stem = signal.resample_poly(stem, SR, sr, axis=0)
        if abs(len(stem) - len(mix)) > 8:
            raise ValueError(f"{name}: length {len(stem)} != song {len(mix)}")
        stem = stem[: len(mix)]
        stem = np.pad(stem, ((0, len(mix) - len(stem)), (0, 0)))
        sf.write(song.stems / f"{name}.wav", stem, SR, subtype="FLOAT")


def verify_stems(song: Song) -> float:
    """ステムの和と song.wav の残差 (dB) を返す。ズレていれば大きくなる。"""
    mix, _ = sf.read(song.song_wav, dtype="float64", always_2d=True)
    total = sum(sf.read(song.stems / f"{n}.wav", dtype="float64", always_2d=True)[0]
                for n in STEM_NAMES)
    return db(float(np.sqrt(np.mean((total - mix) ** 2)))) - db(float(np.sqrt(np.mean(mix**2))))


def prepare(source: Path, song: Song, start: float | None, end: float | None,
            progress: Progress = no_progress) -> dict:
    """diagnosis.json は最後に書く。これがある曲を「準備済み」とみなす（songs.prepared_songs）。"""
    info = sf.info(source)
    lo, hi = padded_range(start, end, info.duration)
    progress("切り出し", None)
    cut_song(source, song, lo, hi)
    (song.work / "source.json").write_text(json.dumps({
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_sec": lo,
        "end_sec": hi,
    }, indent=2, ensure_ascii=False))

    progress("診断", None)
    mix, sr = sf.read(song.song_wav, dtype="float64")
    report = {"song": diagnose(mix, sr)}

    original, _ = sf.read(source, dtype="float64",
                          start=int(lo * info.samplerate), stop=int(hi * info.samplerate))
    report["declip"] = declip_check(original)
    if report["declip"]["needed"]:
        raise SystemExit(f"de-clip が必要だが未実装: {report['declip']}")

    progress("分離", 0.0)
    separate(song, progress)
    progress("ステム整列", None)
    resample_stems(song)
    shutil.rmtree(song.work / "demucs")
    report["stem_sum_residual_db"] = verify_stems(song)

    if not song.config.exists():
        shutil.copyfile(TEMPLATE_YAML, song.config)
    (song.work / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    progress("完了", 1.0)
    return report


def new_song(source: Path, name: str | None, start: float | None, end: float | None,
             force: bool = False, progress: Progress = no_progress) -> tuple[Song, dict]:
    """songs/<name>/ を作って prepare する。途中で失敗したら作りかけの work/ を消す（mix.yaml は残す）。"""
    if not source.is_file():
        raise FileNotFoundError(f"source not found: {source}")
    song = Song(name or slugify(source.stem))
    if song.work.exists():
        if not force:
            raise FileExistsError(f"{song.work} は既にある。作り直すなら --force")
        shutil.rmtree(song.work)
    try:
        return song, prepare(source, song, start, end, progress)
    except BaseException:
        shutil.rmtree(song.work, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="入力の録音ファイル（変更しない）")
    parser.add_argument("--name", help="曲名（songs/<name>/）。省略時は入力ファイル名から作る")
    parser.add_argument("--start", type=float, help="曲の開始位置 [秒]。省略時は先頭")
    parser.add_argument("--end", type=float, help="曲の終了位置 [秒]。省略時は末尾")
    parser.add_argument("--force", action="store_true", help="既存の work/ を作り直す（mix.yaml は残す）")
    args = parser.parse_args()

    try:
        song, report = new_song(args.source, args.name, args.start, args.end, args.force)
    except FileExistsError as e:
        raise SystemExit(str(e))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{song.config} を調整して `.venv/bin/python mix.py {song.name}` を実行")


if __name__ == "__main__":
    main()
