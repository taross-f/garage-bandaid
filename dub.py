"""動画の音声を、曲の区間だけ各曲の out/after.wav に差し替えた新しい動画を書き出す（元の動画は変更しない）。

  uv run dub.py source/live.MP4 recordings/live.wav exports/live_remix.MP4

- recordings/live.wav は動画から書き出した音声で、camera.py に渡したもの。
  camera.json の path がこれと一致する曲を、camera.json の開始位置と時計のずれのとおりに置く
- 曲の外（MC・歓声）は動画の元の音のまま。曲の区間での after.wav との音量差の中央値で音量を揃える
- 映像は再エンコードしない
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_metrics import integrated_lufs
from camera import warp
from songs import Song, prepared_songs

CROSSFADE_SEC = 0.5  # 曲の前後には prep の余白（3秒）があるので、その中でつなぐ
AUDIO_BITRATE = "256k"


def placements(audio: Path) -> list[tuple[Song, dict]]:
    """camera.json がこの音声を指している曲を、動画の中の開始順に返す。"""
    found = []
    for song in prepared_songs():
        path = song.work / "camera.json"
        if path.exists():
            info = json.loads(path.read_text())
            if Path(info["path"]) == audio.resolve():
                found.append((song, info))
    if not found:
        raise SystemExit(f"{audio} に位置合わせした曲がない。先に camera.py を実行する")
    return sorted(found, key=lambda f: f[1]["song_starts_at_sec"])


def to_bed_clock(after: np.ndarray, start: float, slope: float) -> tuple[int, np.ndarray]:
    """song.wav の時間で作った after を、元の録音の時間に置き直す。(元の録音での開始サンプル, 置き直した音) を返す。

    元の録音の位置 start + n * (1 + slope) に after[n] が来るようにする。
    """
    first = int(np.ceil(start))
    n_out = int(np.floor(start + (len(after) - 1) * (1 + slope))) - first + 1
    return first, warp(after, (first - start) / (1 + slope), 1 / (1 + slope), n_out)


def splice(bed: np.ndarray, pieces: list[tuple[int, np.ndarray]], bed_gain_db: float, sr: int) -> np.ndarray:
    """bed に bed_gain_db をかけ、pieces の区間を差し替える（両端は CROSSFADE_SEC の等パワーのクロスフェード）。"""
    out = bed * 10 ** (bed_gain_db / 20)
    fade = int(CROSSFADE_SEC * sr)
    for first, piece in pieces:
        lo, hi = max(first, 0), min(first + len(piece), len(out))
        if hi - lo < 2 * fade:
            raise ValueError(f"差し替える区間が短すぎる（{(hi - lo) / sr:.2f} 秒）")
        mix = np.ones(hi - lo)  # piece の割合
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        mix[:fade], mix[-fade:] = ramp, ramp[::-1]
        out[lo:hi] = out[lo:hi] * np.sqrt(1 - mix)[:, None] + piece[lo - first: hi - first] * np.sqrt(mix)[:, None]
    return out


def dub(video: Path, audio: Path, output: Path) -> dict:
    if output.resolve() == video.resolve():
        raise SystemExit("出力先が元の動画と同じ。元の動画は上書きしない")
    bed, sr = sf.read(audio, dtype="float64", always_2d=True)
    pieces, report, gains = [], [], []
    for song, info in placements(audio):
        after, after_sr = sf.read(song.out / "after.wav", dtype="float64", always_2d=True)
        if after_sr != sr:
            raise ValueError(f"{song.name}: after.wav が {after_sr}Hz（{audio} は {sr}Hz）")
        if after.shape[1] == 1:
            after = np.repeat(after, 2, axis=1)
        first, piece = to_bed_clock(after, info["song_starts_at_sec"] * sr, info["clock_drift_ppm"] * 1e-6)
        end = min(first + len(piece), len(bed))
        gain = integrated_lufs(piece[: end - first], sr) - integrated_lufs(bed[first:end], sr)
        gains.append(gain)
        pieces.append((first, piece))
        report.append({"song": song.name, "start_sec": first / sr, "end_sec": (first + len(piece)) / sr,
                       "cut_at_video_end_sec": max(0, first + len(piece) - len(bed)) / sr,
                       "after_minus_video_db": gain})
    for (a_first, a), (b_first, _) in zip(pieces, pieces[1:]):
        if a_first + len(a) > b_first:
            raise ValueError(f"曲の区間が重なっている（{a_first / sr:.2f} 秒からの曲と {b_first / sr:.2f} 秒からの曲）")

    bed_gain_db = float(np.median(gains))
    mixed = splice(bed, pieces, bed_gain_db, sr)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        raise ValueError(f"差し替えた音声がクリップする（ピーク {20 * np.log10(peak):.2f} dBFS）")

    output.parent.mkdir(parents=True, exist_ok=True)
    track = output.with_suffix(".wav")
    sf.write(track, mixed, sr, subtype="PCM_24")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(track),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                        "-movflags", "+faststart", str(output)], check=True)
    finally:
        track.unlink()
    return {"output": str(output), "video_audio_gain_db": bed_gain_db, "songs": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="元の動画（変更しない）")
    parser.add_argument("audio", type=Path, help="動画から書き出した音声（camera.py に渡したもの）")
    parser.add_argument("output", type=Path, help="書き出す動画")
    args = parser.parse_args()
    print(json.dumps(dub(args.video, args.audio, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
