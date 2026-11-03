"""songs/<name>/mix.yaml の設定でステムをミックスし songs/<name>/out/ に書き出す（手順5-6）。

prep.py で作った work/song.wav と work/stems/ を読むだけで、分離は再実行しない。

例:
  .venv/bin/python mix.py 20261031_my_song
  .venv/bin/python mix.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from pedalboard import Compressor, LowShelfFilter, PeakFilter, Pedalboard
from scipy import ndimage, signal

from audio_metrics import db, diagnose, integrated_lufs, true_peak_dbfs
from songs import STEM_NAMES, Song, prepared_songs

LUFS_TOLERANCE = 0.05
TP_MARGIN_DB = 0.05
MAX_LOUDNESS_ITERATIONS = 10
CAMERA_OFF_DB = -60.0  # camera 節の量がこれ以下なら、その層は足さない


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x, sr


def hpf(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    """ゼロ位相 HPF（24dB/oct）。元音源ブレンド時に位相がずれないようにする。"""
    sos = signal.butter(2, freq, "highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def apply_board(x: np.ndarray, sr: int, plugins: list) -> np.ndarray:
    if not plugins:
        return x
    out = Pedalboard(plugins)(x.T.astype(np.float32), sr)
    return out.T.astype(np.float64)


def eq_plugins(bands: list[dict]) -> list:
    filters = {"peak": PeakFilter, "low_shelf": LowShelfFilter}
    return [filters[b.get("type", "peak")](b["freq"], b["gain_db"], b["q"]) for b in bands]


def compressor(cfg: dict) -> Compressor:
    return Compressor(cfg["threshold_db"], cfg["ratio"], cfg["attack_ms"], cfg["release_ms"])


def rms_db(x: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(x**2))))


def process_stem(x: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, float]:
    """ゲイン・HPF・EQ・コンプを適用し、(処理後, コンプによるRMS減少dB) を返す。"""
    x = hpf(x, sr, cfg["hpf_hz"])
    x = apply_board(x, sr, eq_plugins(cfg["eq"]))
    compressed = apply_board(x, sr, [compressor(cfg["comp"])])
    reduction = rms_db(x) - rms_db(compressed)
    return compressed * 10 ** (cfg["gain_db"] / 20), reduction


def true_peak_limit(x: np.ndarray, sr: int, ceiling_db: float,
                    lookahead_ms: float, release_ms: float) -> np.ndarray:
    """4倍オーバーサンプルで検出するルックアヘッド・リミッター。長さと位置は変えない。"""
    ceiling = 10 ** (ceiling_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=0)
    linked = np.abs(up).max(axis=1)[: len(x) * 4].reshape(len(x), 4)
    required = np.minimum(1.0, ceiling / np.maximum(linked.max(axis=1), 1e-12))

    window = int(sr * lookahead_ms / 1000) | 1
    held = ndimage.minimum_filter1d(required, window)
    coef = 1 - np.exp(-1000 / (release_ms * sr))
    released = np.empty_like(held)
    current = 1.0
    for i, target in enumerate(held.tolist()):
        current = min(target, current + (1.0 - current) * coef)
        released[i] = current
    gain = ndimage.uniform_filter1d(released, window)
    return x * gain[:, None]


def normalize_master(x: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, float]:
    """リミッター込みで目標 LUFS に合わせる。(処理後, 適用ゲインdB) を返す。"""
    target = cfg["target_lufs"]
    ceiling = cfg["true_peak_dbtp"] - TP_MARGIN_DB
    gain_db = target - integrated_lufs(x, sr)
    for _ in range(MAX_LOUDNESS_ITERATIONS):
        limited = true_peak_limit(x * 10 ** (gain_db / 20), sr, ceiling,
                                  cfg["limiter_lookahead_ms"], cfg["limiter_release_ms"])
        error = target - integrated_lufs(limited, sr)
        if abs(error) <= LUFS_TOLERANCE:
            break
        gain_db += error
    else:
        raise RuntimeError(f"loudness did not converge: error {error:.2f} LU")
    peak = true_peak_dbfs(limited)
    if peak > cfg["true_peak_dbtp"]:
        raise RuntimeError(f"true peak {peak:.2f} dBTP exceeds {cfg['true_peak_dbtp']}")
    return limited, gain_db


def add_camera(target: Song, bus: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """camera.py で時間を合わせた別録音（work/camera.wav）から、左右の広がりと高域を借りて足す。

    広がりは camera の左右差（L-R）だけを使う。WAV 側と同じ音を重ねないので、打ち消し合いが起きにくい。
    どちらの層も切っているときは bus をそのまま返す（モノラルのまま）。
    """
    width_on, air_on = cfg["width_db"] > CAMERA_OFF_DB, cfg["air_db"] > CAMERA_OFF_DB
    if not (width_on or air_on):
        return bus
    path = target.work / "camera.wav"
    if not path.exists():
        raise FileNotFoundError(f"{path} がない。先に camera.py で作るか、mix.yaml の camera 節を消す")
    camera, camera_sr = load_audio(path)
    if camera_sr != sr or len(camera) != len(bus) or camera.shape[1] != 2:
        raise ValueError(f"camera.wav {camera.shape}@{camera_sr} は song と同じ長さのステレオが必要")

    out = np.repeat(bus, 2, axis=1) if bus.shape[1] == 1 else bus.copy()
    if air_on:
        air = hpf(camera.mean(axis=1, keepdims=True), sr, cfg["air_hz"])
        out += air * 10 ** (cfg["air_db"] / 20)
    if width_on:
        side = hpf((camera[:, :1] - camera[:, 1:]) / 2, sr, cfg["width_hpf_hz"]) * 10 ** (cfg["width_db"] / 20)
        out += np.hstack([side, -side])
    return out


def master_chain(x: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, float]:
    x = hpf(x, sr, cfg["hpf_hz"])
    x = apply_board(x, sr, [*eq_plugins(cfg["eq"]), compressor(cfg["bus_comp"])])
    return normalize_master(x, sr, cfg)


def render(target: Song, cfg: dict) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """ミックスして (after, before, sr, report) を返す。ファイルには書かない。"""
    song, sr = load_audio(target.song_wav)
    unknown = set(cfg["stems"]) ^ set(STEM_NAMES)
    if unknown:
        raise ValueError(f"mix.yaml stems mismatch: {sorted(unknown)}")

    bus = song * 10 ** (cfg["source_blend_db"] / 20)
    stem_report = {}
    for name in STEM_NAMES:
        stem, stem_sr = load_audio(target.stems / f"{name}.wav")
        if stem_sr != sr or stem.shape != song.shape:
            raise ValueError(f"{name}: {stem.shape}@{stem_sr} != song {song.shape}@{sr}")
        processed, reduction = process_stem(stem, sr, cfg["stems"][name])
        bus += processed
        stem_report[name] = {
            "rms_dbfs": rms_db(stem),
            "processed_rms_dbfs": rms_db(processed),
            "comp_reduction_db": reduction,
        }

    if "camera" in cfg:
        bus = add_camera(target, bus, sr, cfg["camera"])
    after, master_gain = master_chain(bus, sr, cfg["master"])
    achieved = integrated_lufs(after, sr)
    before = song * 10 ** ((achieved - integrated_lufs(song, sr)) / 20)

    report = {
        "stems": stem_report,
        "master_gain_db": master_gain,
        "metrics": {
            "song": diagnose(song, sr),
            "before": diagnose(before, sr),
            "after": diagnose(after, sr),
        },
    }
    return after, before, sr, report


def mix(target: Song, cfg: dict) -> dict:
    after, before, sr, report = render(target, cfg)
    target.out.mkdir(exist_ok=True)
    sf.write(target.out / "after.wav", after, sr, subtype="PCM_24")
    # 音量を揃えただけ。0dBFS を超えても潰さないよう 32bit float で保存する
    sf.write(target.out / "before.wav", before, sr, subtype="FLOAT")
    (target.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def print_report(report: dict) -> None:
    print(f"{'stem':8} {'RMS':>7} {'処理後':>7} {'comp減':>7}")
    for name, s in report["stems"].items():
        print(f"{name:8} {s['rms_dbfs']:7.1f} {s['processed_rms_dbfs']:7.1f} "
              f"{s['comp_reduction_db']:7.1f}")
    print(f"master gain: {report['master_gain_db']:+.2f} dB\n")

    m = report["metrics"]
    rows = [
        ("LUFS", lambda d: f"{d['lufs']:.2f}"),
        ("True Peak dBTP", lambda d: f"{d['true_peak_dbtp']:.2f}"),
        ("RMS dBFS", lambda d: f"{d['rms_dbfs']:.2f}"),
        ("クリップsamples", lambda d: str(d["clip"]["clipped_samples"])),
    ]
    rows += [(band, lambda d, b=band: f"{d['bands_pct'][b]:.1f}%")
             for band in m["after"]["bands_pct"]]
    print(f"{'':22} {'song':>9} {'before':>9} {'after':>9}")
    for label, fmt in rows:
        print(f"{label:22} {fmt(m['song']):>9} {fmt(m['before']):>9} {fmt(m['after']):>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="songs/ の下の曲名（複数可）")
    parser.add_argument("--all", action="store_true", help="分離済みの全曲をミックスする")
    args = parser.parse_args()
    if args.all == bool(args.names):
        parser.error("曲名か --all のどちらか一方を指定する")

    targets = prepared_songs() if args.all else [Song(n) for n in args.names]
    for target in targets:
        if not target.stems.is_dir():
            raise SystemExit(f"{target.stems} がない。先に prep.py を実行する")
        print(f"== {target.name}")
        print_report(mix(target, yaml.safe_load(target.config.read_text())))


if __name__ == "__main__":
    main()
