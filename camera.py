"""同じ演奏を別の機材（カメラなど）で録った音声を、song.wav に時間を合わせて work/camera.wav に書き出す。

  uv run camera.py 20261031_my_song recordings/camera.wav

1. 音の立ち上がり（スペクトルフラックス）の相互相関で、録音のどこに曲があるかを 10ms 単位で探す
2. 曲の中の数か所で GCC-PHAT をとり、サンプル以下の精度でずれを測る
3. ずれを「開始位置 + 時計の速さの違い（ppm）」の直線で近似し、その直線どおりにカメラ側を伸び縮みさせる
4. song.wav と同じ長さ・同じラウドネスにして保存する（mix.yaml の camera 節で使う）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from audio_metrics import integrated_lufs
from songs import Song

FLUX_SR = 16000
FLUX_HOP = 160  # 10ms
FLUX_FFT = 1024
COARSE_WINDOW_SEC = 30.0
COARSE_AGREE_SEC = 0.05     # 曲の3か所で探した開始位置がこの範囲で揃わなければ失敗にする

FINE_BAND_HZ = (300, 4000)  # カメラは低域がほぼ無いので、両方にある帯域で比べる
FINE_WINDOW_SEC = 4.0
FINE_STEP_SEC = 10.0
FINE_SEARCH_SEC = 0.05
PHAT_BETA = 0.7             # 1 で完全な GCC-PHAT。この録音では 1 だとピークがぼやけた
SINC_TAPS = 32
SINC_STEPS = 200            # ピークの前後 1 サンプルを 1/200 サンプル刻みで探す
FINE_MIN_POINTS = 5
FINE_INLIER_SAMPLES = 2.0   # 直線からこのサンプル数以内の地点を、同じ音の経路とみなす
FINE_MAX_RESIDUAL_MS = 0.5  # 直線からの残差（標準偏差）がこれを超えたら失敗にする

WARP_BLOCK = 4096


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def onset_flux(x16: np.ndarray) -> np.ndarray:
    """16kHz モノラルから 10ms ごとの立ち上がりの強さを返す（長い録音でもメモリを食わないよう分けて計算）。"""
    window = np.hanning(FLUX_FFT).astype(np.float32)
    chunk_frames = 6000
    previous = None
    flux = []
    for start in range(0, max(1, len(x16) - FLUX_FFT + 1), FLUX_HOP * chunk_frames):
        seg = x16[start: start + FLUX_HOP * (chunk_frames - 1) + FLUX_FFT]
        if len(seg) < FLUX_FFT:
            break
        frames = sliding_window_view(seg, FLUX_FFT)[::FLUX_HOP] * window
        spec = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
        if previous is not None:
            spec = np.vstack([previous, spec])
        flux.append(np.maximum(np.diff(spec, axis=0), 0).sum(axis=1))
        previous = spec[-1:]
    d = np.concatenate(flux)
    d = d - signal.medfilt(d, 51)
    return ((d - d.mean()) / (d.std() + 1e-9)).astype(np.float32)


def read_mono_16k(path: Path) -> np.ndarray:
    info = sf.info(path)
    parts = [signal.resample_poly(to_mono(block), FLUX_SR, info.samplerate).astype(np.float32)
             for block in sf.blocks(path, blocksize=info.samplerate * 60, dtype="float32")]
    return np.concatenate(parts)


def coarse_offset(song16: np.ndarray, camera_flux: np.ndarray) -> float:
    """song の 0 秒がカメラ録音の何秒にあたるかを返す。"""
    duration = len(song16) / FLUX_SR
    if duration < COARSE_WINDOW_SEC + 20:
        raise ValueError(f"song too short to align: {duration:.1f}s")
    rate = FLUX_SR / FLUX_HOP
    found = []
    for t0 in (10.0, (duration - COARSE_WINDOW_SEC) / 2, duration - COARSE_WINDOW_SEC - 10):
        seg = onset_flux(song16[int(t0 * FLUX_SR): int((t0 + COARSE_WINDOW_SEC) * FLUX_SR)])
        corr = signal.correlate(camera_flux, seg, mode="valid", method="fft")
        found.append(int(np.argmax(corr)) / rate - t0)
    if max(found) - min(found) > COARSE_AGREE_SEC:
        raise RuntimeError(f"この曲はカメラの録音に見つからない（曲の3か所の位置が揃わない: {found}）")
    return float(np.median(found))


def gcc_phat_lag(a: np.ndarray, b: np.ndarray, search: int) -> tuple[float, float]:
    """b の中で a が始まる位置（search を中心にした、サンプル以下の精度）と、ピークの鋭さを返す。

    端の切れ目が偽のピークになるのでテーパーをかけ、白色化は弱め（PHAT_BETA）にする。
    """
    a = a * signal.windows.tukey(len(a), 0.2)
    b = b * signal.windows.tukey(len(b), 0.2)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cross = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    r = np.fft.irfft(cross / (np.abs(cross) ** PHAT_BETA + 1e-12), n)[: 2 * search + 1]
    k = int(np.argmax(r))
    # 放物線の補間は整数側に偏る（最大 0.25 サンプルほど）ので、ピーク付近を sinc 補間して細かく探す
    near = np.arange(max(0, k - SINC_TAPS), min(len(r), k + SINC_TAPS + 1))
    grid = k + np.linspace(-1, 1, 2 * SINC_STEPS + 1)
    fine = np.sinc(grid[:, None] - near[None, :]) @ r[near]
    return float(grid[np.argmax(fine)]) - search, float(r[k] / (np.std(r) + 1e-12))


def measure_lags(song: np.ndarray, camera: np.ndarray, sr: int, camera_start: int) -> list[dict]:
    """song の各地点で、camera[camera_start + n] から何サンプルずれているかを測る。"""
    sos = signal.butter(4, FINE_BAND_HZ, "bandpass", fs=sr, output="sos")
    song_f = signal.sosfiltfilt(sos, to_mono(song))
    camera_f = signal.sosfiltfilt(sos, to_mono(camera))
    length, step, search = int(FINE_WINDOW_SEC * sr), int(FINE_STEP_SEC * sr), int(FINE_SEARCH_SEC * sr)
    points = []
    for t in range(step // 2, len(song_f) - length, step):
        lo = camera_start + t - search
        if lo < 0 or lo + length + 2 * search > len(camera_f):
            continue
        lag, sharpness = gcc_phat_lag(song_f[t: t + length], camera_f[lo: lo + length + 2 * search], search)
        points.append({"center": t + length / 2, "lag": lag, "sharpness": sharpness})
    return points


def fit_drift(points: list[dict], sr: int) -> dict:
    """ずれ = intercept + slope * 位置 の直線を、外れ値を除きながら当てはめる。"""
    pts = [p for p in points if p["sharpness"] >= 0.5 * np.median([q["sharpness"] for q in points])]
    x = np.array([p["center"] for p in pts])
    y = np.array([p["lag"] for p in pts])
    # ライブでは PA と生音で届く時間が違い、区間ごとに数 ms 離れた別の直線に乗ることがある。
    # 2点を通る直線を総当たりし、FINE_INLIER_SAMPLES 以内に乗る点がいちばん多い直線（最も優勢な音の経路）を選ぶ
    best = None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            slope = (y[j] - y[i]) / (x[j] - x[i])
            dist = np.abs(y - (y[i] + slope * (x - x[i])))
            inliers = dist <= FINE_INLIER_SAMPLES
            score = (int(inliers.sum()), -float(dist[inliers].sum()))
            if best is None or score > best[0]:
                best = (score, inliers)
    keep = best[1] if best else np.zeros(len(x), bool)
    if keep.sum() < FINE_MIN_POINTS:
        raise RuntimeError(f"位置合わせに使える地点が少ない（{keep.sum()} / {len(points)} 点）")
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    resid = y - (intercept + slope * x)
    residual_ms = float(np.std(resid[keep]) / sr * 1000)
    if residual_ms > FINE_MAX_RESIDUAL_MS:
        raise RuntimeError(f"位置合わせの残差が大きい: {residual_ms:.2f} ms")
    return {"intercept": float(intercept), "slope": float(slope), "points": int(keep.sum()),
            "total_points": len(points), "residual_ms": residual_ms}


def warp(camera: np.ndarray, start: float, ratio: float, n_out: int) -> np.ndarray:
    """out[n] = camera(start + n * ratio) を、窓かけした小ブロックの小数遅延の重ね合わせで作る。"""
    channels = camera.shape[1]
    hop = WARP_BLOCK // 2
    window = np.hanning(WARP_BLOCK + 1)[:-1]  # 50% 重ねで合計が 1 になる周期ハン窓
    nfft = 2 * WARP_BLOCK
    freqs = np.fft.rfftfreq(nfft)
    out = np.zeros((n_out + 2 * WARP_BLOCK, channels))
    for n0 in range(-hop, n_out, hop):
        u = start + (n0 + WARP_BLOCK / 2) * ratio - WARP_BLOCK / 2
        i0 = int(np.floor(u))
        frac = u - i0
        seg = np.zeros((WARP_BLOCK, channels))
        lo, hi = max(i0, 0), min(i0 + WARP_BLOCK, len(camera))
        if hi > lo:
            seg[lo - i0: hi - i0] = camera[lo:hi]
        shifted = np.fft.irfft(np.fft.rfft(seg, nfft, axis=0) * np.exp(2j * np.pi * freqs * frac)[:, None],
                               nfft, axis=0)[:WARP_BLOCK]
        out[n0 + WARP_BLOCK: n0 + 2 * WARP_BLOCK] += shifted * window[:, None]
    return out[WARP_BLOCK: WARP_BLOCK + n_out]


def align_signal(song: np.ndarray, camera: np.ndarray, sr: int, camera_start: int
                 ) -> tuple[np.ndarray, dict, dict]:
    """camera を song に合わせて (合わせた camera, 当てはめた直線, 合わせた後に測り直したずれ) を返す。

    ずれが動いている窓では相関のピークが少しぼやけるので、一度合わせてから残りのずれを測り直し、2回目で詰める。
    """
    first = fit_drift(measure_lags(song, camera, sr, camera_start), sr)
    start, slope = camera_start + first["intercept"], first["slope"]
    rest = fit_drift(measure_lags(song, warp(camera, start, 1 + slope, len(song)), sr, 0), sr)
    start += rest["intercept"] * (1 + slope)
    slope += rest["slope"] * (1 + slope)
    aligned = warp(camera, start, 1 + slope, len(song))
    check = fit_drift(measure_lags(song, aligned, sr, 0), sr)
    fit = {**first, "start": start, "slope": slope}
    return aligned, fit, check


def align(target: Song, source: Path) -> dict:
    song, sr = sf.read(target.song_wav, dtype="float64", always_2d=True)
    info = sf.info(source)
    if info.samplerate != sr:
        raise ValueError(f"{source}: {info.samplerate}Hz（song.wav は {sr}Hz。先に {sr}Hz で書き出す）")

    offset_sec = coarse_offset(read_mono_16k(target.song_wav), onset_flux(read_mono_16k(source)))
    margin = int(sr * 2)
    first = max(0, int(round(offset_sec * sr)) - margin)
    camera, _ = sf.read(source, dtype="float64", always_2d=True,
                        start=first, stop=min(info.frames, first + len(song) + 2 * margin))
    camera_start = int(round(offset_sec * sr)) - first

    aligned, fit, check = align_signal(song, camera, sr, camera_start)
    gain_db = integrated_lufs(song, sr) - integrated_lufs(aligned, sr)
    aligned *= 10 ** (gain_db / 20)

    sf.write(target.work / "camera.wav", aligned, sr, subtype="FLOAT")
    result = {
        "path": str(source.resolve()),
        "song_starts_at_sec": (first + fit["start"]) / sr,
        "clock_drift_ppm": fit["slope"] * 1e6,
        "fit": fit,
        "after_alignment": {"lag_samples": check["intercept"], "drift_ppm": check["slope"] * 1e6,
                            "residual_ms": check["residual_ms"]},
        "gain_db": gain_db,
    }
    (target.work / "camera.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="songs/ の下の曲名")
    parser.add_argument("source", type=Path, help="同じ演奏を別の機材で録った音声（変更しない）")
    args = parser.parse_args()
    target = Song(args.name)
    if not target.song_wav.exists():
        raise SystemExit(f"{target.song_wav} がない。先に prep.py を実行する")
    print(json.dumps(align(target, args.source), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
