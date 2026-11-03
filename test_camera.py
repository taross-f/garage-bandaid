import numpy as np
import pytest
from scipy import signal

import camera
import mix
from songs import Song

SR = 48000


def bursts(seconds: float, seed: int = 0) -> np.ndarray:
    """拍のような立ち上がりを持つ帯域制限ノイズ。"""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    env = np.zeros(n)
    for start in rng.integers(0, max(1, n - SR // 4), max(1, int(seconds * 3))):
        env[start: start + SR // 4] += np.exp(-np.arange(SR // 4) / (SR / 30))
    sos = signal.butter(4, (200, 6000), "bandpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, rng.normal(size=n) * (env + 0.05))


def fake_camera(x: np.ndarray, start: float, slope: float) -> np.ndarray:
    """camera[start + n * (1 + slope)] == x[n] になるステレオ録音を作る。"""
    stretched = signal.resample(x, int(round(len(x) * (1 + slope))))
    whole = int(np.floor(start))
    n = 1 << int(np.ceil(np.log2(len(stretched) + whole + SR)))
    spectrum = np.fft.rfft(stretched, n) * np.exp(-2j * np.pi * np.fft.rfftfreq(n) * (start - whole))
    shifted = np.concatenate([np.zeros(whole), np.fft.irfft(spectrum, n)])[: whole + len(stretched) + SR]
    return np.stack([shifted, 0.8 * shifted], axis=1)


def test_alignment_recovers_offset_and_clock_drift():
    x = bursts(70)
    start, slope = 2345.37, -15.5e-6
    cam = fake_camera(x, start, slope)

    aligned, fit, check = camera.align_signal(x[:, None], cam, SR, 2345)
    assert fit["start"] == pytest.approx(start, abs=0.05)
    assert fit["slope"] == pytest.approx(slope, abs=0.1e-6)
    assert abs(check["intercept"]) < 0.05
    assert np.corrcoef(aligned[SR:-SR, 0], x[SR:-SR])[0, 1] > 0.99


def test_fit_drift_ignores_outliers():
    points = [{"center": t * SR, "lag": 100 - 0.75 * t, "sharpness": 5.0} for t in range(5, 200, 10)]
    for i in (1, 4, 9):
        points[i]["lag"] += 150
    fit = camera.fit_drift(points, SR)
    assert fit["points"] == len(points) - 3
    assert fit["slope"] * SR == pytest.approx(-0.75, abs=1e-6)


def test_fit_drift_picks_dominant_path_when_lags_jump_between_sections():
    """PA と生音のように、区間ごとに数 ms 離れた平行な直線に乗る場合は、点の多い方を選ぶ。"""
    line = lambda t: 240 - 0.85 * t  # noqa: E731
    offsets = [0] * 6 + [131] * 5 + [-93] * 4 + [0] * 4
    points = [{"center": (5 + 10 * i) * SR, "lag": line(5 + 10 * i) + off, "sharpness": 5.0}
              for i, off in enumerate(offsets)]
    fit = camera.fit_drift(points, SR)
    assert fit["points"] == 10
    assert fit["intercept"] == pytest.approx(240, abs=1e-6)
    assert fit["slope"] * SR == pytest.approx(-0.85, abs=1e-6)


@pytest.fixture
def song(tmp_path, monkeypatch):
    import songs
    monkeypatch.setattr(songs, "SONGS_DIR", tmp_path)
    s = Song("demo")
    s.work.mkdir(parents=True)
    return s


CAMERA_OFF = {"width_db": -60.0, "width_hpf_hz": 150, "air_db": -60.0, "air_hz": 5000}


def test_add_camera_off_keeps_mono_bus(song):
    bus = bursts(2)[:, None]
    assert mix.add_camera(song, bus, SR, CAMERA_OFF) is bus


def test_template_has_camera_section_turned_off(song):
    """新しい曲には camera.wav が無いので、初期値の camera 節は何も足さない設定でなければならない。"""
    import yaml
    from songs import TEMPLATE_YAML
    cfg = yaml.safe_load(TEMPLATE_YAML.read_text())["camera"]
    assert set(cfg) == set(CAMERA_OFF)
    bus = bursts(2)[:, None]
    assert mix.add_camera(song, bus, SR, cfg) is bus


def test_add_camera_width_only_changes_side(song):
    import soundfile as sf
    bus = bursts(3, seed=1)[:, None]
    cam = np.stack([bursts(3, seed=2), bursts(3, seed=3)], axis=1)
    sf.write(song.work / "camera.wav", cam, SR, subtype="FLOAT")

    out = mix.add_camera(song, bus, SR, {**CAMERA_OFF, "width_db": -6.0})
    assert out.shape == (len(bus), 2)
    assert np.allclose((out[:, 0] + out[:, 1]) / 2, bus[:, 0])
    side = mix.hpf((cam[:, :1] - cam[:, 1:]) / 2, SR, 150)[:, 0] * 10 ** (-6 / 20)
    assert np.allclose((out[:, 0] - out[:, 1]) / 2, side, atol=1e-6)


def test_add_camera_requires_aligned_file(song):
    with pytest.raises(FileNotFoundError):
        mix.add_camera(song, bursts(1)[:, None], SR, {**CAMERA_OFF, "air_db": -12.0})
