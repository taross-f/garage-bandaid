import json
import time

import numpy as np
import pytest
import soundfile as sf
import yaml

import songs
import studio
from songs import Song

CONFIG_TEXT = """\
# header line

# blend の説明
source_blend_db: -6.0

stems:
  vocals:
    gain_db: 3.0          # 歌の主ツマミ
    eq:
      - {freq: 350, gain_db: -2.0, q: 1.0}   # こもり取り
      - {freq: 3000, gain_db: 2.5, q: 1.0}   # 明瞭度
master:
  target_lufs: -14.0      # 目標
"""


@pytest.fixture
def song(tmp_path, monkeypatch):
    monkeypatch.setattr(songs, "SONGS_DIR", tmp_path)
    s = Song("demo")
    s.stems.mkdir(parents=True)
    s.config.write_text(CONFIG_TEXT)
    (s.work / "diagnosis.json").write_text(json.dumps({"song": {"lufs": -15.6}}))
    return s


def test_adopt_keeps_comments_and_layout(song):
    config = yaml.safe_load(CONFIG_TEXT)
    config["stems"]["vocals"]["gain_db"] = 5.0
    config["stems"]["vocals"]["eq"][0]["gain_db"] = -3.5
    studio.adopt_config(song, config)
    text = song.config.read_text()
    assert yaml.safe_load(text) == config
    assert "    gain_db: 5.0          # 歌の主ツマミ" in text
    assert "      - {freq: 350, gain_db: -3.5, q: 1.0}   # こもり取り" in text
    assert "# blend の説明" in text


def test_adopt_keeps_float_notation_for_json_integers(song):
    config = json.loads(json.dumps(yaml.safe_load(CONFIG_TEXT)).replace("3.0", "3").replace("-6.0", "-6"))
    config["stems"]["vocals"]["eq"][1]["gain_db"] = 4
    studio.adopt_config(song, config)
    text = song.config.read_text()
    assert "source_blend_db: -6.0\n" in text
    assert "    gain_db: 3.0          # 歌の主ツマミ" in text
    assert "{freq: 3000, gain_db: 4.0, q: 1.0}" in text


def test_adopt_handles_added_and_removed_bands(song):
    config = yaml.safe_load(CONFIG_TEXT)
    config["stems"]["vocals"]["eq"] = [{"freq": 350, "gain_db": -2.0, "q": 1.0},
                                       {"type": "low_shelf", "freq": 200, "gain_db": -3.0, "q": 0.7},
                                       {"freq": 8000, "gain_db": 1.0, "q": 0.8}]
    studio.adopt_config(song, config)
    assert yaml.safe_load(song.config.read_text()) == config
    config["stems"]["vocals"]["eq"] = config["stems"]["vocals"]["eq"][:1]
    studio.adopt_config(song, config)
    assert yaml.safe_load(song.config.read_text()) == config


def test_hints_include_trailing_and_leading_comments(song):
    hints = studio.file_hints(song.config.read_text())
    assert hints["source_blend_db"] == "blend の説明"
    assert hints["stems.vocals.gain_db"] == "歌の主ツマミ"
    assert hints["stems.vocals.eq.1"] == "明瞭度"
    assert hints["master.target_lufs"] == "目標"


def test_versions_are_numbered_and_adoptable(song, monkeypatch):
    def fake_render(target, config):
        after = np.zeros((4800, 1))
        return after, after, 48000, {"metrics": {}, "gain": config["stems"]["vocals"]["gain_db"]}

    monkeypatch.setattr(studio, "render", fake_render)
    monkeypatch.setattr(studio, "mix", lambda target, config: {"mixed": True})
    config = yaml.safe_load(CONFIG_TEXT)
    v1 = studio.create_version(song, config, "first", None)
    config["stems"]["vocals"]["gain_db"] = 6.0
    v2 = studio.create_version(song, config, "louder", "v001")
    assert [v1["meta"]["id"], v2["meta"]["id"]] == ["v001", "v002"]
    assert (song.versions / "v002" / "after.flac").exists()

    state = studio.song_state(song)
    assert [v["meta"]["id"] for v in state["versions"]] == ["v001", "v002"]
    assert state["adopted"] == ["v001"]

    studio.adopt_version(song, "v002")
    assert studio.song_state(song)["adopted"] == ["v002"]


def test_version_ids_are_validated(song):
    with pytest.raises(ValueError):
        studio.version_dir(song, "../work")
    with pytest.raises(FileNotFoundError):
        studio.version_dir(song, "v999")


@pytest.fixture
def sources(song, tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "rec").mkdir(parents=True)
    sf.write(root / "rec" / "live take.wav", np.zeros((48000 * 10, 1)), 48000)
    (root / ".venv").mkdir()
    sf.write(root / ".venv" / "hidden.wav", np.zeros((480, 1)), 48000)
    (root / "rec" / "notes.txt").write_text("x")
    monkeypatch.setattr(studio, "SOURCE_ROOT", root)
    (song.work / "source.json").write_text(json.dumps({"path": str((root / "rec" / "live take.wav").resolve())}))
    return root


def test_list_sources_skips_hidden_dirs_and_marks_used(sources):
    found = studio.list_sources()
    assert [s["path"] for s in found] == ["rec/live take.wav"]
    assert found[0]["name"] == "live_take"
    assert found[0]["used_by"] == ["demo"]
    assert found[0]["duration"] == pytest.approx(10.0)


def wait_job():
    for _ in range(200):
        if studio.prep_job["status"] != "running":
            return studio.prep_job
        time.sleep(0.01)
    raise AssertionError("prep job did not finish")


def test_start_prep_runs_new_song_in_background(sources, monkeypatch):
    calls = []

    def fake_new_song(source, name, start, end, force, progress):
        progress("分離", 0.5)
        calls.append((source, name, start, end, force))

    monkeypatch.setattr(studio.prep, "new_song", fake_new_song)
    job = studio.start_prep("rec/live take.wav", "", 1.0, 5.0)
    assert job["name"] == "live_take"
    assert wait_job()["status"] == "done"
    assert calls == [(sources / "rec" / "live take.wav", "live_take", 1.0, 5.0, False)]


def test_start_prep_reports_failures(sources, monkeypatch):
    def failing(*args, **kwargs):
        raise SystemExit("de-clip が必要")

    monkeypatch.setattr(studio.prep, "new_song", failing)
    studio.start_prep("rec/live take.wav", "x", None, None)
    job = wait_job()
    assert job["status"] == "failed" and "de-clip" in job["error"]


@pytest.mark.parametrize("source,name,start,end,error", [
    ("../secret.wav", "x", None, None, FileNotFoundError),
    ("rec/live take.wav", "a/b", None, None, ValueError),
    ("rec/live take.wav", "demo", None, None, ValueError),   # 既存の曲
    ("rec/live take.wav", "x", 30.0, None, ValueError),     # 範囲がファイル外
])
def test_start_prep_validates_input(sources, source, name, start, end, error):
    with pytest.raises(error):
        studio.start_prep(source, name, start, end)


def test_delete_song_removes_folder_but_not_source(sources):
    source = sources / "rec" / "live take.wav"
    song = studio.get_song("demo")
    studio.delete_song(song)
    assert not song.dir.exists()
    assert source.exists()
    with pytest.raises(FileNotFoundError):
        studio.get_song("demo")


def test_delete_song_refuses_while_preparing(song, monkeypatch):
    monkeypatch.setattr(studio, "prep_job", {"name": "demo", "status": "running"})
    with pytest.raises(ValueError):
        studio.delete_song(song)
    assert song.dir.exists()


def test_set_template_writes_version_config_and_keeps_backup(song, tmp_path, monkeypatch):
    template = tmp_path / "mix.template.yaml"
    template.write_text(CONFIG_TEXT)
    monkeypatch.setattr(songs, "TEMPLATE_YAML", template)
    monkeypatch.setattr(studio, "render", lambda target, config: (np.zeros((480, 1)),) * 2 + (48000, {}))
    config = yaml.safe_load(CONFIG_TEXT)
    config["stems"]["vocals"]["gain_db"] = 6.0
    studio.create_version(song, config, "", None)
    before = song.config.read_text()

    studio.set_template(song, "v001")
    assert yaml.safe_load(template.read_text()) == config
    assert "# 歌の主ツマミ" in template.read_text()
    assert (tmp_path / "mix.template.prev.yaml").read_text() == CONFIG_TEXT
    assert song.config.read_text() == before
    state = studio.song_state(song)
    assert state["template"] == ["v001"]
    assert state["template_config"] == config
