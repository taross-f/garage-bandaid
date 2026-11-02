"""ミックスを画面で比べながら編集するローカルサーバー。

  .venv/bin/python studio.py            # http://127.0.0.1:8765 をブラウザで開く

画面（ui/index.html）は表示と操作だけ。レンダリング・保存はこのPC上の Python が行う。
画面でレンダリングした結果は songs/<name>/versions/<id>/ に版として残り、
「採用」すると songs/<name>/mix.yaml（コメントは保持）と out/ に反映される。
曲の追加（prep.py と同じ切り出し・分離）も画面から裏で実行できる。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import soundfile as sf
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

import prep
import songs
from mix import mix, render
from songs import ROOT, Song, prepared_songs, slugify

UI_FILE = ROOT / "ui" / "index.html"
NAME_RE = re.compile(r"^[\w-]+$")
VERSION_RE = re.compile(r"^v\d{3,}$")
RENDER_LOCK = threading.Lock()

# 曲の追加で選べる録音ファイル: このフォルダ以下（songs/ と隠しフォルダは除く）
SOURCE_ROOT = ROOT
SOURCE_EXTS = {".wav", ".flac", ".aif", ".aiff"}
PREP_LOCK = threading.Lock()
prep_job: dict | None = None  # 分離は CPU を使い切るので同時に1曲だけ


# ---- mix.yaml（コメント保持） ----------------------------------------------

def roundtrip_yaml() -> YAML:
    y = YAML()
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 200
    return y


def apply_values(node, values):
    """ruamel のノードに値だけを反映する（コメントと書式は残す）。新しい構造を返す。"""
    if isinstance(node, CommentedMap) and isinstance(values, dict):
        for key in list(node):
            if key not in values:
                del node[key]
        for key, value in values.items():
            node[key] = apply_values(node[key], value) if key in node else value
        return node
    if isinstance(node, CommentedSeq) and isinstance(values, list):
        del node[len(values):]
        for i, value in enumerate(values):
            if i < len(node):
                node[i] = apply_values(node[i], value)
            else:
                node.append(value)
        return node
    if isinstance(node, float) and isinstance(values, (int, float)) and not isinstance(values, bool):
        # 画面からの JSON では 6.0 が 6 になる。同じ値なら元の表記（6.0）を残し、型も float に揃える
        return node if node == values else float(values)
    return values


def split_comment(raw: str) -> tuple[str, list[str]]:
    """キー直後のコメントを (同じ行の行末コメント, 次のキーの前にあるコメント行) に分ける。"""
    lines = raw.split("\n")
    trailing = lines[0].strip().lstrip("#").strip() if lines[0].strip().startswith("#") else ""
    blocks = "\n".join(lines[1:]).split("\n\n")
    leading = [line.strip().lstrip("#").strip() for line in blocks[-1].splitlines()
               if line.strip().startswith("#")]
    return trailing, leading


def collect_hints(node, prefix: str = "", leading: list[str] | None = None) -> dict[str, str]:
    """mix.yaml のコメントを {"stems.vocals.gain_db": "..."} の形で集める。

    行末コメントと、キーの直前にあるコメント行の両方を拾う。
    """
    hints: dict[str, str] = {}
    if isinstance(node, CommentedMap):
        pending = leading or []
        for key, value in node.items():
            path = f"{prefix}{key}"
            entry = node.ca.items.get(key)
            trailing, next_leading = split_comment(entry[2].value) if entry and entry[2] else ("", [])
            text = "\n".join(t for t in [*pending, trailing] if t)
            if text:
                hints[path] = text
            hints.update(collect_hints(value, f"{path}."))
            pending = next_leading
    elif isinstance(node, CommentedSeq):
        for i, value in enumerate(node):
            entry = node.ca.items.get(i)
            if entry and entry[0] is not None:
                hints[f"{prefix}{i}"] = split_comment(entry[0].value)[0]
            hints.update(collect_hints(value, f"{prefix}{i}."))
    return hints


def file_hints(text: str) -> dict[str, str]:
    data = roundtrip_yaml().load(text)
    header = "".join(t.value for t in (data.ca.comment[1] if data.ca.comment else []))
    return collect_hints(data, leading=split_comment("\n" + header)[1])


def write_config(path: Path, config: dict) -> None:
    """設定ファイルの値だけを config に置き換える（コメントと書式は残す）。"""
    y = roundtrip_yaml()
    data = y.load(path.read_text())
    apply_values(data, config)
    buffer = io.StringIO()
    y.dump(data, buffer)
    if yaml.safe_load(buffer.getvalue()) != config:
        raise RuntimeError(f"{path.name} への書き戻し結果が版の設定と一致しない")
    path.write_text(buffer.getvalue())


def adopt_config(song: Song, config: dict) -> None:
    write_config(song.config, config)


# ---- 版の保存 ---------------------------------------------------------------

def version_dir(song: Song, version_id: str) -> Path:
    if not VERSION_RE.match(version_id):
        raise ValueError(f"invalid version id: {version_id!r}")
    path = song.versions / version_id
    if not path.is_dir():
        raise FileNotFoundError(f"version not found: {version_id}")
    return path


def next_version_id(song: Song) -> str:
    numbers = [int(p.name[1:]) for p in song.versions.glob("v*") if VERSION_RE.match(p.name)]
    return f"v{max(numbers, default=0) + 1:03d}"


def load_version(path: Path) -> dict:
    return {
        "meta": json.loads((path / "meta.json").read_text()),
        "config": json.loads((path / "config.json").read_text()),
        "report": json.loads((path / "report.json").read_text()),
    }


def list_versions(song: Song) -> list[dict]:
    paths = sorted(p for p in song.versions.glob("v*") if VERSION_RE.match(p.name))
    return [load_version(p) for p in paths]


def create_version(song: Song, config: dict, note: str, parent: str | None) -> dict:
    with RENDER_LOCK:
        after, _, sr, report = render(song, config)
        song.versions.mkdir(exist_ok=True)
        version_id = next_version_id(song)
        tmp = song.versions / f".{version_id}.tmp"
        tmp.mkdir()
        sf.write(tmp / "after.flac", after, sr, format="FLAC", subtype="PCM_24")
        (tmp / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
        (tmp / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        (tmp / "meta.json").write_text(json.dumps({
            "id": version_id, "note": note, "parent": parent,
            "created": datetime.now().isoformat(timespec="seconds"),
        }, indent=2, ensure_ascii=False))
        tmp.rename(song.versions / version_id)
    return load_version(song.versions / version_id)


def set_note(song: Song, version_id: str, note: str) -> dict:
    path = version_dir(song, version_id)
    meta = json.loads((path / "meta.json").read_text())
    meta["note"] = note
    (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return load_version(path)


def adopt_version(song: Song, version_id: str) -> dict:
    config = load_version(version_dir(song, version_id))["config"]
    with RENDER_LOCK:
        adopt_config(song, config)
        return mix(song, config)


def set_template(song: Song, version_id: str) -> None:
    """版の設定を新しい曲の初期値（mix.template.yaml）にする。既存の曲は変わらない。

    直前の内容は mix.template.prev.yaml に1世代だけ残す。
    """
    config = load_version(version_dir(song, version_id))["config"]
    template = songs.TEMPLATE_YAML
    shutil.copyfile(template, template.with_name("mix.template.prev.yaml"))
    write_config(template, config)


def song_state(song: Song) -> dict:
    current = yaml.safe_load(song.config.read_text())
    template = yaml.safe_load(songs.TEMPLATE_YAML.read_text())
    versions = list_versions(song)
    return {
        "name": song.name,
        "song_lufs": json.loads((song.work / "diagnosis.json").read_text())["song"]["lufs"],
        "config": current,
        "hints": file_hints(song.config.read_text()),
        "versions": versions,
        "adopted": [v["meta"]["id"] for v in versions if v["config"] == current],
        "template": [v["meta"]["id"] for v in versions if v["config"] == template],
        "template_config": template,
    }


# ---- 曲の追加 ---------------------------------------------------------------

def list_sources() -> list[dict]:
    used: dict[str, list[str]] = {}
    for song in prepared_songs():
        path = json.loads((song.work / "source.json").read_text())["path"]
        used.setdefault(path, []).append(song.name)
    found = []
    for dirpath, dirnames, filenames in os.walk(SOURCE_ROOT):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith((".", "__")) and Path(dirpath) / d != songs.SONGS_DIR)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if filename.startswith(".") or path.suffix.lower() not in SOURCE_EXTS:
                continue
            info = sf.info(path)
            found.append({
                "path": path.relative_to(SOURCE_ROOT).as_posix(),
                "duration": info.duration, "samplerate": info.samplerate, "channels": info.channels,
                "name": slugify(path.stem),
                "used_by": used.get(str(path.resolve()), []),
            })
    return found


def optional_seconds(value) -> float | None:
    return None if value is None or value == "" else float(value)


def start_prep(source: str, name: str, start: float | None, end: float | None) -> dict:
    """曲の追加を裏で始める。進み具合は prep_job を見る。"""
    global prep_job
    sources = {s["path"]: s for s in list_sources()}
    if source not in sources:
        raise FileNotFoundError(f"source not found: {source}")
    name = name.strip() or sources[source]["name"]
    if not NAME_RE.match(name):
        raise ValueError(f"曲名は英数字・_・- だけ: {name!r}")
    prep.padded_range(start, end, sources[source]["duration"])  # 範囲がおかしければここで ValueError
    song = Song(name)
    with PREP_LOCK:
        if prep_job and prep_job["status"] == "running":
            raise ValueError(f"{prep_job['name']} を準備中。終わってから追加する")
        if (song.work / "diagnosis.json").exists():
            raise ValueError(f"{name} は既にある。別の曲名にする")
        job = {"name": name, "source": source, "start": start, "end": end, "status": "running",
               "stage": "開始", "fraction": None, "started": time.time(), "finished": None, "error": None}
        prep_job = job
    # diagnosis.json の無い work/ は中断された準備の残り。作り直す
    threading.Thread(target=run_prep, args=(job, SOURCE_ROOT / source, song.work.exists()), daemon=True).start()
    return dict(job)


def run_prep(job: dict, source: Path, force: bool) -> None:
    def progress(stage: str, fraction: float | None) -> None:
        job["stage"], job["fraction"] = stage, fraction

    try:
        prep.new_song(source, job["name"], job["start"], job["end"], force=force, progress=progress)
        job["status"] = "done"
    except BaseException as e:  # de-clip 判定の SystemExit も画面に出す
        traceback.print_exc()
        job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
    finally:
        job["finished"] = time.time()


def delete_song(song: Song) -> None:
    """songs/<name>/ を丸ごと消す（mix.yaml・ステム・版・out/）。元の録音ファイルには触れない。"""
    with PREP_LOCK:
        if prep_job and prep_job["status"] == "running" and prep_job["name"] == song.name:
            raise ValueError(f"{song.name} は準備中なので消せない")
        with RENDER_LOCK:
            shutil.rmtree(song.dir)


# ---- HTTP -------------------------------------------------------------------

def get_song(name: str) -> Song:
    if not NAME_RE.match(name):
        raise ValueError(f"invalid song name: {name!r}")
    song = Song(name)
    if song not in prepared_songs():
        raise FileNotFoundError(f"song not prepared: {name}")
    return song


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith("/audio/"):
            super().log_message(fmt, *args)

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def dispatch(self, method: str):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        try:
            self.route(method, parts)
        except (ValueError, KeyError, TypeError) as e:
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as e:
            self.send_json({"error": str(e)}, HTTPStatus.NOT_FOUND)
        except Exception as e:  # 画面に原因を出すため。握りつぶさずログにも出す
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def route(self, method: str, parts: list[str]):
        match method, parts:
            case "GET", []:
                self.send_file(UI_FILE, "text/html; charset=utf-8")
            case "GET", ["api", "songs"]:
                self.send_json([s.name for s in prepared_songs()])
            case "GET", ["api", "sources"]:
                self.send_json(list_sources())
            case "GET", ["api", "prep"]:
                self.send_json(prep_job and dict(prep_job))
            case "POST", ["api", "prep"]:
                body = self.read_json()
                self.send_json(start_prep(str(body["source"]), str(body.get("name") or ""),
                                          optional_seconds(body.get("start")), optional_seconds(body.get("end"))))
            case "GET", ["api", "songs", name]:
                self.send_json(song_state(get_song(name)))
            case "POST", ["api", "songs", name, "versions"]:
                body = self.read_json()
                song = get_song(name)
                version = create_version(song, body["config"], str(body.get("note", "")), body.get("parent"))
                self.send_json(version)
            case "POST", ["api", "songs", name, "versions", vid, "note"]:
                self.send_json(set_note(get_song(name), vid, str(self.read_json()["note"])))
            case "POST", ["api", "songs", name, "versions", vid, "template"]:
                set_template(get_song(name), vid)
                self.send_json({"template": vid})
            case "POST", ["api", "songs", name, "versions", vid, "adopt"]:
                self.send_json(adopt_version(get_song(name), vid))
            case "DELETE", ["api", "songs", name]:
                delete_song(get_song(name))
                self.send_json({"deleted": name})
            case "DELETE", ["api", "songs", name, "versions", vid]:
                shutil.rmtree(version_dir(get_song(name), vid))
                self.send_json({"deleted": vid})
            case "GET", ["audio", name, "song.wav"]:
                self.send_file(get_song(name).song_wav, "audio/wav")
            case "GET", ["audio", name, vid, "after.flac"]:
                self.send_file(version_dir(get_song(name), vid) / "after.flac", "audio/flac")
            case _:
                raise FileNotFoundError(f"no route: {method} {self.path}")

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_DELETE(self):
        self.dispatch("DELETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"studio: {url}  (Ctrl+C で終了)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
