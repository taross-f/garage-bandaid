# garage-bandaid

客席で録ったライブ録音の応急処置キット。
1曲ずつ切り出して [demucs](https://github.com/facebookresearch/demucs)（htdemucs）でボーカル・ドラム・ベース・その他に分離し、ステムごとに EQ・コンプをかけ直して、-14 LUFS / True Peak -1 dBTP に整える。
ブラウザの画面で複数のミックス案を同時再生し、再生位置そのままで切り替えながら詰めていける。

## 必要なもの

- Python 3.11
- ffmpeg（切り出しと 48kHz 変換に使う）
- CPU で分離する。1曲（4分前後）で約4〜8分

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 使い方

### 画面で使う（おすすめ）

```bash
.venv/bin/python studio.py
```

`http://127.0.0.1:8765` が開く。サーバーはこの PC の中だけで待ち受け、音声は外に出ない。

- **＋ 曲を追加**: フォルダ内の録音（wav / flac / aiff）を選んで切り出し・診断・分離する
- **編集 → レンダリング**: 値を変えて版（`songs/<曲>/versions/vNNN/`）を作る
- **聴き比べ**: `0` が before（元音源を同じラウドネスに揃えただけ）、`1`〜`9` が版。波形をドラッグでループ
- **採用**: 版の設定を `songs/<曲>/mix.yaml` に書き戻し、`out/after.wav` を作り直す（コメントは残る）
- **初期値に / 初期値を読み込む**: 版の設定を新しい曲の初期値（`mix.template.yaml`）にする / 初期値を他の曲に当てる

### コマンドで使う

```bash
# 切り出し・診断・分離（曲ごとに1回）。前後3秒の余裕を付けて切り出す
.venv/bin/python prep.py "recordings/live.wav" --name my_song --start 612 --end 845

# songs/my_song/mix.yaml を編集してミックス（分離はやり直さない）
.venv/bin/python mix.py my_song
.venv/bin/python mix.py --all
```

## 出力

`songs/<曲>/` に作られる。

| パス | 中身 |
|---|---|
| `work/song.wav` | 切り出した元音源（48kHz / 24bit） |
| `work/stems/*.wav` | 分離したステム |
| `work/diagnosis.json` | クリップ率・LUFS・True Peak・帯域バランス |
| `mix.yaml` | ミックス設定。各項目に意味と目安をコメントで書いてある |
| `out/after.wav` | 仕上がり（48kHz / 24bit、song.wav と同じ長さ・開始位置） |
| `out/before.wav` | 元音源をラウドネスだけ揃えたもの（比較用） |
| `out/report.json` | before / after の数値 |

## 処理の中身

- ステム: HPF（ゼロ位相）→ ピーキング / ローシェルフ EQ → コンプ → ゲイン
- 元音源を少し混ぜて分離ノイズを隠す（`source_blend_db`）
- マスター: HPF 30Hz → EQ → バスコンプ → ラウドネス合わせ → 4倍オーバーサンプルの True Peak リミッター

## テスト

```bash
.venv/bin/python -m pytest -q
```
