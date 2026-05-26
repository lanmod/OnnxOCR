# OnnxOCR

このプロジェクトが役に立ったら、右上の **Star** で応援してください。

![onnx_logo](onnxocr/test_images/onnxocr_logo.png)

**ONNXRuntime ベースの高性能・多言語 OCR プロジェクト**

![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)
![AtomGit Star](https://atomgit.com/OnnxOCR/OnnxOCR/star/badge.svg)

[English](./Readme.md) | [简体中文](./Readme_cn.md) | 日本語

## 更新履歴

- **2026.05.01**
  1. ONNX ベースの車両ナンバープレート検出・認識を追加。
  2. RapidTable ベースの ONNX 表認識を追加。
  3. RapidLayout ベースの中国語・英語レイアウト解析を追加。
  4. RapidDoc ベースの文書レイアウト解析と Markdown 出力を追加。
  5. `/plate`、`/table`、`/layout`、`/layout_markdown` などの HTTP API を追加。

- **2025.05.21**
  1. PP-OCRv5 モデルを追加。簡体字中国語、繁体字中国語、中国語ピンイン、英語、日本語を 1 つのモデルで扱えます。
  2. PP-OCRv4 と比べて全体的な認識精度を改善。
  3. 認識結果は PaddleOCR 3.0 と整合するようにしています。

## 特長

1. **学習フレームワーク不要**：デプロイしやすい汎用 OCR プロジェクトです。
2. **クロスアーキテクチャ対応**：PaddleOCR 由来の ONNX モデルを ARM / x86 環境に展開できます。
3. **統一された推論エンジン**：すべての ONNX モデルは `onnxocr/inference_engine.py` 経由で ONNXRuntime Session を作成します。
4. **多言語対応**：1 つのモデルで 5 種類の文字体系をサポートします。
5. **ソースレベル統合**：`rapid_layout`、`rapid_table`、`rapid_doc` を `onnxocr/` パッケージ内に統合しています。
6. **ハードウェア適配が容易**：GPU / NPU などの下流プロバイダ対応は、統一推論エンジンを中心に進められます。

## 環境構築

```bash
python>=3.8
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

補足：

- リポジトリには、`tests/test_general_ocr.py` に必要な PP-OCRv5 汎用 OCR モデルだけが標準で含まれます。
- ナンバープレート認識、表認識、レイアウト解析、方向分類、RapidDoc Markdown 出力などの拡張モデルはサイズが大きいため、必要に応じてダウンロードしてください。
- 中国本土以外のユーザーには [HuggingFace](https://huggingface.co/jingsongliu/onnxocr_model/tree/main) を推奨します。
- 中国本土では [ModelScope](https://www.modelscope.cn/models/supersong/onnxocr_model/tree/master/models) または `https://hf-mirror.com` が使えます。

## モデルのダウンロード

HuggingFace からダウンロードする場合：

```bash
python scripts/download_models.py --source huggingface
```

ModelScope からダウンロードする場合：

```bash
python scripts/download_models.py
```

ローカルのモデル配置だけを確認する場合：

```bash
python scripts/download_models.py --check-only
```

## ワンクリック実行

```bash
python test_ocr.py
```

`test_ocr.py` はデフォルトでは汎用 OCR のみを実行します。車両ナンバー、表認識、レイアウト解析、RapidDoc Markdown 出力の例はコメントアウトされています。対応モデルをダウンロードした後、必要に応じてコメントを外してください。

機能別テスト：

```bash
python tests/test_general_ocr.py
python tests/test_license_plate_ocr.py
python tests/test_table_ocr.py
python tests/test_layout_analysis.py
python tests/test_layout_markdown.py
```

生成ファイルは `result_img/` に出力されます。このディレクトリは git では無視されます。

## 汎用 OCR

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)
print(result)
```

## 日本語 OCR 例

PP-OCRv5 の汎用 OCR モデルは日本語も認識できます。リポジトリ内の `japan_2.jpg` を使って、すぐに日本語 OCR を確認できます。

![日本語 OCR サンプル](onnxocr/test_images/japan_2.jpg)

```python
import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

img = cv2.imread("onnxocr/test_images/japan_2.jpg")
model = ONNXPaddleOcr(use_angle_cls=False, use_gpu=False)
result = model.ocr(img)

for line in result[0]:
    text, score = line[1]
    print(text, score)
```

実行例：

```text
もちもち 0.9998
天然の 0.9999
とろっと後味のよい 0.9945
濃厚な 0.9442
味わい深い 0.9887
なめらかな 0.9920
焼きたて 0.9996
```

もう 1 つの小さな日本語サンプルもあります。

```python
img = cv2.imread("onnxocr/test_images/japan_1.jpg")
result = model.ocr(img)
print(result)
```

## ナンバープレート認識

ナンバープレート認識は `ONNXPaddleOcr` のオプション機能として統合されています。既存の汎用 OCR の使い方は変わりません。

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2PlateImg

plate_model = ONNXPaddleOcr(
    use_angle_cls=True,
    use_gpu=False,
    use_plate_recognition=True,
    plate_min_score=0.4,
)
plate_result = plate_model.ocr(img)
sav2PlateImg(img, plate_result, name="./result_img/test_plate_vis.jpg")
```

モデルファイル：

```text
onnxocr/models/license_plate/car_plate_detect.onnx
onnxocr/models/license_plate/plate_rec.onnx
```

## 表認識

表認識は RapidTable を統合しています。汎用 OCR の検出・認識結果を再利用し、表構造を復元して HTML、セル枠、論理行列座標を出力します。

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2TableImg

table_model = ONNXPaddleOcr(
    use_angle_cls=True,
    use_gpu=False,
    use_table_recognition=True,
    table_model_type="slanet_plus",
)
table_result = table_model.ocr(img)
print(table_result["html"])
sav2TableImg(img, table_result, name="./result_img/test_table_vis.jpg")
```

## 中国語 / 英語レイアウト解析

レイアウト解析は RapidLayout を統合しています。タイトル、本文、表、図、ヘッダー、フッターなどの文書要素を検出します。

```python
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2LayoutImg

layout_model = ONNXPaddleOcr(
    use_gpu=False,
    use_layout_analysis=True,
    layout_model_type="pp_layout_cdla",
)
layout_result = layout_model.ocr(img)
sav2LayoutImg(img, layout_result, name="./result_img/test_layout_vis.jpg")
```

英語レイアウト解析では `layout_model_type="pp_layout_publaynet"` を指定してください。

## 文書から Markdown へ

RapidDoc を使って、文書画像のタイトル、段落、表、図などを解析し、Markdown ファイルとして保存できます。

```python
from onnxocr.layout_markdown import LayoutMarkdownConverter

converter = LayoutMarkdownConverter(
    layout_model_type="pp_doclayoutv2",
    formula_enable=False,
    table_enable=True,
)
result = converter.convert_file(
    "onnxocr/test_images/layout_cdla.jpg",
    output_md_path="./result_img/test_layout_markdown.md",
)
print(result["markdown_path"])
```

## 推論エンジンの適配

汎用 OCR、ナンバープレート認識、表認識、レイアウト解析、RapidDoc 文書解析は、すべて `onnxocr/inference_engine.py` から ONNXRuntime Session を作成します。

下流の GPU / NPU プロバイダに適配する場合は、まず次の関数から確認してください。

```python
from onnxocr.inference_engine import create_session
```

## API サービス

`app-service.py` はデプロイ向けの JSON API サービスです。`webui.py` はブラウザ UI サービスです。API と UI を分けることで、デプロイ、テスト、保守がしやすくなります。

API サービスを起動します。

```bash
python app-service.py
```

環境変数：

- `ONNXOCR_PORT`：サービスポート。デフォルトは `5005`。
- `ONNXOCR_USE_GPU`：`1` / `true` を指定すると GPU provider を使います。
- `ONNXOCR_DEBUG`：`1` / `true` を指定すると Flask debug を有効にします。
- `ONNXOCR_OUTPUT_DIR`：Markdown とアセットの出力先。デフォルトは `result_img`。
- `ONNXOCR_MAX_UPLOAD_MB`：最大アップロードサイズ。デフォルトは `200` MB。

主なエンドポイントは JSON base64 画像と multipart ファイルアップロードに対応します。

- `/health`：ヘルスチェック。
- `/ocr`：汎用 OCR。
- `/plate`：ナンバープレート認識。
- `/table`：表認識。
- `/layout`：レイアウト解析。
- `/layout_markdown`：文書画像を Markdown に変換。

JSON 例：

```bash
curl -X POST http://127.0.0.1:5005/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"<base64-image>\"}"
```

multipart 例：

```bash
curl -X POST http://127.0.0.1:5005/ocr \
  -F "image=@onnxocr/test_images/japan_2.jpg"
```

WebUI を起動します。

```bash
python webui.py
```

## Docker

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

Docker build では `.dockerignore` により、キャッシュ、生成結果、任意の大きなモデルファイルを除外します。ナンバープレート、表、レイアウト、RapidDoc などの拡張モデルは、必要に応じてコンテナ内でダウンロードするか、実行時にマウントしてください。

## プロジェクト構成

```text
app-service.py                 # デプロイ向け JSON API サービス
webui.py                       # ブラウザ UI サービス
test_ocr.py                    # ローカル OCR デモ
requirements.txt               # 実行時依存
Dockerfile                     # API サービス用コンテナ
.dockerignore                  # キャッシュ、出力、大きなモデルを除外
Readme.md                      # 英語ドキュメント
Readme_cn.md                   # 中国語ドキュメント
Readme_ja.md                   # 日本語ドキュメント
onnxocr/
  inference_engine.py          # 統一 ONNXRuntime 入口
  onnx_paddleocr.py            # 公開 Python API
  predict_det.py               # 汎用 OCR 検出
  predict_rec.py               # 汎用 OCR 認識
  orientation.py               # RapidOrientation ONNX 適配
  license_plate.py             # ナンバープレート認識
  table_recognition.py         # 表認識
  layout_recognition.py        # レイアウト解析
  layout_markdown.py           # RapidDoc Markdown 出力
  rapid_layout/                # RapidLayout 統合
  rapid_table/                 # RapidTable 統合
  rapid_doc/                   # RapidDoc 統合
  models/                      # ローカル ONNX モデル
scripts/
  download_models.py           # 任意モデルのダウンロード/確認
tests/                         # 機能テストと API スモークテスト
```

## コントリビューション

Issues と Pull Requests を歓迎します。オープンソース利用者が README の手順どおりに試せるよう、次のルールを守ってください。

1. API 専用の処理は `app-service.py`、ブラウザ UI の処理は `webui.py` に置いてください。
2. `result_img/`、`results/`、`uploads/`、ローカルキャッシュの生成物をコミットしないでください。
3. 個人情報、銀行カード、医療文書、契約書、配送伝票、秘密鍵などをコミットしないでください。
4. テストと文書サンプルには、公開サンプル、公式サンプル、または許可を得た匿名化サンプルだけを使ってください。
5. 任意モデルが必要な機能では、必要なモデルファイルとダウンロード手順を明記してください。
6. README に書くコマンドは、現在のリポジトリに実装が存在するものだけにしてください。

Pull Request 前の推奨チェック：

```bash
python -B -m pytest tests/test_app_service.py -p no:cacheprovider
python tests/test_general_ocr.py
python tests/test_license_plate_ocr.py
python tests/test_table_ocr.py
python tests/test_layout_analysis.py
python tests/test_layout_markdown.py
```

一部のテストは任意モデルファイルを必要とします。ローカルで実行できない場合は、Pull Request に不足しているモデルファイルを書いてください。

## 表示例

| Example 1 | Example 2 |
|-----------|-----------|
| ![](result_img/r1.png) | ![](result_img/r2.png) |

| 日本語 OCR サンプル |
|--------------------|
| ![](onnxocr/test_images/japan_2.jpg) |

## 連絡先

### OnnxOCR コミュニティ

![WeChat Group](onnxocr/test_images/微信群.jpg)

![QQ Group](onnxocr/test_images/QQ群.jpg)

## 謝辞

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) の技術サポートとモデル参照に感謝します。

[RapidAI](https://github.com/RapidAI) オープンソースコミュニティ、特に [RapidTable](https://github.com/RapidAI/RapidTable)、[RapidLayout](https://github.com/RapidAI/RapidLayout)、[RapidDoc](https://github.com/RapidAI/RapidDoc)、[RapidOrientation](https://github.com/RapidAI/RapidOrientation) の優れたモデル、コード、エンジニアリングに感謝します。

## Star 履歴

[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)
