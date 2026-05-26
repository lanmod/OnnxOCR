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

Flask サービスを起動します。

```bash
python app.py
```

主なエンドポイント：

- `/ocr`：汎用 OCR。
- `/plate`：ナンバープレート認識。
- `/table`：表認識。
- `/layout`：レイアウト解析。
- `/layout_markdown`：文書画像を Markdown に変換。

## Docker

```bash
docker build -t ocr-service .
docker run -itd --name onnxocr-service -p 5006:5005 ocr-service
```

## Agent 向け垂直 OCR CLI

OnnxOCR には、Claude Code、Codex などの Agent ツールから直接呼び出しやすい CLI レイヤーがあります。固定テンプレートの業界文書を、安定した JSON フィールドとして抽出するための仕組みです。

```bash
pip install -e .

onnocr list
onnocr list --candidates
onnocr schema transport.train_ticket
onnocr run transport.train_ticket data/samples/scid_train_ticket.jpg --pretty
```

明示的なモジュール形式も利用できます。

```bash
onnocr onnxocr.skill_cli list
onnocr onnxocr.skill_cli run vehicle.plate onnxocr/test_images/license_plate_single_blue.jpg --pretty
```

デフォルトで有効なシナリオは、実画像でスモークテスト済みのものだけです。候補シナリオは `--candidates` で確認できます。

## なぜ大規模モデル Agent だけではないのか

大規模モデルの視覚理解は自由形式の説明や判断に強い一方で、垂直 OCR では次のような工程上の価値が重要になります。

- **再現性**：同じ画像から安定したフィールドを抽出しやすい。
- **オフライン実行**：ONNXRuntime により、社内ネットワークやエッジデバイスでも動かしやすい。
- **低コスト**：大量帳票をすべて外部の視覚モデルに送る必要がありません。
- **監査性**：どのテキストからどのフィールドを抽出したか追跡しやすい。
- **組み合わせやすさ**：OnnxOCR が構造化 JSON を出し、Agent が確認、修正、登録、業務判断を行えます。

## コントリビューション

Issues と Pull Requests を歓迎します。垂直 OCR CLI に貢献する場合は、次の方針を守ってください。

1. 新しいシナリオは、まず候補テンプレートとして追加してください。
2. フィールド ID は英語の `snake_case`、表示名・説明・ラベルは中国語優先で書いてください。必要に応じて英語別名も追加できます。
3. `tests/test_skills.py` に、モデルファイルに依存しない OCR 行ベースの単体テストを追加してください。
4. 実画像サンプルを使う場合は、公開サンプルまたは許可を得た匿名化サンプルだけを使ってください。
5. 個人情報、銀行カード、医療文書、契約書、配送伝票などの実データをそのままコミットしないでください。
6. デフォルトシナリオへ昇格するには、実画像スモークテストと評価記録が必要です。

ローカル検証：

```bash
pip install -e .
onnocr list
onnocr list --candidates
onnocr schema <skill_id> --candidates
onnocr run <skill_id> <image_path> --pretty --candidates
python -B -m pytest tests/test_skills.py -p no:cacheprovider
```

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
