# 有色金属领域实体识别 — 实验一

本文件夹是**实验一**上课专用内容，从数据标注开始，不含爬虫。

## 目录说明

```text
实验一/
├── 实验手册.md              # 原理 + 步骤 + 结果表
├── README.md                # 本说明
├── requirements.txt
├── config/config.yaml       # 实验超参
├── data/
│   ├── dicts/               # 四类实体词典
│   ├── raw/corpus.txt       # 已准备好的语料
│   ├── processed/           # 标注后的 train/dev/test
│   └── embeddings/          # Word2Vec（运行后生成）
├── scripts/
│   ├── 01_build_dataset.py  # 标注与划分
│   ├── 02_train_embeddings.py
│   └── 03_run_experiments.py
├── src/                     # 实验代码
└── outputs/                 # 结果与图片
```

## 快速开始

在本目录下执行：

```bash
cd 实验一
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_train_embeddings.py
python scripts/03_run_experiments.py
```

详细说明见 **`实验手册.md`**。
