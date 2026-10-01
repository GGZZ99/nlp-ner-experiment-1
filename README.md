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

## Conda 环境（推荐，勿用 Windows Store 自带 Python）

本实验依赖 **NumPy 1.x + Gensim** 加载 Word2Vec；系统自带的 Python 3.11 若装了 NumPy 2.x，会在 `04_predict.py` 报错。

**方式 A — 一键脚本（需已安装 Anaconda / Miniconda）：**

```powershell
cd 实验一
powershell -ExecutionPolicy Bypass -File scripts/setup_conda_env.ps1
conda activate nlp-ner-exp1
```

**方式 B — 手动：**

```powershell
cd 实验一
conda env create -f environment.yml
conda activate nlp-ner-exp1
```

若 `conda` 不在 PATH，可用：`D:\anaconda2024\Scripts\conda.exe` 或 `%USERPROFILE%\miniconda3\Scripts\conda.exe`。

## 快速开始

在本目录下执行（请先 `conda activate nlp-ner-exp1`）：

```bash
cd 实验一
python scripts/01_build_dataset.py
python scripts/02_train_embeddings.py
python scripts/03_run_experiments.py
python scripts/04_predict.py --text "江西铜业提升电解铜产量"
```

详细说明见 **`实验手册.md`**。
