# Reproduction data pipeline framework

这个文件夹是按“项目代码仓库”而不是 notebook 来拆的，重点覆盖你负责的部分：

- 历史成分股获取与重建
- 价格数据清洗
- 750/250 rolling split
- 标签生成
- 防止 look-ahead bias
- 组织最终训练样本和交易样本
- 输出 sample count summary

## 目录结构

```text
repro_lstm_pipeline/
├── config/
│   └── example_config.yaml
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── repro_pipeline/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── constituents.py
│       ├── dataset_builder.py
│       ├── io_utils.py
│       ├── labels.py
│       ├── prices.py
│       ├── splits.py
│       └── summary.py
├── requirements.txt
└── pyproject.toml
```

## 你需要准备的原始数据

### 1) 月末成分股列表 `constituents_month_end.csv`
最少字段：

- `date`: 月末日期
- `ticker`: 股票代码
- `in_index`: 是否在指数中，1/0

建议一行表示一个 ticker 在某个月末是否属于指数。

### 2) 日频价格数据 `prices_daily.csv`
最少字段：

- `date`
- `ticker`
- `price`

如果你有 total return index / adjusted close，优先放在 `price`。
论文里使用的是 **daily total return indices**，因为它天然处理分红、拆股等 corporate actions。

## 核心与论文对齐的地方

- 训练/交易 study period：**750 个交易日训练 + 250 个交易日交易**
- 全样本切成 **23 个非重叠交易窗口**
- 成分股按**训练期最后一天**的指数成员来确定当期股票池
- LSTM 输入特征使用 **1 日收益率标准化后形成长度 240 的滚动序列**
- 标签为 **t+1 日收益率是否高于当日横截面中位数**
- 标准化参数（均值、标准差）**只能用训练集估计**，不能看未来

## 快速开始

```bash
cd repro_lstm_pipeline
pip install -r requirements.txt

python scripts/run_pipeline.py --config config/config.yaml
```

## 输出

运行后默认会在 `output/` 下生成：

- `master_dataset.parquet`：clean master dataset
- `study_periods.parquet`：每个 study period 的日期边界
- `sample_count_summary.csv`：每个 study period 的样本统计
- `constituent_reconstruction_note.md`：成分股重建说明
- `train_samples_period_XX.parquet`：某期训练样本
- `trade_samples_period_XX.parquet`：某期交易样本

## 注意

这份代码是一个**稳健的复现框架**，对接真实数据库时你只需要改：

- 配置文件里的列名/路径
- `constituents.py` 中如果你的成分股原始表结构不同
- `prices.py` 中如果你的价格字段不是单一 `price`



## WRDS raw data download

Use the dedicated downloader instead of manual CSV export:

```bash
export WRDS_PASSWORD='your_password'
python scripts/download_wrds_crsp_sp500.py   --wrds-username your_wrds_username   --output-dir data/raw
```

This script downloads:

- `constituents_month_end.csv`: month-end S&P 500 snapshots from CRSP
- `prices_daily.csv`: daily CRSP prices for all securities that were ever in the S&P 500 during the sample window

Why this is aligned with the paper:

- month-end constituent snapshots are later shifted forward to represent membership in the subsequent month
- all-ever-constituents extraction avoids survivorship bias
- downstream standardization must be estimated from the training window only
