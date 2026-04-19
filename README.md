# Reproduction data pipeline framework

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

## 原始数据

### 1) 月末成分股列表 `constituents_month_end.csv`

- `date`: 月末日期
- `permno`: 公司唯一标识（避免股票代码更名问题）
- `ticker`: 股票代码
- `in_index`: 是否在指数中，1/0

### 2) 日频价格数据 `prices_daily.csv`
- `date`: 交易日期
- `permno`: 公司唯一标识（避免股票代码更名问题）
- `ticker`: 股票代码
- `prc`: 收盘价
- `ret`: 收益率（含分红）——同论文一致：returns are cum-dividend prices and account for all relevant corporate actions and stock splits


## 数据准备
- 训练/交易 study period：**750 个交易日训练 + 250 个交易日交易**
- 全样本切成 **23 个非重叠交易窗口**
- 成分股按**训练期最后一天**的指数成员来确定当期股票池
- LSTM 输入特征使用 **1日收益率标准化后形成长度240的滚动序列**
- 标签为 **t+1日收益率是否高于当日横截面中位数（0/1变量）**
- 标准化参数（均值、标准差）

## 运行

### WRDS raw data download
```bash
cd repro_lstm_pipeline
export WRDS_PASSWORD='7vG&3E^4gbwX35_'
PYTHONPATH=src python scripts/download_wrds_crsp_sp500.py --wrds-username xcloudyun --output-dir data/raw
```
This script downloads:

- `constituents_month_end.csv`: month-end S&P 500 snapshots from CRSP
- `prices_daily.csv`: daily CRSP prices for all securities that were ever in the S&P 500 during the sample window


### 价格数据清洗、750/250 rolling split、标签生成
```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config config/config.yaml
```

## 输出

运行后默认会在 `output/` 下生成：

- `master_dataset.parquet`：clean master dataset
- `study_periods.parquet`：每个study period的日期边界
- `sample_count_summary.csv`：每个study period的样本统计
- `constituent_reconstruction_note.md`：成分股重建说明
- `train_samples_period_XX.parquet`：某期训练样本
- `trade_samples_period_XX.parquet`：某期交易样本

- month-end constituent snapshots are later shifted forward to represent membership in the subsequent month
- all-ever-constituents extraction avoids survivorship bias
- downstream standardization must be estimated from the training window only
