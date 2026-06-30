# Soft Sensor AutoResearch

本地离线软测量适用性验证工具。给一份过程数据和目标列，它会自动搜索低风险特征，用本地 FDE TabPFN3/TPT_tab 评估这个目标是否值得建软测量模型，并生成 HTML 报告。

它可以作为 Codex skill 使用，也可以直接作为 Python CLI 使用。

## 一句话流程

```text
CSV/Parquet + 目标列
  -> 自动识别时间列和数值特征
  -> 构造多个 holdout
  -> 比较原始/趋势/窗口/覆盖度/可选频域特征
  -> 用 FDE TabPFN3 或 TPT_tab 打分
  -> 输出 report.html
```

适合快速回答：

- 这个目标有没有软测量建模价值？
- 哪类特征最有效？
- 不同 holdout 上效果是否稳定？
- 当前点 `h=0` 或指定未来 horizon 是否可预测？

不适合直接当在线部署回测，也不会在 FDE 模型不可用时静默退回 XGBoost。

## 快速开始

数据要求：

- 文件格式：`.csv` 或 `.parquet`
- 第一列最好是时间列；也支持列名包含 `time`、`timestamp`、`date`
- 目标列显式传入
- 其他数值列作为候选特征

运行：

```bash
python scripts/run_soft_sensor_autoresearch.py /path/to/data.csv '<target_column>'
```

常用参数：

```bash
python scripts/run_soft_sensor_autoresearch.py /path/to/data.csv '<target_column>' \
  --time-budget-minutes 15 \
  --num-train-samples 400 \
  --top-features-n 32 \
  --forecast-horizons 0 \
  --model-type tabpfn3 \
  --tabpfn-device auto \
  --fde-root /path/to/FDE \
  --output-dir /path/to/output
```

安装后也可以运行：

```bash
soft-sensor-autoresearch /path/to/data.csv '<target_column>'
```

## 安装和环境

安装：

```bash
python -m pip install -e .
```

测试依赖：

```bash
python -m pip install -e '.[dev]'
```

可选频域特征：

```bash
python -m pip install -e '.[frequency]'
```

FDE 查找顺序：

1. `--fde-root`
2. `FDE_SOURCE_PATH`
3. 当前目录及父目录
4. 同级 `FDE` 或 `benchmark` 目录

TabPFN3 权重：

```bash
export FDE_TPT_WEIGHTS_DIR=/path/to/FDE/packages/kernels/kernels/weights
```

`FDE_TPT_WEIGHTS_DIR` 应指向包含 `tabpfn3/` 的父目录。FDE、模型或权重不可用时，程序会直接失败并说明原因。

## 输出怎么看

每次运行生成：

```text
autoresearch_YYYYMMDD_HHMMSS/
  report.html
  resource_usage.csv
```

先看 `report.html`：

- `Run Parameters`：确认目标列、模型、horizon、窗口、Top 特征数
- 候选排序：按平均 R-squared 排序
- 每个 holdout：看 R-squared、RMSE、MAE 和拟合图
- 失败候选：看失败原因

`resource_usage.csv` 记录 CPU/RSS。MPS 行是 PyTorch MPS 内存事件，不是 GPU 利用率。

## Horizon 怎么理解

默认：

```bash
--forecast-horizons 0
```

- `h=0`：用截至 `t` 的特征预测 `t`
- `h>0`：用截至 `t-h` 的特征预测 `t`

示例：

```bash
--forecast-horizons 0
--forecast-horizons 0:10
--forecast-horizons 0,1,3,6,10
```

一个 horizon step 等于一个采样步长。10 分钟数据的 `h=10` 表示提前 100 分钟。

## Codex Skill

安装到 skill 目录：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)" "${CODEX_HOME:-$HOME/.codex}/skills/soft-sensor-autoresearch"
```

显式调用：

```text
Use $soft-sensor-autoresearch on /path/to/data.csv with target column <target_column>.
```

## 常见坑

所有低风险候选都是明显负 R-squared 时，先查数据：

- 泄漏列或同步重复 tag
- 缺失值处理
- 采样粒度
- 降采样/聚合是否破坏目标
- holdout 目标分布漂移
- RMSE 相对目标标准差是否其实不大

不要马上堆复杂合成特征。这个 skill 不使用 SISSO 风格候选。

Apple Silicon 上优先用 MPS。如果沙箱里 PyTorch 报 MPS 不可用，先用提权 smoke test 验证；只有明确想用 CPU 时才传 `--tabpfn-device cpu`。

## 同步和测试

```bash
git fetch origin
git status --short --branch
python -m pytest
```
