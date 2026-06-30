# Soft Sensor AutoResearch

[English](README.md)

面向软测量适用性验证的本地离线 AutoResearch 工具，使用 FDE TabPFN3 进行建模评估。

这个仓库同时是：

- 一个 Codex skill，需要显式通过 `$soft-sensor-autoresearch` 触发
- 一个可本地运行的 Python 包和命令行工具

它适用于用过程特征预测同一时刻或显式指定未来 horizon 目标值的软测量任务。默认 horizon 是 `h=0`，即使用截至时间 `t` 的特征预测时间 `t` 的目标值。

## 功能概览

Soft Sensor AutoResearch 会在一组低风险特征候选上执行有限、规则驱动的搜索，并用本地 FDE 模型评估每个候选。

流程：

1. 读取 `.csv` 或 `.parquet` 数据集。
2. 推断时间列和数值特征列。
3. 构造稳健的目标 holdout 时间窗口。
4. 对原始特征、上下文特征、趋势特征、窗口特征、覆盖度特征和可选频域特征进行排序。
5. 使用 TabPFN3 或 TPT_tab 评估候选。
6. 生成包含运行参数的交互式 Plotly `report.html` 和资源日志。

候选排序使用已完成 holdout 的直接平均 R-squared。报告中会保留每个 holdout 的 R-squared、RMSE、MAE、目标标准差和失败原因，方便诊断。

## 环境要求

- Python `>=3.11`
- 本地 FDE 或 benchmark checkout
- 本地 FDE 模型权重
- `pyproject.toml` 中声明的 Python 依赖

FDE 发现顺序：

1. `--fde-root`
2. `FDE_SOURCE_PATH`
3. 当前工作目录及其父目录
4. 同级 `FDE` 或 `benchmark` 目录

使用 TabPFN3 时，`FDE_TPT_WEIGHTS_DIR` 应指向包含 `tabpfn3/` 的父目录，例如：

```bash
export FDE_TPT_WEIGHTS_DIR=/path/to/FDE/packages/kernels/kernels/weights
```

如果请求的 FDE 模型或权重不可用，工具会快速失败。它不会静默回退到 XGBoost 作为预测模型；XGBoost 只用于特征筛选和 Top-N 排序。

## 安装

在本地 checkout 中安装：

```bash
python -m pip install -e .
```

安装测试依赖：

```bash
python -m pip install -e '.[dev]'
```

安装可选频域候选依赖：

```bash
python -m pip install -e '.[frequency]'
```

## 作为 Codex Skill 使用

将仓库克隆或同步到 Codex skill 目录，或创建符号链接：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)" "${CODEX_HOME:-$HOME/.codex}/skills/soft-sensor-autoresearch"
```

然后显式调用：

```text
Use $soft-sensor-autoresearch on /path/to/data.csv with target column <target_column>.
```

skill 需要两个输入：

- 数据集文件路径
- 目标列名

## CLI 用法

通过仓库内脚本运行：

```bash
python scripts/run_soft_sensor_autoresearch.py /path/to/data.csv '<target_column>'
```

或在 editable install 后运行：

```bash
soft-sensor-autoresearch /path/to/data.csv '<target_column>'
```

常用参数示例：

```bash
soft-sensor-autoresearch /path/to/data.csv '<target_column>' \
  --time-budget-minutes 15 \
  --num-train-samples 400 \
  --top-features-n 32 \
  --validation-fraction 0.30 \
  --forecast-horizons 0 \
  --model-type tabpfn3 \
  --tabpfn-device auto \
  --fde-root /path/to/FDE \
  --output-dir /path/to/output
```

常用选项：

- `--time-budget-minutes 0`：运行完整的有限候选列表。
- `--num-train-samples <n>`：控制 ICL 上下文样本数。
- `--top-features-n <n>`：控制进入模型的排序后特征数量。
- `--window-minutes <n>`：覆盖自动推断的采样窗口。
- `--forecast-horizons <steps>`：评估 `0`、`0:10` 或 `0,1,3,6,10` 等 horizon。
- `--model-type tabpfn3|tpt`：选择 FDE 模型路径。
- `--tabpfn-device cpu|auto|mps|cuda`：选择 TabPFN 运行设备。
- `--include-frequency-candidate`：启用 tsfresh/频域特征候选。
- `--no-resource-log`：关闭资源日志。
- `--open`：运行后打开生成的 HTML 报告。

## 输入数据

支持格式：

- `.csv`
- `.parquet`

列推断规则：

- 目标列必须显式提供。
- 第一个可解析为 datetime 的列会被视为时间列。
- 如果第一列无法解析为时间，会继续检查列名包含 `time`、`timestamp` 或 `date` 的列。
- 非目标的数值列会作为特征列。

如果无法推断时间列、找不到目标列或无法推断数值特征列，程序会失败并给出错误。

## 输出

每次运行会创建一个带时间戳的目录：

```text
autoresearch_YYYYMMDD_HHMMSS/
  report.html
  resource_usage.csv
```

`report.html` 包含候选排序、拟合图、每个 holdout 的指标和失败诊断。

报告开头包含 `Run Parameters` 区域。比较模型分数前应先查看这里；它记录目标、数据文件、模型类型、窗口大小、ICL 训练样本数、Top 特征数、验证集比例、forecast horizons、是否启用频域候选、FDE root 和模型运行参数。

`resource_usage.csv` 记录进程树 CPU/RSS 资源使用。MPS 运行可能还会包含 PyTorch MPS 内存事件。这些行是内存观测值，不是 Apple GPU 利用率百分比。

## Forecast Horizons

默认 forecast horizon 是 `0`。

- `h=0` 使用截至 `t` 的特征窗口预测 `t` 时刻目标。
- `h>0` 使用截至 `t-h` 的特征窗口预测 `t` 时刻目标。
- 一个 horizon step 等于一个数据采样步长；例如 10 分钟数据上的 `h=10` 表示提前 100 分钟。

未来 horizon 评估会在不同 horizon 之间保持 holdout 目标时间范围一致。比较不同 horizon 时，应同时查看 mean R-squared、worst holdout R-squared、RMSE 和 MAE，并用目标自相关和工艺因果关系验证任何看似可用的提前预测信号。

## Apple Silicon 与 MPS

Apple Silicon 运行时优先使用 MPS 上的 TabPFN3。

在沙箱环境中，即使 Mac 支持 MPS，PyTorch 也可能报告 `torch.backends.mps.is_available() == False`。遇到这种情况时，应先用提权的 MPS smoke test 验证，再判断机器是否缺少 MPS 支持。只有在明确希望 CPU 回退时，才使用 `--tabpfn-device cpu`。

## 负 R-Squared 排查

如果所有低风险候选都出现明显负 R-squared，应先把它视为数据或预处理诊断信号，而不是立即增加合成公式特征。

优先检查：

- 泄漏列和同步重复 tag
- 缺失值处理
- 自然采样粒度
- 原始数据与降采样聚合的差异
- holdout 目标分布漂移
- RMSE 相对目标标准差的大小

基础数据检查明确之后，再扩展特征族。不要在这个 skill 中使用 SISSO 风格的合成特征候选。

## 本地同步

查看远端：

```bash
git remote -v
```

期望 origin：

```text
https://github.com/axin1212/soft-sensor-autoresearch.git
```

拉取最新 GitHub 状态：

```bash
git fetch origin
git status --short --branch
```

如果直接在 `main` 工作，且工作区干净，可用：

```bash
git pull --ff-only origin main
```

如果在功能分支工作，merge 或 rebase 前先检查分叉：

```bash
git log --oneline --left-right --cherry-pick HEAD...origin/main
git diff --stat HEAD..origin/main
```

## 测试

```bash
python -m pytest
```
