# 压疮风险因子与预警模型：最简技术路线（含代码）

> 目标：在尽量少的步骤与依赖下，完成**时间序列数据集构建**、**独立风险因子识别**与**风险预警模型**（非预测模型）输出。

---

## 0. 路径与约定

```python
DATA_DIR = r"E:\迅雷下载\mimic-iv-3.0\mimic-iv-3.0"
OUT_DIR = r"C:\Users\huxia\Desktop\压疮项目风险因子寻找"
```

---

## 1. 生成时间序列数据集（最简原则）

**最简原则**：
1. 以 `stay_id` 为单位；
2. 以 ICU `intime` 对齐；
3. 观察窗口 24h，标签窗口未来 24h；
4. 排除已存在压疮（prevalent_before_pred=1）。

**最简字段**：
- `stay_id`, `day_index`, `window_start`, `window_end`
- `y_future_24h`
- 若干核心特征（生命体征/实验室/Braden）
- `dataset_split`

---

## 2. 清洗与特征类型统一（直接用你的脚本）

> 把路径改成统一路径，并输出到指定目录。

```python
import os
from your_cleaning_script import clean_for_dl

DATA_DIR = r"E:\迅雷下载\mimic-iv-3.0\mimic-iv-3.0"
OUT_DIR = r"C:\Users\huxia\Desktop\压疮项目风险因子寻找"

MODELREADY = os.path.join(DATA_DIR, "icu", "pu_timeseries_modelready_dropNaN_stage2plus_only_20260112_141652.csv")
MISSINGMASK = os.path.join(DATA_DIR, "icu", "pu_timeseries_missingmask_day24_stage2plus_only_20260112_141652.csv")

cleaned_csv = clean_for_dl(MODELREADY, MISSINGMASK, out_dir=OUT_DIR)
print(">>> cleaned_csv =", cleaned_csv)
```

---

## 3. 独立风险因子识别（最简 Logistic）

```python
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

DATA_DIR = r"E:\迅雷下载\mimic-iv-3.0\mimic-iv-3.0"
OUT_DIR = r"C:\Users\huxia\Desktop\压疮项目风险因子寻找"

CLEANED = os.path.join(OUT_DIR, "pu_timeseries_modelready_dropNaN_stage2plus_only_20260112_141652_CLEANED.csv")

df = pd.read_csv(CLEANED, low_memory=False)

target = "y_future_24h"

# 排除已存在压疮病例
if "prevalent_before_pred" in df.columns:
    df = df[df["prevalent_before_pred"] == 0].copy()

# 只取训练集
df = df[df["dataset_split"] == "train"].copy()

# 选择特征列
exclude = ["subject_id","hadm_id","stay_id","day_index","dataset_split",
           "window_start","window_end","intime","outtime",target]
features = [c for c in df.columns if c not in exclude and not c.startswith("mask__")]

X = df[features].fillna(0)
X = sm.add_constant(X)
Y = df[target].astype(int)

model = sm.Logit(Y, X)
res = model.fit(disp=False, maxiter=200)

# 输出 OR 和 95% CI
params = res.params
conf = res.conf_int()
OR = np.exp(params)
CI_low = np.exp(conf[0])
CI_high = np.exp(conf[1])

out = pd.DataFrame({
    "feature": params.index,
    "coef": params.values,
    "OR": OR.values,
    "CI_low": CI_low.values,
    "CI_high": CI_high.values,
    "p_value": res.pvalues.values,
}).sort_values("p_value")

os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(os.path.join(OUT_DIR, "risk_factor_logistic.csv"), index=False)
print("Done")
```

---

## 4. 风险预警模型（最简可解释评分）

```python
import numpy as np
import pandas as pd

# 假设已加载模型回归系数（risk_factor_logistic.csv）
coef_df = pd.read_csv(r"C:\Users\huxia\Desktop\压疮项目风险因子寻找\risk_factor_logistic.csv")

# 输入：某天某患者的特征向量（与训练特征一致）
# risk_score = sum(beta_i * x_i)
# risk_prob = 1/(1+exp(-risk_score))

# 简单分级
# 低: <0.1  中: [0.1,0.3)  高: >=0.3
```

---

## 5. 输出预警结果（最简格式）

| stay_id | day_index | risk_prob | risk_level | top_factors |
|---|---|---|---|---|
| 3001 | 2 | 0.42 | high | 低白蛋白、Braden 低、血红蛋白低 |

---

## 6. 最简执行顺序

1. 生成 `modelready.csv` 与缺失 mask。
2. 运行清洗脚本生成 `_CLEANED.csv`。
3. 运行 Logistic 识别独立风险因子。
4. 基于回归系数计算风险评分并分级预警。

