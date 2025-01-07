# LightGBM 模型訓練與性能評估

## Table of Contents

- [Introduction](#introduction)
- [Key Functions and Topics](#key-functions-and-topics)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Model Training and Feature Importance](#model-training-and-feature-importance)
  - [Performance Evaluation](#performance-evaluation)
  - [Threshold Adjustment and Confusion Matrix](#threshold-adjustment-and-confusion-matrix)
  - [Top Customers Selection](#top-customers-selection)
  - [Custom Sample Weight Adjustment](#custom-sample-weight-adjustment)
- [Highlights](#highlights)
- [How to Use](#how-to-use)
- [Summary](#summary)

---

## Introduction

此 Notebook 使用 LightGBM 進行模型訓練，並對預測結果進行詳細分析和優化。主要目的是通過預測客戶行為（例如基金購買概率）來支持業務決策，並進一步調整模型性能以提升預測準確性。

---

## Key Functions and Topics

### 1. Data Loading and Preprocessing

- 加載數據集並進行基本檢查（數據類型、唯一值）。
- 將類別特徵轉換為 `category` 類型以支持 LightGBM。
- 採用基於唯一值數量的自動類別特徵檢測。

### 2. Model Training and Feature Importance

- 使用 LightGBM 訓練分類模型，設置類別特徵。
- 提取並可視化模型的特徵重要性，幫助識別關鍵變量。

### 3. Performance Evaluation

- 計算並繪製 ROC 曲線，分析 AUC 分數。
- 使用交叉驗證評估模型穩定性。

### 4. Threshold Adjustment and Confusion Matrix

- 計算最佳閾值以平衡精確率和召回率。
- 使用混淆矩陣與分類報告進行深入的結果分析。

### 5. Top Customers Selection

- 根據預測概率選擇目標客戶群（例如，購買概率最高的 1000 人）。
- 預測目標客戶的平均購買率及購買人數。

### 6. Custom Sample Weight Adjustment

- 自動計算正負樣本比例，重新平衡模型權重。
- 重新訓練模型並比較性能。

---

## Highlights

- **靈活的數據處理流程**：支持自動類別檢測與權重調整。
- **詳細的性能分析**：包括 ROC 曲線、AUC 分數、混淆矩陣。
- **實際應用場景**：目標客戶選擇與商業預測。

---

## How to Use

1. 確保已安裝必要的依賴：
   ```bash
   pip install lightgbm matplotlib seaborn

### Data Loading and Preprocessing
# 數據加載與預處理

```python
import pandas as pd
import numpy as np

# 加載數據
merged_df = pd.read_csv(taishin\merged_df1.csv')

# 顯示每列數據類型和獨特值數量
print(merged_df.dtypes)
for col in merged_df.columns:
    unique_count = merged_df[col].nunique()
    print(f"{col} has {unique_count} unique values")

# 將 unique 值小於 50 的列設為類別變數
categorical_features = [col for col in merged_df.columns if merged_df[col].nunique() < 50]
for feature in categorical_features:
    merged_df[feature] = merged_df[feature].astype('category')

```
### Model Training and Feature Importance
# 模型訓練與特徵重要性

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# 定義特徵和目標
X = merged_df.drop('rs_prod_3', axis=1)
y = merged_df['rs_prod_3']

# 拆分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型並標記類別特徵
model = LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=categorical_features)

# 獲取特徵重要性
importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# 顯示前 30 個重要特徵
top_features = feature_importances.head(30)
print(top_features)

```
### Performance Evaluation
# 性能評估與預測
```python
from sklearn.metrics import roc_curve, auc

# 預測概率
y_scores = model.predict_proba(X_test)[:, 1]

# 計算 ROC 曲線和 AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 交叉驗證的 AUC 分數
from sklearn.model_selection import cross_val_score
cv_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print(f"5-fold CV AUC: {cv_auc.mean():.2f}")

```
### Threshold Adjustment and Confusion Matrix
# 閾值調整與混淆矩陣
```python
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

# 計算最佳閾值
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * precisions * recalls / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1_scores)]

# 使用最佳閾值進行預測
y_pred_adjusted = (y_scores >= best_threshold).astype(int)

# 混淆矩陣與分類報告
conf_mat = confusion_matrix(y_test, y_pred_adjusted)
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", classification_report(y_test, y_pred_adjusted))

```
### Top Customers Selection
# 頂尖客戶選擇與預期購買計算
```python
# 將預測概率與測試集結合
test_set_with_scores = X_test.copy()
test_set_with_scores['probability'] = y_scores

# 選擇概率最高的前 1000 位客戶
top_1000_customers = test_set_with_scores.sort_values('probability', ascending=False).head(1000)

# 預測購買概率與預期購買數量
predicted_buy_rate = top_1000_customers['probability'].mean()
expected_buyers = predicted_buy_rate * 1000
print(f"預期在選定的 1000 名客戶中，約有 {expected_buyers:.0f} 人購買基金。")

```
### Custom Sample Weight Adjustment
# 自定義正負樣本權重調整
```python
# 計算正負樣本比例並調整模型權重
negative_count = len(y_train[y_train == 0])
positive_count = len(y_train[y_train == 1])
model = LGBMClassifier(scale_pos_weight=negative_count / positive_count)
model.fit(X_train, y_train)

```

