# 机器学习入门

Python 是机器学习的首选语言，拥有丰富的库和框架。本章将介绍机器学习基础概念、常用算法和实践项目，帮助您开始机器学习之旅。

## 机器学习基础

### 什么是机器学习？

机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进。基本思想是从数据中学习模式，而不是显式编程。

### 机器学习类型

机器学习主要分为三类：

1. **监督学习**：使用标记数据训练模型
   - 分类问题：预测离散值（如垃圾邮件检测）
   - 回归问题：预测连续值（如房价预测）

2. **无监督学习**：使用未标记数据，寻找数据中的模式
   - 聚类：将相似数据分组（如客户细分）
   - 降维：减少数据特征数量

3. **强化学习**：通过与环境互动和反馈学习
   - 代理通过尝试和错误学习最佳行动

### 机器学习工作流程

一个标准的机器学习项目通常包括以下步骤：

1. **问题定义**：明确目标和评估指标
2. **数据收集**：获取相关数据
3. **数据预处理**：清洗、转换和准备数据
4. **特征工程**：选择和创建特征
5. **模型选择与训练**：选择算法并训练模型
6. **模型评估**：评估模型性能
7. **模型调优**：优化模型超参数
8. **部署**：将模型应用于实际问题

## 机器学习环境设置

### 安装必要的库

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

更高级的机器学习项目可能需要：

```bash
pip install tensorflow keras pytorch xgboost lightgbm
```

### 虚拟环境（推荐）

创建隔离的 Python 环境：

```bash
# 创建虚拟环境
python -m venv ml_env

# 激活环境
# Windows
ml_env\Scripts\activate
# macOS/Linux
source ml_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用 scikit-learn 进行机器学习

[scikit-learn](https://scikit-learn.org/) 是 Python 中最流行的机器学习库，提供了简单而高效的工具进行数据分析和模型构建。

### 加载和探索数据

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# 加载内置数据集
iris = datasets.load_iris()
X = iris.data  # 特征
y = iris.target  # 目标变量

# 探索数据
print(f"数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"目标类别: {iris.target_names}")

# 将数据转换为 DataFrame 以便更好地可视化
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]
print(df.head())

# 数据统计摘要
print(df.describe())

# 可视化特征分布
plt.figure(figsize=(12, 8))
pd.plotting.scatter_matrix(df.iloc[:, :4], figsize=(12, 8),
                           c=y, marker='o', hist_kwds={'bins': 20},
                           s=60, alpha=0.8)
plt.suptitle('鸢尾花数据集特征散点矩阵')
plt.show()
```

### 数据预处理

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
```

### 分类模型

#### K最近邻分类

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 创建 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train_scaled, y_train)

# 预测
y_pred = knn.predict(X_test_scaled)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

#### 支持向量机

```python
from sklearn.svm import SVC

# 创建 SVM 分类器
svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)

# 训练模型
svm.fit(X_train_scaled, y_train)

# 预测
y_pred_svm = svm.predict(X_test_scaled)

# 评估
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM 准确率: {accuracy_svm:.4f}")
print("SVM 分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))
```

#### 随机森林

```python
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)  # 随机森林可以处理未标准化的数据

# 预测
y_pred_rf = rf.predict(X_test)

# 评估
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"随机森林准确率: {accuracy_rf:.4f}")
print("随机森林分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("特征重要性")
plt.bar(range(X.shape[1]), importances[indices],
        align="center")
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```

### 回归模型

#### 线性回归

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载波士顿房价数据集
# 注意：此数据集已从 scikit-learn 中移除，这里仅作示例
try:
    boston = load_boston()
    X_boston = boston.data
    y_boston = boston.target
except:
    # 如果无法加载，创建虚拟数据
    X_boston = np.random.rand(506, 13)
    y_boston = 10 + 2 * X_boston[:, 0] + np.random.randn(506) * 3

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X_train_scaled, y_train)

# 预测
y_pred_lr = lr.predict(X_test_scaled)

# 评估
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print(f"均方误差: {mse:.4f}")
print(f"R² 分数: {r2:.4f}")

# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('线性回归: 实际值 vs 预测值')
plt.show()
```

#### 决策树回归

```python
from sklearn.tree import DecisionTreeRegressor

# 创建决策树回归器
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)

# 训练模型
dt_reg.fit(X_train, y_train)  # 决策树可以处理未标准化的数据

# 预测
y_pred_dt = dt_reg.predict(X_test)

# 评估
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"决策树均方误差: {mse_dt:.4f}")
print(f"决策树 R² 分数: {r2_dt:.4f}")
```

### 模型评估与超参数调优

#### 交叉验证

```python
from sklearn.model_selection import cross_val_score

# 使用交叉验证评估模型
scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
print(f"交叉验证准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")
```

#### 网格搜索

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# 创建网格搜索
grid_search = GridSearchCV(
    SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 训练
grid_search.fit(X_train_scaled, y_train)

# 最佳参数和最佳分数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"最佳模型测试集准确率: {accuracy_best:.4f}")
```

#### 学习曲线

```python
from sklearn.model_selection import learning_curve

# 计算学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42),
    X_train_scaled, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

# 计算均值和标准差
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集分数")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证分数")
plt.xlabel("训练样本数")
plt.ylabel("准确率")
plt.title("学习曲线 - SVM")
plt.legend(loc="best")
plt.show()
```

### 特征工程

#### 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择 K 个最佳特征
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X, y)

# 查看每个特征的得分
scores = selector.scores_
print("特征得分:")
for i, score in enumerate(scores):
    print(f"{iris.feature_names[i]}: {score:.4f}")

# 可视化选择的两个特征
plt.figure(figsize=(10, 6))
for target, target_name in enumerate(iris.target_names):
    plt.scatter(X_new[y == target, 0], X_new[y == target, 1],
                label=target_name)
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('选择的两个最佳特征')
plt.legend()
plt.show()
```

#### 主成分分析 (PCA)

```python
from sklearn.decomposition import PCA

# 创建 PCA 对象
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# 解释方差比
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

# 可视化 PCA 结果
plt.figure(figsize=(10, 6))
for target, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y_train == target, 0], X_pca[y_train == target, 1],
                label=target_name)
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.title('PCA 降维结果')
plt.legend()
plt.show()
```

## 实际项目：手写数字识别

我们将使用 MNIST 数据集构建一个手写数字识别模型：

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 探索数据
print(f"数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"类别: {np.unique(y)}")

# 可视化一些数字
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r)
    plt.title(f"标签: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建和训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 预测
y_pred = rf.predict(X_test_scaled)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 可视化一些预测
plt.figure(figsize=(15, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    img = X_test[i].reshape(8, 8)
    plt.imshow(img, cmap=plt.cm.gray_r)
    pred = rf.predict([X_test_scaled[i]])[0]
    plt.title(f"真实: {y_test[i]}, 预测: {pred}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## 深度学习入门

### 什么是深度学习？

深度学习是机器学习的一个子集，使用深度神经网络自动学习数据的层次化表示。它特别擅长处理图像、文本、音频等非结构化数据。

### 安装 TensorFlow

```bash
pip install tensorflow
```

### 使用 TensorFlow 和 Keras 创建神经网络

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

# 调整数据形状
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 将类别标签转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 创建 CNN 模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 输出模型摘要
model.summary()

# 训练模型（为节省时间，我们只训练 1 个 epoch）
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=1,
                    validation_data=(x_test, y_test))

# 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print(f"测试损失: {score[0]:.4f}")
print(f"测试准确率: {score[1]:.4f}")

# 进行预测
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 可视化一些预测结果
plt.figure(figsize=(15, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"真实: {y_true_classes[i]}, 预测: {y_pred_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## 模型保存与加载

保存和加载模型对于部署和重用很重要：

### scikit-learn 模型

```python
import joblib

# 保存模型
joblib.dump(rf, 'random_forest_model.joblib')

# 加载模型
loaded_model = joblib.load('random_forest_model.joblib')

# 使用加载的模型进行预测
loaded_predictions = loaded_model.predict(X_test_scaled)
print(f"加载的模型准确率: {accuracy_score(y_test, loaded_predictions):.4f}")
```

### TensorFlow/Keras 模型

```python
# 保存整个模型
model.save('full_model.h5')

# 仅保存权重
model.save_weights('model_weights.h5')

# 加载整个模型
loaded_model = tf.keras.models.load_model('full_model.h5')

# 加载权重
# 首先需要创建相同结构的模型，然后加载权重
new_model = Sequential([
    # 与原模型相同的架构
])
new_model.load_weights('model_weights.h5')
```

## Python 3.13 机器学习相关新特性

Python 3.13 带来了一些改进，可以提升机器学习工作流程：

```python
# 改进的数据处理性能
import numpy as np
import pandas as pd

# 更快的数组操作
large_array = np.random.rand(1000000)
result = np.exp(large_array)  # 性能改进

# 改进的 pandas 分组操作
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar'] * 1000,
    'B': np.random.randn(4000)
})

# 更快的分组聚合
result = df.groupby('A')['B'].agg(['mean', 'sum', 'min', 'max'])
```

## 最佳实践

### 避免常见陷阱

1. **数据泄漏**：确保测试数据不会影响训练过程
2. **过拟合**：使用交叉验证、正则化和早停
3. **特征缩放不一致**：确保测试数据使用训练数据的缩放参数
4. **类别不平衡**：使用过采样、欠采样或加权损失函数

### 机器学习项目步骤

1. **明确问题**：确定目标和评估指标
2. **数据收集与探索**：获取和理解数据
3. **数据预处理**：清洗、转换和准备数据
4. **特征工程**：创建、选择和转换特征
5. **基线模型**：建立简单的基线模型
6. **模型选择与训练**：尝试不同的算法
7. **超参数调优**：优化模型性能
8. **模型评估**：在测试集上评估最终模型
9. **模型解释**：理解模型的决策过程
10. **部署与监控**：将模型投入生产环境并监控性能

### 资源管理

对于大型数据集和复杂模型：

```python
# 内存优化
# 使用生成器而不是一次加载所有数据
def data_generator(X, y, batch_size=32):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], y[batch_indices]

# GPU 内存管理 (TensorFlow)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制 GPU 内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

## 进一步学习

若要深入学习，可以探索：

1. **高级模型**：XGBoost、LightGBM、神经网络等
2. **自然语言处理**：文本分类、情感分析、命名实体识别
3. **计算机视觉**：图像分类、对象检测、图像分割
4. **推荐系统**：协同过滤、矩阵分解、深度推荐模型
5. **强化学习**：Q-Learning、策略梯度、深度强化学习

## 下一步

现在您已经了解了机器学习的基础知识，接下来可以尝试探索 [自动化脚本](/projects/automation) 或回顾 [数据分析](/projects/data-analysis) 以提升整体数据科学技能。 