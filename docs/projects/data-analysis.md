# Python 数据分析

Python 是数据分析领域最流行的编程语言之一，拥有丰富的库和工具生态系统。本章将介绍如何使用 Python 进行数据分析，包括数据处理、可视化和分析。

## 数据分析基础

### 常用库

在 Python 进行数据分析时，以下库是必不可少的：

- **NumPy**：提供高性能的多维数组对象和数学函数
- **pandas**：提供数据结构和数据分析工具
- **Matplotlib**：用于创建静态、交互式和动画可视化
- **Seaborn**：基于 Matplotlib 的统计数据可视化
- **SciPy**：用于科学和技术计算的库
- **scikit-learn**：机器学习库

### 环境准备

创建一个数据分析环境：

```bash
# 安装必要的库
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

推荐使用 Jupyter Notebook 进行数据分析，这是一个交互式的开发环境，非常适合探索性数据分析。

```bash
# 启动 Jupyter Notebook
jupyter notebook
```

## NumPy 基础

[NumPy](https://numpy.org/) 是 Python 科学计算的基础库，提供了高性能的多维数组对象和处理这些数组的工具。

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])

# 数组属性
print(f"维度: {a.ndim}")
print(f"形状: {b.shape}")
print(f"类型: {a.dtype}")

# 创建特殊数组
zeros = np.zeros((3, 3))  # 3x3 全零数组
ones = np.ones((2, 4))    # 2x4 全一数组
rand = np.random.random((2, 2))  # 2x2 随机数组

# 数组切片
arr = np.arange(10)  # [0, 1, 2, ..., 9]
print(arr[2:5])      # [2, 3, 4]

# 多维数组切片
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[0, 1])  # 2
print(arr_2d[:2, 1:])  # [[2, 3], [5, 6]]

# 数组运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)      # [5, 7, 9]
print(a * b)      # [4, 10, 18]
print(np.dot(a, b))  # 32 (点积)

# 统计函数
arr = np.array([1, 2, 3, 4])
print(f"和: {np.sum(arr)}")
print(f"平均值: {np.mean(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"最小值: {np.min(arr)}")
print(f"最大值: {np.max(arr)}")

# 数组变形
arr = np.arange(12)
arr_reshaped = arr.reshape(3, 4)  # 3行4列
arr_transposed = arr_reshaped.T   # 转置

# 广播 (Broadcasting)
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)  # 广播 b 到 a 的形状
```

## pandas 数据处理

[pandas](https://pandas.pydata.org/) 是基于 NumPy 的数据分析库，提供了高性能、易用的数据结构和数据分析工具。

### Series 和 DataFrame

pandas 的两个主要数据结构是 Series (一维数据) 和 DataFrame (二维数据)。

```python
import pandas as pd
import numpy as np

# 创建 Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# 创建 DataFrame
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

# 从字典创建 DataFrame
data = {'name': ['张三', '李四', '王五'],
        'age': [25, 30, 35],
        'city': ['北京', '上海', '广州']}
df2 = pd.DataFrame(data)
print(df2)
```

### 数据导入与导出

pandas 支持多种格式的数据导入导出：

```python
# CSV 文件
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Excel 文件
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', sheet_name='Data')

# JSON 文件
df = pd.read_json('data.json')
df.to_json('output.json', orient='records')

# SQL 数据库
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
df.to_sql('table_name', conn, if_exists='replace')
```

### 数据查看与处理

```python
# 查看数据
df.head()  # 显示前 5 行
df.tail(3)  # 显示后 3 行
df.info()  # 数据信息
df.describe()  # 统计摘要

# 选取数据
df['A']  # 选择一列
df[0:3]  # 选择行
df.loc['20230102':'20230104', ['A', 'B']]  # 按标签选择
df.iloc[1:3, 0:2]  # 按位置选择

# 条件筛选
df[df.A > 0]  # A 列大于 0 的行
df[(df.A > 0) & (df.B < 0)]  # A 列大于 0 且 B 列小于 0 的行

# 修改数据
df.loc[:, 'E'] = np.random.randn(len(df))  # 添加新列
df.loc['20230101', 'A'] = 0  # 修改值

# 缺失值处理
df.dropna()  # 删除包含缺失值的行
df.fillna(value=0)  # 将缺失值填充为 0
df.isna()  # 判断是否为缺失值

# 数据操作
df.sort_values(by='B')  # 按 B 列排序
df.groupby('A').sum()  # 按 A 列分组并计算总和
```

### 时间序列处理

pandas 对时间序列数据有很好的支持：

```python
# 创建时间序列
rng = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)

# 重采样
ts_month = ts.resample('M').mean()  # 按月重采样并计算平均值

# 时区转换
ts_utc = ts.tz_localize('UTC')
ts_cn = ts_utc.tz_convert('Asia/Shanghai')

# 时期转换
periods = pd.period_range('2023-01', '2023-12', freq='M')
ts_period = pd.Series(np.random.randn(len(periods)), index=periods)
```

### 数据合并与连接

```python
# 连接 DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5']})
result = pd.concat([df1, df2])

# SQL 风格的连接
left = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                     'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K3'],
                      'C': ['C0', 'C1', 'C2'],
                      'D': ['D0', 'D1', 'D2']})
result = pd.merge(left, right, on='key')  # 内连接
result = pd.merge(left, right, on='key', how='outer')  # 外连接
```

## 数据可视化

数据可视化是数据分析的重要组成部分，Python 提供了多种可视化库。

### Matplotlib

Matplotlib 是 Python 最流行的可视化库，提供了丰富的图表类型。

```python
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# 创建简单的折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'r-', label='sin(x)')
plt.plot(x, np.cos(x), 'b--', label='cos(x)')
plt.title('正弦和余弦函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('trig_functions.png')
plt.show()

# 子图
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(x, y, 'r')
plt.title('正弦函数')

plt.subplot(2, 2, 2)
plt.plot(x, np.cos(x), 'g')
plt.title('余弦函数')

plt.subplot(2, 2, 3)
plt.plot(x, np.tan(x), 'b')
plt.title('正切函数')

plt.subplot(2, 2, 4)
plt.plot(x, np.exp(x), 'k')
plt.title('指数函数')

plt.tight_layout()
plt.show()
```

### Seaborn

Seaborn 是基于 Matplotlib 的统计数据可视化库，提供了更美观的默认样式和高级统计图表。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置样式
sns.set_theme(style="whitegrid")

# 加载示例数据集
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")

# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)
plt.title("消费金额与小费的关系")
plt.show()

# 线形图
plt.figure(figsize=(10, 6))
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")
sns.lineplot(data=flights_pivot)
plt.title("每月乘客数量")
plt.show()

# 直方图和密度图
plt.figure(figsize=(10, 6))
sns.histplot(tips["total_bill"], kde=True)
plt.title("消费金额分布")
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("不同日期的消费金额分布")
plt.show()

# 热力图
plt.figure(figsize=(10, 8))
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("相关性热力图")
plt.show()

# 分面网格图
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.histplot, "total_bill")
plt.show()
```

### Plotly

Plotly 是一个用于创建交互式可视化的库，特别适合于 Web 应用程序和仪表板。

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 准备数据
df = px.data.gapminder().query("year == 2007")

# 散点图
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)
fig.update_layout(title="2007年各国生活期望值与人均GDP")
fig.show()

# 线图
df_time = px.data.gapminder()
fig = px.line(df_time, x="year", y="lifeExp", color="country",
              line_group="country", hover_name="country",
              line_shape="spline", render_mode="svg")
fig.update_layout(title="国家生活期望值随时间的变化")
fig.show()

# 地图
fig = px.choropleth(df, locations="iso_alpha", color="lifeExp",
                    hover_name="country", projection="natural earth")
fig.update_layout(title="全球生活期望值分布")
fig.show()

# 3D 散点图
df_3d = px.data.iris()
fig = px.scatter_3d(df_3d, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species')
fig.update_layout(title="鸢尾花数据集的 3D 可视化")
fig.show()
```

## 实际案例：COVID-19 数据分析

我们来使用真实的 COVID-19 数据进行分析：

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
# 数据来源: https://github.com/CSSEGISandData/COVID-19
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

# 数据处理
# 转换数据格式从宽格式到长格式
df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                    var_name='Date', value_name='Confirmed')
df_melted['Date'] = pd.to_datetime(df_melted['Date'])

# 按国家和日期分组计算确诊总数
df_country = df_melted.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

# 选取几个国家进行分析
countries = ['China', 'US', 'India', 'Brazil', 'Russia']
df_selected = df_country[df_country['Country/Region'].isin(countries)]

# 创建可视化
plt.figure(figsize=(12, 8))
for country in countries:
    country_data = df_selected[df_selected['Country/Region'] == country]
    plt.plot(country_data['Date'], country_data['Confirmed'], label=country)

plt.title('COVID-19 确诊病例累计数')
plt.xlabel('日期')
plt.ylabel('确诊病例数')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

# 每日新增病例
df_selected = df_selected.sort_values(['Country/Region', 'Date'])
df_selected['New_cases'] = df_selected.groupby('Country/Region')['Confirmed'].diff().fillna(0)

# 计算7天移动平均
df_selected['New_cases_MA7'] = df_selected.groupby('Country/Region')['New_cases'].rolling(7).mean().reset_index(0, drop=True)

# 创建新增病例图
plt.figure(figsize=(12, 8))
for country in countries:
    country_data = df_selected[df_selected['Country/Region'] == country]
    plt.plot(country_data['Date'], country_data['New_cases_MA7'], label=country)

plt.title('COVID-19 每日新增病例 (7天移动平均)')
plt.xlabel('日期')
plt.ylabel('新增病例数')
plt.legend()
plt.grid(True)
plt.show()

# 创建热力图
pivot_table = df_country.pivot_table(index='Country/Region', columns=pd.Grouper(key='Date', freq='M'), values='Confirmed')
pivot_table = pivot_table.fillna(0)
pivot_table = pivot_table.diff(axis=1).iloc[:, 1:]

# 选择前 20 个国家
top_countries = df_country.groupby('Country/Region')['Confirmed'].max().sort_values(ascending=False).head(20).index
pivot_top20 = pivot_table.loc[top_countries]

plt.figure(figsize=(15, 12))
sns.heatmap(pivot_top20, cmap='YlOrRd', linewidths=0.5, annot=False)
plt.title('COVID-19 每月新增病例热力图 (按国家)')
plt.xlabel('月份')
plt.ylabel('国家')
plt.show()
```

## 统计分析

使用 SciPy 和 statsmodels 库进行统计分析：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns

# 生成示例数据
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 2 * x + 3 + np.random.normal(0, 1, 1000)

# 基本统计量
print(f"均值: {np.mean(x)}")
print(f"中位数: {np.median(x)}")
print(f"标准差: {np.std(x)}")
print(f"方差: {np.var(x)}")
print(f"最小值: {np.min(x)}")
print(f"最大值: {np.max(x)}")
print(f"分位数: {np.percentile(x, [25, 50, 75])}")

# 假设检验
# 正态性检验
k2, p_value = stats.normaltest(x)
print(f"D'Agostino-Pearson 正态性检验 p 值: {p_value}")

# t 检验
t_stat, p_value = stats.ttest_ind(x, y)
print(f"独立样本 t 检验 p 值: {p_value}")

# 皮尔逊相关系数
corr, p_value = stats.pearsonr(x, y)
print(f"皮尔逊相关系数: {corr}, p 值: {p_value}")

# 线性回归
X = sm.add_constant(x)  # 添加常数项
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# 可视化回归结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.plot(np.sort(x), results.predict(sm.add_constant(np.sort(x))), 'r')
plt.title('线性回归')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Q-Q 图 (检查残差分布)
residuals = results.resid
fig, ax = plt.subplots(figsize=(10, 6))
sm.qqplot(residuals, line='45', ax=ax)
plt.title('Q-Q 图')
plt.grid(True)
plt.show()

# 残差分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('残差分布')
plt.xlabel('残差')
plt.ylabel('频率')
plt.grid(True)
plt.show()
```

## 机器学习与数据分析

使用 scikit-learn 进行机器学习：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import seaborn as sns

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建分类器
# 逻辑回归
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"逻辑回归准确率: {accuracy_log:.4f}")

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"随机森林准确率: {accuracy_rf:.4f}")

# 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('随机森林混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

# 分类报告
report = classification_report(y_test, y_pred_rf, target_names=iris.target_names)
print("分类报告:")
print(report)

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=90)
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()
```

## Python 3.13 数据分析新特性

Python 3.13 在数据分析方面引入了一些改进，包括性能优化和新功能：

```python
import numpy as np
import pandas as pd

# 数组函数改进
arr = np.array([1, 2, 3, 4, 5])
result = np.exp2(arr)  # 2 的幂函数
print(result)  # [2. 4. 8. 16. 32.]

# pandas 改进
# 字符串处理性能提升
s = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])
result = s.str.contains('a')
print(result)

# 改进的分组聚合
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
    'B': [1, 2, 3, 4, 5, 6],
    'C': [2.0, 5., 8., 1., 2., 9.]
})

# 新的分组聚合方法
result = df.groupby('A')['B'].agg(
    total=('sum'),
    mean=('mean'),
    min_val=('min'),
    max_val=('max')
)
print(result)

# 改进的缺失值处理
df_missing = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan],
    'B': [np.nan, 5, 6, np.nan],
    'C': [7, 8, np.nan, 10]
})

# 两列填充
df_missing['A'] = df_missing['A'].fillna(df_missing['C'])
print(df_missing)
```

## 最佳实践

### 数据处理技巧

1. **数据清洗**：处理缺失值、异常值和重复数据
2. **特征工程**：创建新特征、转换现有特征
3. **数据标准化**：标准化/归一化数据
4. **降维**：如 PCA、t-SNE 等
5. **数据管道**：创建数据处理管道以自动化处理

```python
# 创建数据处理管道示例
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# 创建管道
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 用均值填充缺失值
    ('scaler', StandardScaler()),                 # 标准化
    ('pca', PCA(n_components=2))                  # 降维到 2 维
])

# 在数据上应用管道
X_transformed = pipeline.fit_transform(X)
```

### 性能优化

1. **向量化操作**：使用 NumPy/pandas 的向量化操作而不是循环
2. **分批处理大数据**：对于大数据集，使用分批处理
3. **使用适当的数据类型**：如 category 类型、适当的数值类型
4. **并行处理**：使用多进程或 Dask 进行并行计算

```python
# 向量化示例
import numpy as np
import time

# 准备数据
n = 1000000
a = np.random.rand(n)
b = np.random.rand(n)

# 循环方法
start = time.time()
result = np.zeros(n)
for i in range(n):
    result[i] = a[i] * b[i]
loop_time = time.time() - start

# 向量化方法
start = time.time()
result_vec = a * b
vec_time = time.time() - start

print(f"循环时间: {loop_time:.6f}秒")
print(f"向量化时间: {vec_time:.6f}秒")
print(f"加速比: {loop_time / vec_time:.1f}倍")
```

### 报告与可视化

1. **选择适当的图表类型**：根据数据类型和问题选择合适的可视化
2. **保持简洁明了**：避免过度装饰和无关信息
3. **使用合适的颜色**：考虑色盲友好的配色方案
4. **交互式可视化**：对于复杂数据，考虑使用交互式工具如 Plotly
5. **创建报告**：使用 Jupyter Notebook 或 R Markdown 创建综合报告

## 进一步学习

若要继续提升数据分析技能，可以研究：

1. **高级统计方法**：时间序列分析、贝叶斯统计等
2. **机器学习**：深入学习不同的 ML 算法和技术
3. **大数据工具**：Spark、Dask 等
4. **专业领域应用**：金融分析、生物信息学等
5. **数据伦理**：隐私、偏见和伦理考量

## 下一步

现在您已经了解了 Python 数据分析的基础知识，接下来可以尝试探索 [机器学习入门](/projects/machine-learning) 或 [自动化脚本](/projects/automation) 等其他 Python 应用领域。 