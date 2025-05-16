# Python 模块与包

模块和包是 Python 中组织代码和复用功能的重要机制，它们帮助我们将代码分解成更易于管理的部分。

## 模块

模块是一个包含 Python 定义和语句的文件，文件名就是模块名加上 `.py` 后缀。

### 导入模块

导入整个模块：

```python
import math

# 使用模块中的函数
radius = 5
area = math.pi * radius ** 2
print(f"圆的面积: {area}")
```

导入特定函数或变量：

```python
from math import pi, sqrt

# 直接使用导入的函数和变量
radius = 5
area = pi * radius ** 2
print(f"圆的面积: {area}")
print(f"5的平方根: {sqrt(5)}")
```

使用别名导入：

```python
import math as m
from math import pi as PI, sqrt as square_root

print(m.cos(PI))
print(square_root(16))
```

导入模块中的所有内容（不推荐）：

```python
from math import *  # 不推荐，可能导致命名冲突

print(pi)
print(sqrt(16))
```

### 常用内置模块

Python 标准库包含许多内置模块：

```python
# 数学运算
import math
print(math.cos(math.pi))

# 随机数
import random
print(random.randint(1, 10))

# 日期和时间
import datetime
print(datetime.datetime.now())

# 操作系统接口
import os
print(os.getcwd())  # 获取当前工作目录

# 文件路径操作
import pathlib
path = pathlib.Path(".")
print(list(path.glob("*.py")))  # 列出当前目录中的 Python 文件

# 数据压缩
import gzip
with gzip.open("file.txt.gz", "wt") as f:
    f.write("压缩文本")

# JSON 处理
import json
data = {"name": "张三", "age": 30}
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)
```

### 创建自己的模块

创建一个简单的模块（保存为 `mymath.py`）：

```python
# mymath.py
def square(x):
    """计算一个数的平方"""
    return x ** 2

def cube(x):
    """计算一个数的立方"""
    return x ** 3

PI = 3.14159

if __name__ == "__main__":
    # 当模块作为脚本运行时执行的代码
    print("模块被直接运行")
    print(f"5的平方: {square(5)}")
    print(f"5的立方: {cube(5)}")
```

使用自定义模块：

```python
import mymath

print(mymath.square(4))  # 16
print(mymath.cube(4))    # 64
print(mymath.PI)         # 3.14159
```

### 模块搜索路径

Python 在导入模块时会按照特定的顺序搜索模块：

1. 当前目录
2. `PYTHONPATH` 环境变量中列出的目录
3. 安装的库目录

可以查看和修改模块搜索路径：

```python
import sys

# 查看当前的模块搜索路径
print(sys.path)

# 添加自定义路径
sys.path.append("/path/to/your/modules")
```

## 包

包是一种组织 Python 模块的方式，简单来说就是包含 `__init__.py` 文件的目录。

### 创建包

创建一个简单的包结构：

```
mypackage/
│
├── __init__.py
├── module1.py
└── module2.py
```

`__init__.py` 文件可以为空，也可以包含初始化代码：

```python
# mypackage/__init__.py
print("初始化 mypackage")

# 可以在这里导入子模块
from . import module1
from . import module2

# 定义包级别的变量
__version__ = "0.1"
```

子模块内容：

```python
# mypackage/module1.py
def function1():
    return "This is function1 from module1"

# mypackage/module2.py
def function2():
    return "This is function2 from module2"
```

### 导入包

导入整个包：

```python
import mypackage

# 如果在 __init__.py 中导入了子模块，可以这样使用
print(mypackage.module1.function1())
print(mypackage.module2.function2())
print(mypackage.__version__)
```

导入包中的特定模块：

```python
from mypackage import module1

print(module1.function1())
```

直接导入特定函数：

```python
from mypackage.module1 import function1
from mypackage.module2 import function2

print(function1())
print(function2())
```

### 相对导入

在包内部，模块可以使用相对导入引用其他模块：

```python
# mypackage/module3.py
from . import module1  # 导入同级的 module1
from .. import anotherpackage  # 导入父级包中的 anotherpackage
from ..anotherpackage import somemodule  # 导入父级包中其他包的模块
```

### 子包

包可以嵌套，创建层次结构：

```
mypackage/
│
├── __init__.py
├── module1.py
│
└── subpackage/
    ├── __init__.py
    └── module2.py
```

导入子包：

```python
import mypackage.subpackage.module2
from mypackage.subpackage import module2
from mypackage.subpackage.module2 import some_function
```

## 命名空间包

Python 3.3 引入了命名空间包，它们不需要 `__init__.py` 文件，可以跨越多个目录：

```
project1/
└── mypackage/
    └── module1.py

project2/
└── mypackage/
    └── module2.py
```

如果这两个目录都在 Python 的模块搜索路径中，则 `mypackage` 将成为一个命名空间包，可以同时导入 `module1` 和 `module2`：

```python
import mypackage.module1
import mypackage.module2
```

## 包管理

### pip

`pip` 是 Python 的标准包管理工具，用于安装和管理第三方包：

```bash
# 安装包
pip install requests

# 指定版本安装
pip install requests==2.25.1

# 安装最低版本
pip install "requests>=2.0.0"

# 查看已安装的包
pip list

# 查看特定包的信息
pip show requests

# 卸载包
pip uninstall requests

# 从 requirements.txt 安装依赖
pip install -r requirements.txt
```

### 虚拟环境

虚拟环境是一个独立的 Python 环境，可以为不同项目创建隔离的依赖：

```bash
# 创建虚拟环境
python -m venv myenv

# 激活虚拟环境
# Windows
myenv\Scripts\activate
# Linux/macOS
source myenv/bin/activate

# 在虚拟环境中安装包
pip install requests

# 退出虚拟环境
deactivate
```

### requirements.txt

`requirements.txt` 文件用于记录项目依赖：

```
# requirements.txt
requests==2.25.1
numpy>=1.20.0
pandas
```

生成 requirements.txt：

```bash
pip freeze > requirements.txt
```

## 实际应用示例

### 组织一个小型项目

```
myproject/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── myproject/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── validators.py
│   │
│   └── models/
│       ├── __init__.py
│       └── user.py
│
└── tests/
    ├── __init__.py
    ├── test_helpers.py
    └── test_user.py
```

### 设置脚本 (setup.py)

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="myproject",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.20.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A small example package",
    keywords="example, package",
    url="http://example.com/myproject",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
```

安装自己的包（开发模式）：

```bash
pip install -e .
```

## 最佳实践

1. **有意义的模块和包名**：使用简短、描述性的名称，避免与标准库冲突。
2. **清晰的导入结构**：在模块顶部导入，按顺序排列（标准库、第三方库、自己的模块）。
3. **合理使用 `__init__.py`**：在 `__init__.py` 中只定义必要的内容。
4. **减少通配符导入**：避免使用 `from module import *`，以防命名冲突。
5. **利用 `__all__` 变量**：在模块中定义 `__all__` 变量，指定公共 API。
   ```python
   # module.py
   __all__ = ['public_function', 'public_class']
   
   def public_function():
       pass
   
   def _private_function():  # 下划线前缀表示私有
       pass
   
   class public_class:
       pass
   ```
6. **包的向后兼容性**：当重构包结构时，保持公共 API 兼容。
7. **使用相对导入**：在包内部使用相对导入提高可维护性。

## 下一步

现在您已经了解了 Python 的模块和包，接下来可以学习 [Python 异常处理](/basic/exceptions)。 