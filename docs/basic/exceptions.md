# Python 异常处理

异常是程序运行时发生的错误，Python 使用异常处理机制来应对这些错误情况，保证程序的健壮性。

## 异常基础

### 常见异常类型

Python 内置了许多异常类型，以下是一些常见的：

| 异常名称 | 描述 |
|---------|------|
| `SyntaxError` | 语法错误 |
| `NameError` | 使用未定义的变量或函数 |
| `TypeError` | 操作或函数应用于错误类型的对象 |
| `ValueError` | 操作或函数接收到类型正确但值不合适的参数 |
| `IndexError` | 序列中没有此索引 |
| `KeyError` | 字典中没有此键 |
| `ImportError` | 导入模块失败 |
| `ModuleNotFoundError` | 找不到模块 |
| `FileNotFoundError` | 找不到文件或目录 |
| `ZeroDivisionError` | 除数为零 |
| `AttributeError` | 对象没有此属性 |
| `RuntimeError` | 一般的运行时错误 |
| `OverflowError` | 数值运算超出最大限制 |
| `RecursionError` | 递归超出最大深度 |
| `PermissionError` | 操作没有足够的权限 |
| `TimeoutError` | 操作超时 |

### 异常的结构

Python 的异常都有一个基类 `BaseException`，大多数内置异常都是 `Exception` 类的子类：

```
BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- GeneratorExit
 +-- Exception
      +-- StopIteration
      +-- StopAsyncIteration
      +-- ArithmeticError
      |    +-- FloatingPointError
      |    +-- OverflowError
      |    +-- ZeroDivisionError
      +-- AssertionError
      +-- AttributeError
      +-- BufferError
      +-- EOFError
      +-- ImportError
      |    +-- ModuleNotFoundError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- MemoryError
      +-- NameError
      +-- OSError
      |    +-- BlockingIOError
      |    +-- ChildProcessError
      |    +-- ConnectionError
      |    +-- FileExistsError
      |    +-- FileNotFoundError
      |    +-- InterruptedError
      |    +-- IsADirectoryError
      |    +-- NotADirectoryError
      |    +-- PermissionError
      |    +-- ProcessLookupError
      |    +-- TimeoutError
      +-- ReferenceError
      +-- RuntimeError
      |    +-- NotImplementedError
      |    +-- RecursionError
      +-- SyntaxError
      |    +-- IndentationError
      |         +-- TabError
      +-- SystemError
      +-- TypeError
      +-- ValueError
      |    +-- UnicodeError
      |         +-- UnicodeDecodeError
      |         +-- UnicodeEncodeError
      |         +-- UnicodeTranslateError
      +-- Warning
           +-- DeprecationWarning
           +-- PendingDeprecationWarning
           +-- RuntimeWarning
           +-- SyntaxWarning
           +-- UserWarning
           +-- FutureWarning
           +-- ImportWarning
           +-- UnicodeWarning
           +-- BytesWarning
           +-- ResourceWarning
```

## try-except 语句

使用 `try-except` 语句捕获和处理异常：

```python
try:
    # 可能引发异常的代码
    x = int(input("请输入一个数字: "))
    result = 10 / x
    print(f"10 除以 {x} 等于 {result}")
except ZeroDivisionError:
    # 处理除零异常
    print("错误：不能除以零！")
except ValueError:
    # 处理值错误异常
    print("错误：请输入有效的数字！")
```

### 处理多个异常

可以用多种方式处理多个异常：

```python
# 方式1：为每种异常类型提供单独的处理程序
try:
    # 可能引发异常的代码
    pass
except ValueError:
    # 处理值错误
    pass
except TypeError:
    # 处理类型错误
    pass

# 方式2：一次处理多种异常类型
try:
    # 可能引发异常的代码
    pass
except (ValueError, TypeError, KeyError):
    # 处理多种异常
    pass

# 方式3：捕获所有异常
try:
    # 可能引发异常的代码
    pass
except Exception as e:
    # 处理任何 Exception 类型的异常
    print(f"发生错误: {e}")
```

::: warning 注意
应避免使用过于宽泛的异常捕获（如 `except:` 或 `except Exception:`），这可能会捕获意外的错误并隐藏真正的问题。尽量只捕获预期的异常类型。
:::

### 获取异常信息

使用 `as` 关键字获取异常对象：

```python
try:
    file = open("不存在的文件.txt", "r")
except FileNotFoundError as e:
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print(f"错误细节: {e.args}")
```

## else 和 finally 子句

`else` 子句在 `try` 块没有引发异常时执行，`finally` 子句无论是否发生异常都会执行：

```python
try:
    x = int(input("请输入一个正数: "))
    if x <= 0:
        raise ValueError("输入必须是正数")
except ValueError as e:
    print(f"错误: {e}")
else:
    # 没有异常时执行
    print(f"您输入的正数是: {x}")
finally:
    # 总是执行的代码
    print("程序执行完毕")
```

## 主动引发异常

使用 `raise` 语句主动引发异常：

```python
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("除数不能为零")
    return a / b

try:
    result = divide(10, 0)
except ZeroDivisionError as e:
    print(f"错误: {e}")
```

### 重新引发异常

在 `except` 块中使用不带参数的 `raise` 语句可以重新引发当前异常：

```python
try:
    # 尝试连接数据库
    db.connect()
except ConnectionError as e:
    # 记录错误
    log_error(e)
    # 重新引发异常，让上层处理
    raise
```

### 异常链

使用 `from` 关键字可以在引发新异常时保留原始异常的上下文：

```python
try:
    # 可能引发异常的代码
    int("abc")
except ValueError as e:
    # 引发新异常，但保留原始异常的上下文
    raise RuntimeError("处理数据时出错") from e
```

## 断言

`assert` 语句用于在调试阶段测试条件，如果条件为假，则引发 `AssertionError`：

```python
def calculate_average(numbers):
    assert len(numbers) > 0, "列表不能为空"
    return sum(numbers) / len(numbers)

try:
    avg = calculate_average([])
except AssertionError as e:
    print(f"断言错误: {e}")
```

::: warning 注意
在生产环境中，可以使用 `-O` 选项运行 Python 来禁用断言。因此，不要在断言中包含必须执行的代码。
:::

## 自定义异常

可以通过继承 `Exception` 类（或其任何子类）创建自定义异常：

```python
class InsufficientFundsError(Exception):
    """当账户资金不足时引发的异常"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        self.deficit = amount - balance
        super().__init__(f"余额不足。当前余额: {balance}，尝试提取: {amount}，不足: {self.deficit}")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return amount

try:
    account = BankAccount(100)
    account.withdraw(150)
except InsufficientFundsError as e:
    print(f"错误: {e}")
    print(f"当前余额: {e.balance}, 需要追加: {e.deficit}")
```

## 上下文管理器（with 语句）

`with` 语句提供了一种优雅的方式来处理需要设置和清理的操作，尤其是在处理文件时：

```python
# 不使用 with 语句
try:
    file = open("example.txt", "r")
    content = file.read()
    print(content)
finally:
    file.close()

# 使用 with 语句（推荐）
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
# 文件会自动关闭
```

### 创建自定义上下文管理器

通过实现 `__enter__` 和 `__exit__` 方法创建自定义上下文管理器：

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
        
    def __enter__(self):
        print(f"连接到数据库: {self.connection_string}")
        self.connection = {"connected": True}  # 模拟连接
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("关闭数据库连接")
        if self.connection:
            self.connection["connected"] = False  # 模拟断开连接
        
        # 如果返回 True，则抑制异常
        # 如果返回 False 或 None，则允许异常传播
        return False

try:
    with DatabaseConnection("mysql://localhost/mydb") as conn:
        print(f"连接状态: {conn['connected']}")
        # 模拟操作数据库
        # raise ValueError("模拟数据库错误")
except ValueError as e:
    print(f"捕获到异常: {e}")
```

也可以使用 `contextlib` 模块的 `@contextmanager` 装饰器创建上下文管理器：

```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    try:
        file = open(filename, mode)
        yield file
    finally:
        file.close()

with file_manager("example.txt", "r") as file:
    content = file.read()
    print(content)
```

## 最佳实践

1. **只捕获特定异常**：避免使用宽泛的 `except:` 语句，尽量捕获具体的异常类型。

2. **尽早捕获，晚些抛出**：在离异常可能发生的地方捕获它，但如果无法处理，考虑将其传递给能够处理的上层代码。

3. **异常信息应明确**：自定义异常时，提供清晰的错误信息，帮助理解问题。

4. **利用 finally 进行清理**：无论是否发生异常，在 `finally` 块中执行必要的清理操作。

5. **使用 with 语句管理资源**：对于需要显式清理的资源（如文件、网络连接等），使用 `with` 语句自动管理。

6. **避免在异常处理程序中引发相同的异常**：这会导致难以追踪的问题。如果需要重新引发，要么使用无参数的 `raise`，要么使用 `from` 语法提供上下文。

7. **不要忽略异常**：避免空的 `except` 块，至少记录异常信息：

   ```python
   try:
       # 某些操作
       pass
   except Exception as e:
       # 不好的做法
       pass  # 忽略异常
       
       # 更好的做法
       logging.error(f"发生错误: {e}")
   ```

8. **正确使用 assert**：断言应该用于验证程序的内部不变量，而不是验证用户输入或外部条件。

## 实际应用示例

### 带有重试机制的网络请求

```python
import requests
import time
from requests.exceptions import RequestException

def fetch_with_retry(url, max_retries=3, backoff_factor=0.5):
    """发送 HTTP 请求，出错时自动重试"""
    retries = 0
    while retries <= max_retries:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # 检查状态码
            return response.json()
        except RequestException as e:
            retries += 1
            if retries > max_retries:
                print(f"达到最大重试次数 ({max_retries})，放弃")
                raise  # 重新抛出最后一个异常
            
            wait_time = backoff_factor * (2 ** (retries - 1))
            print(f"请求失败: {e}. 将在 {wait_time:.2f} 秒后重试 (尝试 {retries}/{max_retries})")
            time.sleep(wait_time)

try:
    data = fetch_with_retry("https://api.example.com/data")
    print(f"获取到数据: {data}")
except RequestException as e:
    print(f"无法获取数据: {e}")
```

### 带验证的数据处理

```python
class DataValidationError(Exception):
    """数据验证错误"""
    pass

def process_user_data(data):
    try:
        # 验证必需的字段
        if "name" not in data:
            raise DataValidationError("缺少 'name' 字段")
        if "age" not in data:
            raise DataValidationError("缺少 'age' 字段")
        
        # 验证字段类型和值
        if not isinstance(data["name"], str):
            raise DataValidationError("'name' 必须是字符串")
        
        try:
            age = int(data["age"])
        except (ValueError, TypeError):
            raise DataValidationError("'age' 必须是有效的整数")
        
        if age < 0 or age > 120:
            raise DataValidationError("'age' 必须在 0 到 120 之间")
        
        # 处理数据
        return {
            "name": data["name"].strip(),
            "age": age
        }
    except DataValidationError:
        # 重新抛出验证错误
        raise
    except Exception as e:
        # 处理其他异常
        raise DataValidationError(f"无法处理数据: {e}") from e

# 使用示例
sample_data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": "25"},
    {"name": 123, "age": 40},
    {"name": "Charlie", "age": 150},
    {"name": "Dave"}
]

for i, user_data in enumerate(sample_data):
    try:
        processed_data = process_user_data(user_data)
        print(f"用户 {i+1}: {processed_data}")
    except DataValidationError as e:
        print(f"用户 {i+1} 数据无效: {e}")
```

## 下一步

恭喜！您已经学完了 Python 基础部分。接下来，您可以继续学习 [Python 面向对象编程](/intermediate/oop)，开始探索 Python 的中级主题。 