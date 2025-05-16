# Python 上下文管理器

上下文管理器是 Python 中的一种特殊类型的对象，它允许您分配和释放资源精确地只在需要的时候。它们的主要用途是通过 `with` 语句提供资源获取和释放的便利机制，确保操作完成后资源被正确清理，即使发生异常也不例外。

## 上下文管理器基础

### with 语句和上下文管理器

`with` 语句的基本语法如下：

```python
with expression as variable:
    # 使用由表达式生成的变量代表的资源
    # 代码块结束时，资源会自动释放
```

其中，expression 必须返回一个上下文管理器对象，该对象实现了 `__enter__` 和 `__exit__` 方法。

### 上下文管理协议

上下文管理器需要实现两个方法：

1. `__enter__(self)`: 在 `with` 代码块开始前调用，返回值赋给 `as` 子句中的变量
2. `__exit__(self, exc_type, exc_val, exc_tb)`: 在 `with` 代码块结束时调用，无论是否有异常

当 `with` 语句执行时，发生以下步骤：
1. 执行 `with` 表达式获取上下文管理器对象
2. 调用上下文管理器的 `__enter__` 方法
3. 如果存在 `as` 子句，将 `__enter__` 的返回值赋给指定变量
4. 执行 `with` 语句体
5. 无论语句体是否抛出异常，都会调用 `__exit__` 方法

### 自定义上下文管理器

下面是一个简单的自定义上下文管理器示例：

```python
class MyContextManager:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        print(f"进入上下文：{self.name}")
        return self  # 返回值将赋给 as 子句中的变量
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"退出上下文：{self.name}")
        if exc_type:
            print(f"发生异常：{exc_type.__name__}: {exc_val}")
            # 返回 True 表示异常已处理，不会再向上传播
            # 返回 False 或 None 表示异常会继续传播
            return False  # 让异常继续传播
        return None

# 使用自定义上下文管理器
try:
    with MyContextManager("示例") as cm:
        print(f"在上下文中使用：{cm.name}")
        # 引发一个异常
        raise ValueError("示例异常")
except ValueError as e:
    print(f"捕获到异常：{e}")

# 输出：
# 进入上下文：示例
# 在上下文中使用：示例
# 退出上下文：示例
# 发生异常：ValueError: 示例异常
# 捕获到异常：示例异常
```

### 常见内置上下文管理器

Python 提供了许多内置的上下文管理器：

#### 文件操作

最常用的上下文管理器可能是文件对象，它在 `with` 语句结束时自动关闭文件：

```python
# 使用 with 自动关闭文件
with open('example.txt', 'w') as f:
    f.write('Hello, World!')
# 文件在这里自动关闭

# 不使用 with 需要手动关闭
f = open('example.txt', 'r')
try:
    content = f.read()
    print(content)
finally:
    f.close()  # 必须显式关闭文件
```

#### 锁

`threading` 模块的锁对象也支持上下文管理协议：

```python
import threading

lock = threading.Lock()

# 使用 with 自动释放锁
with lock:
    # 这里的代码是线程安全的
    print("锁已获取")
# 锁在这里自动释放

# 不使用 with 需要手动释放
lock.acquire()
try:
    # 这里的代码是线程安全的
    print("锁已获取")
finally:
    lock.release()  # 必须显式释放锁
```

## contextlib 模块

`contextlib` 模块提供了用于创建和使用上下文管理器的实用工具。

### contextmanager 装饰器

`contextmanager` 装饰器让您可以使用生成器函数创建上下文管理器，大大简化了实现：

```python
from contextlib import contextmanager

@contextmanager
def my_context(name):
    print(f"进入上下文：{name}")
    try:
        # yield 前的代码相当于 __enter__ 方法
        yield name  # yield 的值会赋给 as 子句中的变量
    except Exception as e:
        # 处理异常
        print(f"发生异常：{type(e).__name__}: {e}")
        raise  # 重新引发异常
    finally:
        # finally 块中的代码相当于 __exit__ 方法
        print(f"退出上下文：{name}")

# 使用生成器创建的上下文管理器
try:
    with my_context("生成器示例") as name:
        print(f"在上下文中使用：{name}")
        raise ValueError("生成器示例异常")
except ValueError as e:
    print(f"捕获到异常：{e}")

# 输出：
# 进入上下文：生成器示例
# 在上下文中使用：生成器示例
# 发生异常：ValueError: 生成器示例异常
# 退出上下文：生成器示例
# 捕获到异常：生成器示例异常
```

### suppress 上下文管理器

`suppress` 上下文管理器用于临时忽略特定类型的异常：

```python
from contextlib import suppress

# 忽略 FileNotFoundError 异常
with suppress(FileNotFoundError):
    with open('不存在的文件.txt', 'r') as f:
        content = f.read()
    print("这行代码不会执行，因为会抛出异常")

print("但程序会继续执行，因为异常被抑制了")

# 忽略多种异常
with suppress(FileNotFoundError, PermissionError, ValueError):
    # 执行可能引发这些异常的代码
    pass
```

### redirect_stdout 和 redirect_stderr

重定向标准输出和标准错误流：

```python
from contextlib import redirect_stdout, redirect_stderr
import sys

# 重定向标准输出到文件
with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        print("这将写入文件而不是控制台")

# 重定向标准错误到一个字符串 IO 对象
from io import StringIO
stderr_output = StringIO()
with redirect_stderr(stderr_output):
    print("这是正常输出", file=sys.stdout)
    print("这是错误输出", file=sys.stderr)

print(f"捕获的错误输出: {stderr_output.getvalue()}")
```

### closing 上下文管理器

`closing` 上下文管理器确保对象在 `with` 代码块结束时调用其 `close` 方法：

```python
from contextlib import closing
from urllib.request import urlopen

# 确保 urlopen 返回的对象在 with 块结束时被关闭
with closing(urlopen('https://www.python.org')) as page:
    for line in page:
        print(line.decode('utf-8').strip())
```

### ExitStack 管理多个上下文

`ExitStack` 允许您动态地创建任意数量的上下文管理器：

```python
from contextlib import ExitStack

def process_files(files):
    with ExitStack() as stack:
        # 动态创建文件对象并添加到栈中
        file_objects = [stack.enter_context(open(fname)) for fname in files]
        # 现在可以处理所有打开的文件
        for file_obj in file_objects:
            print(file_obj.read())
    # 所有文件将在这里自动关闭

try:
    process_files(['file1.txt', 'file2.txt', 'file3.txt'])
except FileNotFoundError:
    print("示例中的文件不存在，但这只是演示用途")
```

## 高级上下文管理器技巧

### 异步上下文管理器 (Python 3.7+)

从 Python 3.7 开始，可以使用 `async with` 语句和异步上下文管理器：

```python
import asyncio

class AsyncContextManager:
    async def __aenter__(self):
        print("进入异步上下文")
        await asyncio.sleep(1)  # 模拟异步操作
        return "异步资源"
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("退出异步上下文")
        await asyncio.sleep(0.5)  # 模拟异步清理
        return False  # 不抑制异常

async def main():
    async with AsyncContextManager() as resource:
        print(f"使用资源: {resource}")
        await asyncio.sleep(0.5)  # 模拟异步工作

# 运行异步代码
if __name__ == "__main__":
    asyncio.run(main())
```

使用 `contextlib.asynccontextmanager` 创建异步上下文管理器：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_context():
    print("进入异步上下文")
    try:
        await asyncio.sleep(1)  # 模拟异步设置
        yield "异步资源"
    finally:
        print("退出异步上下文")
        await asyncio.sleep(0.5)  # 模拟异步清理

async def main():
    async with async_context() as resource:
        print(f"使用资源: {resource}")
        await asyncio.sleep(0.5)  # 模拟异步工作

# 运行异步代码
if __name__ == "__main__":
    asyncio.run(main())
```

### 可重入上下文管理器

创建可以多次进入的上下文管理器：

```python
import threading

class ReentrantLock:
    """可重入锁的上下文管理器实现"""
    
    def __init__(self):
        self._lock = threading.RLock()
    
    def __enter__(self):
        self._lock.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False  # 不抑制异常

# 使用可重入上下文管理器
lock = ReentrantLock()

with lock:
    print("获取锁第一次")
    with lock:  # 可以再次进入同一个上下文
        print("获取锁第二次")
    print("释放内层锁")
print("释放外层锁")
```

### 参数化上下文管理器

创建可以接受参数的上下文管理器工厂：

```python
class Indenter:
    """一个用于缩进输出的上下文管理器"""
    
    def __init__(self, indent_level=0):
        self.indent_level = indent_level
    
    def __enter__(self):
        self.indent_level += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indent_level -= 1
        return False
    
    def print(self, text):
        """打印带缩进的文本"""
        print("    " * self.indent_level + text)

# 使用参数化上下文管理器
with Indenter() as indent:
    indent.print("第一级")
    with indent:
        indent.print("第二级")
        with indent:
            indent.print("第三级")
        indent.print("回到第二级")
    indent.print("回到第一级")
```

## 实用上下文管理器示例

### 临时目录管理器

创建在使用后自动删除的临时目录：

```python
import os
import shutil
import tempfile
from contextlib import contextmanager

@contextmanager
def temporary_directory():
    """创建一个临时目录，在上下文结束时自动删除"""
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")
    try:
        yield temp_dir
    finally:
        print(f"删除临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)

# 使用临时目录
with temporary_directory() as temp_dir:
    # 在临时目录中创建一个文件
    file_path = os.path.join(temp_dir, "temp_file.txt")
    with open(file_path, "w") as f:
        f.write("这是临时文件中的内容")
    
    # 读取文件
    with open(file_path, "r") as f:
        print(f"文件内容: {f.read()}")
    
    print(f"临时文件位置: {file_path}")
# 退出 with 块后，临时目录及其内容将被删除
```

### 计时上下文管理器

测量代码块执行时间的上下文管理器：

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """测量代码块执行时间的上下文管理器"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{name} 耗时: {elapsed:.6f} 秒")

# 使用计时上下文管理器
with timer("排序操作"):
    # 一些耗时操作
    sorted([5, 2, 8, 1, 9, 3, 7] * 10000)

with timer("睡眠操作"):
    time.sleep(0.5)
```

### 数据库连接管理器

管理数据库连接的上下文管理器：

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def database_connection(db_path):
    """创建和管理数据库连接的上下文管理器"""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

# 使用内存数据库
with database_connection(":memory:") as conn:
    # 创建游标
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    
    # 插入数据
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("张三", 30))
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("李四", 25))
    
    # 提交事务
    conn.commit()
    
    # 查询数据
    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        print(f"用户: ID={row[0]}, 姓名={row[1]}, 年龄={row[2]}")
# 退出 with 块后，数据库连接自动关闭
```

### 事务上下文管理器

管理数据库事务的上下文管理器：

```python
@contextmanager
def transaction(connection):
    """管理数据库事务的上下文管理器"""
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("事务已提交")
    except Exception as e:
        connection.rollback()
        print(f"事务已回滚: {e}")
        raise

# 使用事务上下文管理器
with database_connection(":memory:") as conn:
    # 创建表
    with transaction(conn) as cursor:
        cursor.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
        cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("笔记本电脑", 5999.99))
    
    # 成功的事务
    with transaction(conn) as cursor:
        cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("智能手机", 3999.99))
        cursor.execute("SELECT * FROM products")
        for row in cursor.fetchall():
            print(f"产品: ID={row[0]}, 名称={row[1]}, 价格={row[2]}")
    
    try:
        # 失败的事务
        with transaction(conn) as cursor:
            cursor.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("平板电脑", 2999.99))
            # 故意引发错误
            cursor.execute("INSERT INTO nonexistent_table VALUES (1, 'error')")
    except sqlite3.OperationalError:
        print("捕获到数据库操作错误")
    
    # 验证最后一个事务是否回滚
    with transaction(conn) as cursor:
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
        print(f"产品数量: {len(products)}")  # 应该是 2，而不是 3
```

### 改变工作目录的上下文管理器

临时改变工作目录的上下文管理器：

```python
import os
from contextlib import contextmanager

@contextmanager
def change_directory(path):
    """临时改变当前工作目录的上下文管理器"""
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        print(f"已更改目录到: {path}")
        yield
    finally:
        os.chdir(original_dir)
        print(f"已恢复目录到: {original_dir}")

# 使用目录更改上下文管理器
print(f"当前目录: {os.getcwd()}")

# 一个存在的目录
tmp_dir = tempfile.gettempdir()
with change_directory(tmp_dir):
    print(f"内部目录: {os.getcwd()}")
    # 在临时目录中执行操作
    with open("test_file.txt", "w") as f:
        f.write("测试内容")

print(f"外部目录: {os.getcwd()}")
```

### 环境变量管理器

临时修改环境变量的上下文管理器：

```python
import os
from contextlib import contextmanager

@contextmanager
def environment_variables(**kwargs):
    """临时设置环境变量的上下文管理器"""
    # 保存原始的环境变量
    original_values = {}
    for key, value in kwargs.items():
        if key in os.environ:
            original_values[key] = os.environ[key]
        else:
            original_values[key] = None
    
    # 设置新的环境变量
    for key, value in kwargs.items():
        os.environ[key] = value
    
    try:
        yield
    finally:
        # 恢复原始的环境变量
        for key in kwargs:
            if original_values[key] is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_values[key]

# 使用环境变量上下文管理器
print(f"DEBUG 环境变量: {os.environ.get('DEBUG', '未设置')}")

with environment_variables(DEBUG="1", APP_ENV="testing"):
    print(f"在上下文中 DEBUG 环境变量: {os.environ.get('DEBUG')}")
    print(f"在上下文中 APP_ENV 环境变量: {os.environ.get('APP_ENV')}")

print(f"退出上下文后 DEBUG 环境变量: {os.environ.get('DEBUG', '未设置')}")
print(f"退出上下文后 APP_ENV 环境变量: {os.environ.get('APP_ENV', '未设置')}")
```

## 最佳实践和设计模式

### 何时使用上下文管理器

上下文管理器特别适合以下场景：

1. 资源管理：获取和释放临时资源（文件、网络连接、数据库连接等）
2. 事务控制：数据库事务、原子操作等
3. 临时状态更改：改变工作目录、环境变量、配置设置等
4. 锁和并发控制：线程锁、进程锁等
5. 计时和性能测量：统计代码块执行时间
6. 错误处理和日志记录：捕获和处理特定代码块的错误

### 上下文管理器设计原则

设计良好的上下文管理器应遵循以下原则：

1. **单一职责原则**：一个上下文管理器应该只做一件事，而且做好
2. **清晰的资源生命周期**：明确资源的获取和释放时机
3. **异常安全**：确保在发生异常时资源也能正确释放
4. **可复用性**：设计通用的上下文管理器，以便在多处使用
5. **可组合性**：允许与其他上下文管理器组合使用

### 嵌套上下文管理器的注意事项

嵌套使用上下文管理器时，需要注意以下几点：

1. **释放顺序**：上下文管理器的 `__exit__` 方法的调用顺序与 `__enter__` 相反（后进先出）
2. **异常处理**：内层上下文管理器的异常可能会影响外层上下文管理器
3. **资源依赖**：确保内层上下文不依赖于已经释放的外层资源

### 上下文管理器与装饰器的结合

将上下文管理器与装饰器结合使用可以创建更强大的工具：

```python
from functools import wraps

def with_logging(func):
    """记录函数调用的装饰器，使用上下文管理器实现"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with timer(func.__name__):  # 使用之前定义的 timer 上下文管理器
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"函数 {func.__name__} 抛出异常: {e}")
                raise
    return wrapper

@with_logging
def process_data(data):
    """处理数据的函数"""
    # 模拟处理数据
    time.sleep(0.1 * len(data))
    result = sum(data)
    return result

# 调用被装饰的函数
result = process_data([1, 2, 3, 4, 5])
print(f"处理结果: {result}")
```

## 下一步

现在您已经了解了 Python 上下文管理器的各种应用场景，接下来可以继续学习 Python 高级特性，例如 [Python 并发编程](/advanced/concurrency)。通过上下文管理器和并发编程的结合，您可以构建更加复杂和高效的 Python 应用程序。 