# Python 装饰器

装饰器是 Python 中强大而优雅的特性，它允许您修改或增强函数和类的行为，而无需修改其源代码。本章将详细介绍装饰器的概念、工作原理以及实际应用场景。

## 装饰器基础

### 什么是装饰器

装饰器是一个函数，它接受另一个函数作为参数，并返回一个新的函数，通常在不修改原始函数代码的情况下扩展其功能。装饰器的语法使用 `@` 符号，放在目标函数的定义之前。

### 函数是一等公民

理解装饰器的关键是认识到 Python 中函数是"一等公民"，这意味着函数可以：
- 赋值给变量
- 作为参数传递给其他函数
- 作为函数的返回值
- 存储在数据结构中

```python
# 函数可以赋值给变量
def greet(name):
    return f"你好，{name}！"

say_hello = greet
print(say_hello("小明"))  # 输出: 你好，小明！

# 函数可以作为参数传递
def apply_function(func, value):
    return func(value)

print(apply_function(len, "Python"))  # 输出: 6
```

### 简单装饰器示例

下面是一个简单的装饰器示例，它在函数执行前后打印消息：

```python
def simple_decorator(func):
    def wrapper():
        print("函数即将执行...")
        func()
        print("函数已执行完毕！")
    return wrapper

# 使用装饰器语法
@simple_decorator
def say_hello():
    print("你好，世界！")

# 调用被装饰的函数
say_hello()

# 输出:
# 函数即将执行...
# 你好，世界！
# 函数已执行完毕！
```

上面的代码等同于：

```python
def simple_decorator(func):
    def wrapper():
        print("函数即将执行...")
        func()
        print("函数已执行完毕！")
    return wrapper

def say_hello():
    print("你好，世界！")

# 手动应用装饰器
say_hello = simple_decorator(say_hello)

# 调用被装饰的函数
say_hello()
```

### 带参数的装饰器

前面的示例只适用于没有参数的函数，下面是一个处理任意参数的装饰器：

```python
def general_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"函数 {func.__name__} 即将执行...")
        result = func(*args, **kwargs)
        print(f"函数 {func.__name__} 已执行完毕！")
        return result
    return wrapper

@general_decorator
def add(a, b):
    return a + b

@general_decorator
def greet(name):
    return f"你好，{name}！"

print(add(3, 5))  # 输出函数执行信息和结果 8
print(greet("小红"))  # 输出函数执行信息和结果 "你好，小红！"
```

### 保留被装饰函数的元数据

装饰器会修改被装饰函数的某些元数据，如函数名、文档字符串等。使用 `functools.wraps` 装饰器可以保留这些信息：

```python
import functools

def better_decorator(func):
    @functools.wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        """这是装饰器的文档字符串"""
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@better_decorator
def my_function():
    """这是我的函数的文档字符串"""
    pass

print(my_function.__name__)  # 输出: my_function（而不是 wrapper）
print(my_function.__doc__)   # 输出: 这是我的函数的文档字符串
```

## 高级装饰器模式

### 带参数的装饰器

有时候我们希望装饰器本身也能接受参数，这需要额外的嵌套：

```python
def repeat(n=1):
    """重复执行函数 n 次的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                result = func(*args, **kwargs)
                results.append(result)
            return results
        return wrapper
    return decorator

@repeat(3)
def say_hi(name):
    return f"你好，{name}！"

print(say_hi("小明"))  # 输出: ['你好，小明！', '你好，小明！', '你好，小明！']

@repeat()  # 使用默认参数
def say_hello():
    return "你好！"

print(say_hello())  # 输出: ['你好！']
```

### 类作为装饰器

除了函数，类也可以作为装饰器使用，只需实现 `__call__` 方法：

```python
class CountCalls:
    """计算函数调用次数的装饰器类"""
    
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 已被调用 {self.count} 次")
        return self.func(*args, **kwargs)

@CountCalls
def hello():
    print("你好！")

hello()  # 输出: hello 已被调用 1 次 你好！
hello()  # 输出: hello 已被调用 2 次 你好！
```

### 带参数的类装饰器

类装饰器也可以接受参数：

```python
class Prefix:
    """为函数返回值添加前缀的装饰器类"""
    
    def __init__(self, prefix):
        self.prefix = prefix
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, str):
                return f"{self.prefix}{result}"
            return result
        return wrapper

@Prefix("结果: ")
def get_message():
    return "操作成功！"

print(get_message())  # 输出: 结果: 操作成功！
```

### 多个装饰器的应用顺序

当多个装饰器应用于同一个函数时，装饰器的应用顺序是从下到上的（最接近函数定义的装饰器先应用）：

```python
def decorator1(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器1开始")
        result = func(*args, **kwargs)
        print("装饰器1结束")
        return result
    return wrapper

def decorator2(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("装饰器2开始")
        result = func(*args, **kwargs)
        print("装饰器2结束")
        return result
    return wrapper

@decorator1
@decorator2
def my_function():
    print("函数执行")

my_function()
# 输出顺序:
# 装饰器1开始
# 装饰器2开始
# 函数执行
# 装饰器2结束
# 装饰器1结束
```

## 实用装饰器示例

### 计时装饰器

测量函数执行时间的装饰器：

```python
import time
import functools

def timer(func):
    """计算函数执行时间的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行耗时: {end_time - start_time:.6f} 秒")
        return result
    return wrapper

@timer
def slow_function():
    """一个执行较慢的函数"""
    time.sleep(1)
    return "完成"

slow_function()  # 输出: slow_function 执行耗时: 1.000xxx 秒
```

### 缓存装饰器

缓存函数结果以避免重复计算的装饰器（Python 3.2+ 可以使用内置的 `functools.lru_cache`）：

```python
def memoize(func):
    """缓存函数结果的简单装饰器"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 创建可哈希的键
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"计算结果: {cache[key]}")
        else:
            print(f"使用缓存结果: {cache[key]}")
            
        return cache[key]
    
    return wrapper

@memoize
def fibonacci(n):
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))  # 第一次计算
print(fibonacci(10))  # 使用缓存
```

使用内置的 `functools.lru_cache`：

```python
@functools.lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))  # 非常快速，使用了缓存
```

### 重试装饰器

当函数失败时自动重试的装饰器：

```python
def retry(max_attempts=3, delay=1):
    """
    当函数抛出异常时自动重试的装饰器
    max_attempts: 最大尝试次数
    delay: 两次尝试之间的延迟（秒）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"尝试 {attempts}/{max_attempts} 失败: {e}")
                    print(f"将在 {delay} 秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    """一个不稳定的函数，有时会失败"""
    import random
    if random.random() < 0.7:  # 70% 的概率失败
        raise ValueError("随机错误")
    return "成功"

try:
    result = unreliable_function()
    print(f"结果: {result}")
except ValueError as e:
    print(f"最终失败: {e}")
```

### 身份验证装饰器

一个用于简单身份验证的装饰器：

```python
def requires_auth(func):
    """要求用户提供身份验证才能访问功能的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 这只是一个简单示例，实际应用中可能需要更复杂的认证逻辑
        auth = input("请输入密码: ")
        if auth != "secret":
            raise ValueError("身份验证失败")
        return func(*args, **kwargs)
    return wrapper

@requires_auth
def secret_function():
    return "这是机密信息"

try:
    print(secret_function())
except ValueError as e:
    print(e)
```

### 单例类装饰器

确保类只有一个实例的装饰器：

```python
def singleton(cls):
    """确保类只有一个实例的装饰器"""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self, host="localhost"):
        self.host = host
        print(f"连接到数据库: {host}")
    
    def query(self, sql):
        return f"在 {self.host} 上执行查询: {sql}"

# 创建实例
db1 = DatabaseConnection()  # 输出: 连接到数据库: localhost
db2 = DatabaseConnection("127.0.0.1")  # 不输出任何内容

# 验证是相同的实例
print(db1 is db2)  # 输出: True
print(db1.host)    # 输出: localhost
print(db2.host)    # 输出: localhost（因为是同一个实例，第二次创建被忽略）
```

## 装饰器的高级模式

### 类方法和静态方法装饰器

Python 内置的 `@classmethod` 和 `@staticmethod` 装饰器用于定义类方法和静态方法：

```python
class MyClass:
    value = 10
    
    def __init__(self, x):
        self.x = x
    
    # 实例方法（默认）
    def instance_method(self):
        return f"实例方法，self.x = {self.x}"
    
    # 类方法
    @classmethod
    def class_method(cls):
        return f"类方法，cls.value = {cls.value}"
    
    # 静态方法
    @staticmethod
    def static_method(y):
        return f"静态方法，y = {y}"

obj = MyClass(5)
print(obj.instance_method())  # 输出: 实例方法，self.x = 5
print(MyClass.class_method())  # 输出: 类方法，cls.value = 10
print(obj.class_method())     # 也可以通过实例调用类方法
print(MyClass.static_method(3))  # 输出: 静态方法，y = 3
print(obj.static_method(3))      # 也可以通过实例调用静态方法
```

### 属性装饰器

`@property` 装饰器可以让方法像属性一样被访问：

```python
class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def email(self):
        return f"{self.first_name.lower()}.{self.last_name.lower()}@example.com"
    
    @full_name.setter
    def full_name(self, name):
        first, last = name.split(' ')
        self.first_name = first
        self.last_name = last
    
    @full_name.deleter
    def full_name(self):
        print("删除姓名！")
        self.first_name = None
        self.last_name = None

person = Person("张", "三")
print(person.full_name)  # 输出: 张 三
print(person.email)     # 输出: zhang.san@example.com

# 使用 setter
person.full_name = "李 四"
print(person.first_name)  # 输出: 李
print(person.last_name)   # 输出: 四
print(person.full_name)   # 输出: 李 四
print(person.email)       # 输出: li.si@example.com

# 使用 deleter
del person.full_name      # 输出: 删除姓名！
print(person.first_name)  # 输出: None
```

### 描述符与装饰器结合

描述符是 Python 的高级特性，可以与装饰器结合使用来创建更复杂的属性访问模式：

```python
class ValidatedProperty:
    """一个属性描述符，强制值满足某个条件"""
    
    def __init__(self, validator, error_msg="值不合法"):
        self.validator = validator
        self.error_msg = error_msg
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"{self.name}: {self.error_msg}")
        setattr(obj, self.private_name, value)

class Person:
    # 使用描述符验证年龄是否为正整数
    age = ValidatedProperty(lambda x: isinstance(x, int) and x > 0, "年龄必须是正整数")
    
    # 使用描述符验证邮箱格式
    email = ValidatedProperty(
        lambda x: isinstance(x, str) and '@' in x, 
        "邮箱格式不正确"
    )
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

try:
    person = Person("张三", 30, "zhangsan@example.com")
    print(f"{person.name}, {person.age}, {person.email}")
    
    # 尝试设置无效值
    person.age = -5  # 引发 ValueError
except ValueError as e:
    print(f"错误: {e}")

try:
    person = Person("李四", 25, "invalid-email")  # 引发 ValueError
except ValueError as e:
    print(f"错误: {e}")
```

### 装饰器注册模式

使用装饰器注册功能是一种常见模式，特别是在插件系统中：

```python
# 简单的命令注册系统
class CommandRegistry:
    _commands = {}
    
    @classmethod
    def register(cls, name=None):
        """注册命令的装饰器"""
        def decorator(func):
            nonlocal name
            command_name = name or func.__name__
            cls._commands[command_name] = func
            return func
        return decorator
    
    @classmethod
    def execute(cls, command_name, *args, **kwargs):
        """执行已注册的命令"""
        if command_name not in cls._commands:
            raise ValueError(f"未知命令: {command_name}")
        return cls._commands[command_name](*args, **kwargs)
    
    @classmethod
    def list_commands(cls):
        """列出所有已注册的命令"""
        return list(cls._commands.keys())

# 使用装饰器注册命令
@CommandRegistry.register()
def hello(name):
    return f"你好，{name}！"

@CommandRegistry.register("add_numbers")
def add(a, b):
    return a + b

@CommandRegistry.register()
def help():
    return f"可用命令: {', '.join(CommandRegistry.list_commands())}"

# 执行命令
print(CommandRegistry.execute("hello", "小明"))  # 输出: 你好，小明！
print(CommandRegistry.execute("add_numbers", 3, 5))  # 输出: 8
print(CommandRegistry.execute("help"))  # 输出: 可用命令: hello, add_numbers, help
```

### 基于类的装饰器工厂

创建一个基于类的装饰器工厂，可以更灵活地配置装饰器：

```python
class DecoratorFactory:
    """创建各种装饰器的工厂类"""
    
    @staticmethod
    def timer(log_level="info"):
        """创建一个计时装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                if log_level == "debug":
                    print(f"[DEBUG] {func.__name__} 执行耗时: {duration:.6f} 秒")
                else:
                    print(f"{func.__name__} 执行耗时: {duration:.6f} 秒")
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def retry(max_attempts=3, exceptions=(Exception,)):
        """创建一个重试装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        print(f"尝试 {attempt}/{max_attempts} 失败: {e}")
                        if attempt < max_attempts:
                            time.sleep(0.5 * attempt)  # 递增延迟
                
                # 所有尝试都失败
                raise last_exception
            return wrapper
        return decorator

# 使用装饰器工厂
@DecoratorFactory.timer("debug")
def slow_func():
    time.sleep(0.5)
    return "完成"

@DecoratorFactory.retry(max_attempts=2, exceptions=(ValueError,))
def unstable_func():
    import random
    if random.random() < 0.7:
        raise ValueError("随机错误")
    return "成功"

print(slow_func())  # 输出: [DEBUG] slow_func 执行耗时: 0.500xxx 秒

try:
    print(unstable_func())
except ValueError as e:
    print(f"最终失败: {e}")
```

## 最佳实践与注意事项

### 装饰器使用建议

1. **使用 `functools.wraps`**：保留被装饰函数的元数据。
2. **装饰器应该是透明的**：被装饰的函数在使用上应该与未装饰时一致。
3. **具体而非通用**：每个装饰器应该专注于一个具体的功能。
4. **避免过度使用**：装饰器很强大，但过度使用会使代码难以理解。
5. **考虑性能影响**：装饰器会在每次函数调用时执行额外代码，可能影响性能。

### 常见陷阱与解决方案

1. **装饰带默认可变参数的函数**：

```python
def dangerous_decorator(func):
    """这个装饰器可能导致问题"""
    cache = {}  # 在装饰器定义时创建，所有被装饰函数共享
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 使用共享缓存可能导致意外行为
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper

# 更安全的方法是将状态关联到特定的函数实例:
def safer_decorator(func):
    """这个装饰器更安全"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 在函数对象上存储缓存
        if not hasattr(wrapper, "_cache"):
            wrapper._cache = {}
        
        key = str(args) + str(sorted(kwargs.items()))
        if key not in wrapper._cache:
            wrapper._cache[key] = func(*args, **kwargs)
        return wrapper._cache[key]
    
    return wrapper
```

2. **修改装饰器装饰的函数**：

```python
@decorator
def my_function():
    pass

# 如果你想用不同的装饰器替换，应该重新定义整个函数:
def decorator2(func):
    # ...
    return wrapper

@decorator2
def my_function():
    pass

# 而不是尝试：
my_function = decorator2(my_function)  # 这通常不会按预期工作
```

### 调试装饰器相关问题

使用装饰器可能会使调试变得复杂，因为错误发生在包装函数中。一些有用的调试技巧：

1. **检查装饰器链**：使用 `__wrapped__` 属性查看原始函数。

```python
import inspect

@decorator1
@decorator2
def function():
    pass

# 查看原始函数
original = function.__wrapped__.__wrapped__
```

2. **创建可关闭的装饰器**：

```python
def debuggable_decorator(func):
    """可以通过环境变量禁用的装饰器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import os
        # 检查环境变量是否禁用了装饰器
        if os.environ.get("DISABLE_DECORATORS") == "1":
            return func(*args, **kwargs)
        
        # 正常的装饰器逻辑
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper
```

3. **在装饰器中添加更多的调试信息**：

```python
def debug_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        print(f"调用 {func.__name__}({signature})")
        try:
            result = func(*args, **kwargs)
            print(f"{func.__name__} 返回 {result!r}")
            return result
        except Exception as e:
            print(f"{func.__name__} 抛出异常: {e}")
            raise
    
    return wrapper
```

## 在实际项目中的应用

### Web 框架中的装饰器

以 Flask 为例，路由注册和请求处理都大量使用了装饰器：

```python
from flask import Flask, request

app = Flask(__name__)

# 路由装饰器
@app.route('/hello/<name>')
def hello(name):
    return f'你好，{name}！'

# 请求方法装饰器
@app.route('/user', methods=['POST'])
def create_user():
    username = request.json.get('username')
    return f'创建用户: {username}'

# 自定义装饰器：要求身份验证
def requires_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization')
        if not auth or auth != 'Bearer secret-token':
            return {'error': '未授权'}, 401
        return f(*args, **kwargs)
    return decorated

@app.route('/admin')
@requires_auth
def admin_page():
    return {'message': '欢迎来到管理页面'}
```

### 测试中的装饰器

在测试框架中，装饰器可以用于设置和清理测试环境：

```python
import unittest
import tempfile
import os

# 测试用的装饰器
def with_temp_file(func):
    """为测试函数提供临时文件的装饰器"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 创建临时文件
        fd, path = tempfile.mkstemp()
        try:
            # 将临时文件路径作为额外参数传递给测试函数
            return func(self, path, *args, **kwargs)
        finally:
            # 清理临时文件
            os.close(fd)
            os.unlink(path)
    return wrapper

class TestFileProcessing(unittest.TestCase):
    
    @with_temp_file
    def test_write_and_read(self, temp_file):
        """测试文件写入和读取"""
        # 写入测试数据
        with open(temp_file, 'w') as f:
            f.write('测试数据')
        
        # 读取并验证
        with open(temp_file, 'r') as f:
            content = f.read()
        
        self.assertEqual(content, '测试数据')
```

### 面向切面编程（AOP）

装饰器是实现面向切面编程的重要工具，可以将横切关注点（如日志、事务、权限）从业务逻辑中分离出来：

```python
# 定义切面（横切关注点）装饰器
def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[日志] 开始执行 {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"[日志] {func.__name__} 执行成功")
            return result
        except Exception as e:
            print(f"[日志] {func.__name__} 执行失败: {e}")
            raise
    return wrapper

def transactional(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[事务] 开始事务")
        try:
            result = func(*args, **kwargs)
            print(f"[事务] 提交事务")
            return result
        except:
            print(f"[事务] 回滚事务")
            raise
    return wrapper

# 应用多个切面
@log_execution
@transactional
def update_user(user_id, name):
    print(f"更新用户 {user_id} 的姓名为 {name}")
    if user_id < 0:
        raise ValueError("无效的用户ID")
    return True

# 测试
try:
    update_user(1, "张三")
except ValueError as e:
    print(f"捕获到错误: {e}")

try:
    update_user(-1, "李四")
except ValueError as e:
    print(f"捕获到错误: {e}")
```

## 下一步

现在您已经了解了 Python 装饰器的各种应用，接下来可以探索 [Python 上下文管理器](/intermediate/context-managers)，学习如何使用 `with` 语句和上下文管理器来简化资源管理。 