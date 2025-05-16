# Python 元编程

元编程是一种强大的编程技术，它允许程序在运行时检查、生成或修改自身的代码。Python 的动态特性使其成为实现元编程的理想语言。本章将介绍 Python 元编程的核心概念和技术。

## 元编程基础概念

元编程可以简单理解为"编写能够生成或操作其他代码的代码"。在 Python 中，主要有以下几种元编程方式：

1. **内省**：检查对象的结构和特性
2. **动态属性访问和修改**：在运行时访问和修改对象的属性
3. **装饰器**：修改函数和类的行为
4. **元类**：控制类的创建过程
5. **动态代码生成和执行**：在运行时生成和执行代码

## Python 内省机制

内省是元编程的基础，它允许程序检查自身的结构和属性。Python 提供了丰富的内省工具：

### 基本内省函数

```python
# type() - 获取对象的类型
print(type(42))  # <class 'int'>
print(type("hello"))  # <class 'str'>

# isinstance() - 检查对象是否是特定类的实例
print(isinstance(42, int))  # True
print(isinstance("hello", list))  # False

# dir() - 列出对象的所有属性和方法
print(dir("hello"))  # 列出字符串对象的所有属性和方法

# id() - 获取对象的唯一标识
a = [1, 2, 3]
b = a
print(id(a) == id(b))  # True，a 和 b 引用同一个对象

# vars() - 返回对象的 __dict__ 属性
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("张三", 30)
print(vars(p))  # {'name': '张三', 'age': 30}
```

### inspect 模块

`inspect` 模块提供了更强大的内省功能：

```python
import inspect

# 检查函数的参数
def greet(name, greeting="你好"):
    """向某人问候"""
    return f"{greeting}, {name}!"

print(inspect.signature(greet))  # (name, greeting='你好')

# 获取函数的文档字符串
print(inspect.getdoc(greet))  # 向某人问候

# 检查函数的源代码
print(inspect.getsource(greet))

# 获取对象的类层次结构
print(inspect.getmro(str))  # (<class 'str'>, <class 'object'>)

# 检查调用栈
def func_a():
    func_b()
    
def func_b():
    func_c()
    
def func_c():
    for frame_info in inspect.stack():
        print(f"{frame_info.function} 在 {frame_info.filename} 的第 {frame_info.lineno} 行")

# func_a()  # 调用以查看栈信息
```

## 动态属性访问与修改

Python 允许在运行时动态访问和修改对象的属性：

### getattr, setattr, delattr, hasattr

```python
class DynamicObject:
    def __init__(self):
        self.x = 10
        self.y = 20

obj = DynamicObject()

# getattr - 获取属性值
print(getattr(obj, "x"))  # 10
print(getattr(obj, "z", "不存在"))  # 不存在（默认值）

# setattr - 设置属性值
setattr(obj, "z", 30)
print(obj.z)  # 30

# hasattr - 检查是否有属性
print(hasattr(obj, "x"))  # True
print(hasattr(obj, "w"))  # False

# delattr - 删除属性
delattr(obj, "y")
print(hasattr(obj, "y"))  # False
```

### 特殊方法 __getattr__, __setattr__, __delattr__

通过实现这些特殊方法，可以自定义属性访问行为：

```python
class SmartObject:
    def __init__(self):
        self.data = {}
    
    def __getattr__(self, name):
        """当属性不存在时调用"""
        print(f"获取不存在的属性: {name}")
        return self.data.get(name, None)
    
    def __setattr__(self, name, value):
        """在设置属性时调用"""
        if name == "data":
            # 防止递归调用 __setattr__
            super().__setattr__(name, value)
        else:
            print(f"设置属性: {name} = {value}")
            self.data[name] = value
    
    def __delattr__(self, name):
        """在删除属性时调用"""
        if name in self.data:
            print(f"删除属性: {name}")
            del self.data[name]
        else:
            raise AttributeError(f"属性 {name} 不存在")

# 使用自定义属性行为
smart = SmartObject()
smart.name = "智能对象"  # 设置属性: name = 智能对象
print(smart.name)  # 智能对象
print(smart.age)   # 获取不存在的属性: age None
delattr(smart, "name")  # 删除属性: name
```

### 属性描述符

描述符是通过实现 `__get__`, `__set__` 和 `__delete__` 方法的对象，用于自定义属性的访问行为：

```python
class Validator:
    """一个简单的属性验证描述符"""
    
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None  # 属性名称
    
    def __set_name__(self, owner, name):
        """设置描述符的名称"""
        self.name = name
    
    def __get__(self, instance, owner):
        """获取属性值"""
        if instance is None:
            return self
        return instance.__dict__.get(self.name, None)
    
    def __set__(self, instance, value):
        """设置属性值，带验证"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} 必须是数字")
        
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} 必须大于等于 {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} 必须小于等于 {self.max_value}")
        
        instance.__dict__[self.name] = value

class Person:
    age = Validator(min_value=0, max_value=150)
    height = Validator(min_value=0)
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

# 使用描述符
try:
    p = Person("张三", 30, 175)
    print(f"{p.name}, {p.age}岁, {p.height}cm")
    
    p.age = 200  # 引发 ValueError
except ValueError as e:
    print(f"验证错误: {e}")
```

## 装饰器进阶

装饰器是 Python 最常用的元编程工具之一。在前面的章节中已经介绍了基本的装饰器用法，这里我们将深入一些更高级的技术：

### 装饰器工厂

创建可配置的装饰器工厂：

```python
import functools
import time

def rate_limit(calls_per_second=1):
    """限制函数调用频率的装饰器工厂"""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_called = [0.0]  # 使用列表存储，以便在闭包中修改
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    
    return decorator

# 使用装饰器工厂
@rate_limit(calls_per_second=2)  # 每秒最多调用2次
def process_item(item):
    print(f"处理 {item}，时间：{time.strftime('%H:%M:%S')}")

# 快速连续调用
for i in range(5):
    process_item(i)  # 将会看到每次调用间隔至少0.5秒
```

### 装饰器描述符混合使用

将装饰器与描述符结合使用：

```python
class cached_property:
    """一个缓存属性值的描述符装饰器"""
    
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.name = func.__name__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        # 计算并缓存值
        value = self.func(instance)
        # 将值存储在实例的字典中，下次直接从那里获取
        instance.__dict__[self.name] = value
        return value

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def processed_data(self):
        """处理数据（耗时操作）"""
        print("处理数据...")
        time.sleep(1)  # 模拟耗时操作
        return [x * 2 for x in self.data]

# 使用缓存属性
processor = DataProcessor([1, 2, 3, 4, 5])
print("第一次访问:")
print(processor.processed_data)  # 会执行处理
print("第二次访问:")
print(processor.processed_data)  # 直接使用缓存值
```

### 类装饰器

装饰整个类的装饰器：

```python
def add_method(cls):
    """为类添加方法的装饰器"""
    def greet(self, name):
        return f"{self.__class__.__name__} 向 {name} 问好!"
    
    # 添加方法到类
    cls.greet = greet
    return cls

@add_method
class Person:
    def __init__(self, name):
        self.name = name

# 使用添加的方法
p = Person("张三")
print(p.greet("李四"))  # Person 向 李四 问好!
```

## 元类编程

元类是 Python 最强大的元编程特性之一，它控制类的创建过程。

### 理解元类

在 Python 中，类也是对象，它们是由元类创建的：

```python
# 类是元类的实例
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>

# 可以使用 type() 动态创建类
DynamicClass = type(
    'DynamicClass',     # 类名
    (object,),          # 继承的基类
    {'x': 10, 'y': 20}  # 类属性和方法
)

obj = DynamicClass()
print(obj.x, obj.y)  # 10 20
```

### 自定义元类

通过定义元类，可以控制类的创建过程：

```python
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        print(f"创建类: {name}")
        print(f"基类: {bases}")
        print(f"命名空间: {namespace}")
        
        # 修改类的命名空间
        namespace['meta_created'] = True
        namespace['meta_name'] = mcs.__name__
        
        # 调用原始的 __new__ 方法创建类
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace):
        print(f"初始化类: {name}")
        super().__init__(name, bases, namespace)

# 使用自定义元类
class MyClass(metaclass=Meta):
    x = 10
    
    def say_hello(self):
        return "Hello"

# 检查元类添加的属性
obj = MyClass()
print(f"meta_created: {obj.meta_created}")
print(f"meta_name: {obj.meta_name}")
```

### 实用元类示例：单例模式

使用元类实现单例模式：

```python
class Singleton(type):
    """单例元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    def __init__(self, name="default"):
        self.name = name
        print(f"初始化数据库连接: {name}")

# 测试单例行为
db1 = Database("主数据库")
db2 = Database("副数据库")  # 不会创建新实例
print(f"db1 是 db2: {db1 is db2}")
print(f"db1.name: {db1.name}")
print(f"db2.name: {db2.name}")  # 仍是"主数据库"
```

### 元类实现注册机制

使用元类实现类的自动注册：

```python
class RegisterMeta(type):
    """自动注册类的元类"""
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # 不注册抽象基类
        if namespace.get('abstract', False):
            return cls
        
        # 注册类
        mcs.registry[name] = cls
        return cls

class Shape(metaclass=RegisterMeta):
    abstract = True  # 标记为抽象类
    
    def area(self):
        raise NotImplementedError

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2

# 查看注册的类
print("注册的类:")
for name, cls in RegisterMeta.registry.items():
    print(f"- {name}")

# 通过注册表创建实例
shape_name = "Circle"
if shape_name in RegisterMeta.registry:
    shape_cls = RegisterMeta.registry[shape_name]
    shape = shape_cls(5)
    print(f"{shape_name} 的面积: {shape.area():.2f}")
```

## 动态代码生成与执行

Python 提供了多种方式在运行时生成和执行代码：

### eval 和 exec

`eval` 用于计算表达式的值，`exec` 用于执行语句：

```python
# eval - 计算表达式
x = 10
y = 20
result = eval("x + y")
print(f"eval('x + y') = {result}")  # 30

# exec - 执行语句
code = """
for i in range(3):
    print(f"i = {i}")
"""
exec(code)  # 输出 i = 0, i = 1, i = 2

# 传递命名空间
local_vars = {"x": 100, "y": 200}
result = eval("x * y", {}, local_vars)
print(f"使用本地命名空间计算 x * y = {result}")  # 20000
```

### 动态创建函数：types.FunctionType

使用 `types.FunctionType` 可以动态创建函数：

```python
import types

# 创建代码对象
code = compile("result = a + b", "<string>", "exec")

# 动态创建函数
def create_adder(a):
    # 创建一个闭包函数
    def adder(b):
        # 执行代码，在当前作用域中
        loc = {"a": a, "b": b}
        exec(code, globals(), loc)
        return loc["result"]
    
    return adder

# 创建和使用动态函数
add_5 = create_adder(5)
print(f"add_5(10) = {add_5(10)}")  # 15

# 使用 types.FunctionType 创建函数
def make_function(name, param_names, body):
    param_str = ", ".join(param_names)
    source = f"def {name}({param_str}):\n"
    for line in body.split("\n"):
        source += f"    {line}\n"
    
    # 编译源代码
    code_obj = compile(source, f"<{name}>", "exec")
    
    # 创建函数的命名空间
    namespace = {}
    exec(code_obj, globals(), namespace)
    
    # 返回创建的函数
    return namespace[name]

# 使用函数生成器
multiply = make_function(
    "multiply",
    ["x", "y"],
    "return x * y"
)

print(f"multiply(6, 7) = {multiply(6, 7)}")  # 42
```

### 动态类创建的高级技术

使用 `type` 可以动态创建类，结合其他元编程技术可以实现更复杂的功能：

```python
def create_model(name, fields):
    """动态创建一个数据模型类"""
    
    # 创建 __init__ 函数
    def __init__(self, **kwargs):
        for field_name in fields:
            setattr(self, field_name, kwargs.get(field_name))
    
    # 创建 __str__ 函数
    def __str__(self):
        fields_str = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
        return f"{name}({fields_str})"
    
    # 创建 as_dict 方法
    def as_dict(self):
        return {field: getattr(self, field) for field in fields}
    
    # 创建类的命名空间
    namespace = {
        "__init__": __init__,
        "__str__": __str__,
        "as_dict": as_dict,
        "fields": fields
    }
    
    # 使用 type 创建类
    return type(name, (object,), namespace)

# 使用动态类创建工厂
User = create_model("User", ["id", "name", "email"])
Product = create_model("Product", ["id", "name", "price", "category"])

# 创建实例
user = User(id=1, name="张三", email="zhangsan@example.com")
product = Product(id=101, name="笔记本电脑", price=5999, category="电子产品")

print(user)
print(product)
print(product.as_dict())
```

## 元编程中的常见模式

### 注册与工厂模式

使用元编程实现注册和工厂模式：

```python
class Registry:
    """通用的注册表"""
    
    def __init__(self):
        self._registry = {}
    
    def register(self, name=None):
        """注册装饰器"""
        def decorator(cls):
            key = name or cls.__name__
            self._registry[key] = cls
            return cls
        return decorator
    
    def create(self, name, *args, **kwargs):
        """工厂方法，创建已注册类的实例"""
        if name not in self._registry:
            raise ValueError(f"未知类型: {name}")
        return self._registry[name](*args, **kwargs)
    
    def get_registry(self):
        """获取注册表"""
        return dict(self._registry)

# 创建数据库驱动注册表
db_registry = Registry()

@db_registry.register()
class SQLiteDriver:
    def __init__(self, database):
        self.database = database
    
    def connect(self):
        print(f"连接到 SQLite 数据库: {self.database}")

@db_registry.register("mysql")
class MySQLDriver:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
    
    def connect(self):
        print(f"连接到 MySQL: {self.user}@{self.host}/{self.database}")

# 使用工厂创建实例
sqlite_db = db_registry.create("SQLiteDriver", "app.db")
sqlite_db.connect()

mysql_db = db_registry.create("mysql", 
                             host="localhost",
                             user="root",
                             password="password",
                             database="myapp")
mysql_db.connect()

# 列出所有注册的驱动
print("已注册的数据库驱动:")
for name in db_registry.get_registry():
    print(f"- {name}")
```

### 混入(Mixin)和特征(Traits)

使用元类实现混入和特征模式：

```python
class TraitMeta(type):
    """特征元类"""
    
    def __new__(mcs, name, bases, namespace):
        # 合并所有特征的属性
        traits = namespace.get('traits', [])
        for trait in traits:
            for key, value in trait.__dict__.items():
                # 跳过魔术方法
                if not key.startswith('__'):
                    namespace[key] = value
        
        return super().__new__(mcs, name, bases, namespace)

# 定义特征
class JSONSerializable:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class Printable:
    def print_info(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

# 使用特征
class User(metaclass=TraitMeta):
    traits = [JSONSerializable, Printable]
    
    def __init__(self, name, email):
        self.name = name
        self.email = email

# 测试特征功能
user = User("李四", "lisi@example.com")
print(user.to_json())
user.print_info()
```

### 自动属性

使用元类自动创建属性：

```python
class AutoPropertyMeta(type):
    """自动为字段创建属性的元类"""
    
    def __new__(mcs, name, bases, namespace):
        # 获取字段列表
        fields = namespace.get('fields', [])
        
        # 为每个字段创建属性
        for field in fields:
            # 创建私有存储名称
            private_name = f"_{field}"
            
            # 创建 getter
            def make_getter(field_name, private_field):
                def getter(self):
                    return getattr(self, private_field, None)
                return getter
            
            # 创建 setter
            def make_setter(field_name, private_field):
                def setter(self, value):
                    setattr(self, private_field, value)
                return setter
            
            # 创建属性并添加到命名空间
            namespace[field] = property(
                make_getter(field, private_name),
                make_setter(field, private_name)
            )
        
        return super().__new__(mcs, name, bases, namespace)

# 使用自动属性元类
class Person(metaclass=AutoPropertyMeta):
    fields = ['name', 'age', 'email']
    
    def __init__(self, **kwargs):
        for field in self.fields:
            setattr(self, field, kwargs.get(field))

# 测试自动属性
p = Person(name="王五", age=35)
print(f"姓名: {p.name}")  # 使用自动生成的 getter
p.email = "wangwu@example.com"  # 使用自动生成的 setter
print(f"电子邮件: {p.email}")
```

## 最佳实践与注意事项

### 元编程的利弊

**优点**：
- 减少重复代码，提高抽象级别
- 创建更灵活、更通用的程序
- 实现领域特定语言(DSL)

**缺点**：
- 增加代码复杂性，可能难以理解和调试
- 可能降低程序运行效率
- 滥用可能导致"黑魔法"般的难以维护的代码

### 使用建议

1. **适度使用**：只在真正需要时使用元编程
2. **良好文档**：为复杂的元编程特性提供清晰的文档
3. **明确目标**：明确元编程的目标，避免过度工程
4. **保持简单**：尽量使用最简单的方法实现目标
5. **单元测试**：彻底测试元编程代码

### 安全注意事项

在使用 `eval`、`exec` 等动态执行代码的函数时需要特别小心：

```python
# 不安全的示例 - 永远不要在生产环境中这样做!
user_input = "print('hello')"  # 假设这是用户输入
eval(user_input)  # 危险！

# 如果必须使用 eval/exec，请限制命名空间
safe_globals = {'__builtins__': {}}  # 非常受限的全局命名空间
safe_locals = {}

try:
    # 安全一些，但仍不建议用于处理不可信输入
    result = eval(user_input, safe_globals, safe_locals)
except Exception as e:
    print(f"执行出错: {e}")
```

## 实际应用示例

### ORM(对象关系映射)

一个简单的 ORM 实现：

```python
class Field:
    """数据库字段描述符"""
    
    def __init__(self, field_type, required=False, default=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._data.get(self.name, self.default)
    
    def __set__(self, instance, value):
        if value is not None and not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} 必须是 {self.field_type.__name__} 类型")
        instance._data[self.name] = value

class ModelMeta(type):
    """模型元类"""
    
    def __new__(mcs, name, bases, namespace):
        if name == "Model":
            return super().__new__(mcs, name, bases, namespace)
        
        # 收集字段
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                fields[key] = value
        
        # 添加字段到类的属性
        namespace['_fields'] = fields
        namespace['_table'] = namespace.get('_table', name.lower())
        
        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ModelMeta):
    """ORM 基类"""
    
    def __init__(self, **kwargs):
        self._data = {}
        
        # 设置默认值
        for name, field in self._fields.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif field.default is not None:
                setattr(self, name, field.default)
            elif field.required:
                raise ValueError(f"字段 {name} 是必需的")
    
    def __str__(self):
        fields_str = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._fields)
        return f"{self.__class__.__name__}({fields_str})"
    
    def save(self):
        """模拟保存到数据库"""
        print(f"保存到表 {self._table}:")
        for name in self._fields:
            print(f"  {name}: {getattr(self, name)!r}")
        print("保存成功")

# 使用 ORM
class User(Model):
    _table = "users"  # 自定义表名
    
    id = Field(int, required=True)
    name = Field(str, required=True)
    email = Field(str, required=True)
    age = Field(int, default=0)
    active = Field(bool, default=True)

# 创建和保存用户
try:
    user = User(id=1, name="赵六", email="zhaoliu@example.com", age=40)
    print(user)
    user.save()
    
    # 验证类型
    user.age = "不是数字"  # 应该引发 TypeError
except (ValueError, TypeError) as e:
    print(f"错误: {e}")
```

### 模板引擎

一个简单的模板引擎实现：

```python
class Template:
    """简单的字符串模板引擎"""
    
    def __init__(self, template):
        self.template = template
    
    def render(self, **kwargs):
        # 使用动态代码生成实现模板渲染
        namespace = dict(kwargs)
        
        # 替换模板变量
        # 格式: {{ variable }}
        import re
        
        def replace_var(match):
            var_name = match.group(1).strip()
            return f"{{{{{var_name}}}}}"
        
        # 将模板转换为 Python 代码
        template_code = re.sub(r'\{\{(.*?)\}\}', replace_var, self.template)
        
        # 创建渲染函数
        render_code = f'''
def render_template({", ".join(namespace.keys())}):
    return f"""{template_code}"""
'''
        # 编译并执行渲染函数
        exec_namespace = {}
        exec(render_code, {}, exec_namespace)
        
        # 调用渲染函数
        return exec_namespace['render_template'](**namespace)

# 使用模板引擎
template = Template('''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>{{ header }}</h1>
    <p>欢迎, {{ name }}!</p>
    <p>您的年龄是: {{ age }}</p>
</body>
</html>
''')

html = template.render(
    title="个人资料",
    header="用户资料",
    name="钱七",
    age=45
)

print(html)
```

## 下一步

现在您已经了解了 Python 元编程的基础知识，您可以进一步探索更高级的主题，如性能优化技术。请查看 [Python 性能优化](/advanced/performance) 了解更多内容。 