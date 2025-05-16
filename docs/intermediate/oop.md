# Python 面向对象编程

面向对象编程（Object-Oriented Programming，简称 OOP）是一种编程范式，它使用"对象"来组织代码。Python 是一种多范式的编程语言，完全支持面向对象编程。

## 面向对象编程基础

### 类与对象

类（Class）是对象（Object）的蓝图或模板，定义了对象的属性和方法。对象是类的实例。

```python
# 定义一个简单的类
class Dog:
    # 类属性（被所有实例共享）
    species = "哺乳动物"
    
    # 初始化方法（构造函数）
    def __init__(self, name, age):
        # 实例属性（每个实例的属性可能不同）
        self.name = name
        self.age = age
    
    # 实例方法
    def bark(self):
        return f"{self.name} 说：汪汪！"
    
    def get_info(self):
        return f"{self.name} 是 {self.age} 岁的 {self.species}"

# 创建类的实例（对象）
dog1 = Dog("小黑", 3)
dog2 = Dog("小白", 2)

# 访问实例属性
print(dog1.name)  # 输出：小黑
print(dog2.age)   # 输出：2

# 调用实例方法
print(dog1.bark())     # 输出：小黑 说：汪汪！
print(dog2.get_info())  # 输出：小白 是 2 岁的 哺乳动物

# 访问类属性
print(Dog.species)  # 输出：哺乳动物
print(dog1.species)  # 通过实例也可以访问类属性
```

### self 参数

在类的方法中，`self` 是一个特殊参数，它指向当前的实例。它是 Python 中的约定（不是关键字）。

```python
class Person:
    def __init__(self, name):
        self.name = name
    
    def say_hello(self):
        # self 在这里引用调用此方法的实例
        return f"{self.name} 说：你好！"

p = Person("张三")
print(p.say_hello())  # 输出：张三 说：你好！
```

## 类的特性

### 属性与方法

Python 类可以有不同类型的属性和方法：

#### 实例属性与类属性

```python
class Circle:
    # 类属性
    pi = 3.14159
    
    def __init__(self, radius):
        # 实例属性
        self.radius = radius
    
    def area(self):
        return Circle.pi * self.radius ** 2

c1 = Circle(5)
print(c1.area())  # 输出：78.53975

# 修改类属性会影响所有实例
Circle.pi = 3.14
print(c1.area())  # 输出：78.5
```

#### 实例方法、类方法与静态方法

```python
class MyClass:
    count = 0
    
    def __init__(self):
        MyClass.count += 1
    
    # 实例方法（使用实例调用）
    def instance_method(self):
        return f"实例方法，可访问 self: {self}"
    
    # 类方法（使用类或实例调用）
    @classmethod
    def class_method(cls):
        return f"类方法，可访问 cls: {cls}, 实例数量: {cls.count}"
    
    # 静态方法（使用类或实例调用，但不能访问类或实例的状态）
    @staticmethod
    def static_method(x, y):
        return f"静态方法，无法访问类或实例的状态，但可以接收参数: {x}, {y}"

obj = MyClass()
print(obj.instance_method())
print(MyClass.class_method())
print(obj.class_method())  # 也可以通过实例调用类方法
print(MyClass.static_method(10, 20))
print(obj.static_method(10, 20))  # 也可以通过实例调用静态方法
```

### 封装

封装是将实现细节隐藏起来，只暴露必要的接口。Python 使用命名约定来实现封装。

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner      # 公有属性
        self._balance = balance  # 受保护的属性（约定）
        self.__id = "12345"     # 私有属性（通过名字改写实现）
    
    def deposit(self, amount):
        if amount <= 0:
            return "存款金额必须为正数"
        self._balance += amount
        return f"存款 {amount} 元成功，当前余额：{self._balance} 元"
    
    def withdraw(self, amount):
        if amount <= 0:
            return "取款金额必须为正数"
        if amount > self._balance:
            return "余额不足"
        self._balance -= amount
        return f"取款 {amount} 元成功，当前余额：{self._balance} 元"
    
    def get_balance(self):
        return self._balance
    
    def __get_id(self):  # 私有方法
        return self.__id

account = BankAccount("张三", 1000)
print(account.owner)  # 输出：张三
print(account.get_balance())  # 输出：1000
print(account.deposit(500))  # 输出：存款 500 元成功，当前余额：1500 元
print(account.withdraw(200))  # 输出：取款 200 元成功，当前余额：1300 元

# 可以访问受保护的属性，但约定不应该这样做
print(account._balance)  # 输出：1300

# 试图访问私有属性会失败
# print(account.__id)  # 会引发 AttributeError
# print(account.__get_id())  # 会引发 AttributeError

# 但可以通过名字改写访问私有属性（不推荐）
print(account._BankAccount__id)  # 输出：12345
```

### 属性装饰器

Python 提供了 `@property` 装饰器，可以将方法转换为类似属性的访问方式：

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
        first, last = name.split()
        self.first_name = first
        self.last_name = last

person = Person("张", "三")
print(person.full_name)  # 输出：张 三
print(person.email)  # 输出：张.三@example.com

# 使用 setter 修改 full_name
person.full_name = "李 四"
print(person.first_name)  # 输出：李
print(person.last_name)  # 输出：四
print(person.full_name)  # 输出：李 四
print(person.email)  # 输出：李.四@example.com

# email 只有 getter，没有 setter
# person.email = "test@example.com"  # 会引发 AttributeError
```

## 继承

继承允许一个类（子类）继承另一个类（父类）的属性和方法。

### 基本继承

```python
# 父类
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "某种声音"
    
    def get_info(self):
        return f"{self.name} 是一种 {self.species}"

# 子类
class Dog(Animal):  # Dog 继承自 Animal
    def __init__(self, name, breed):
        # 调用父类的 __init__ 方法
        super().__init__(name, species="犬类")
        self.breed = breed
    
    # 重写父类的方法
    def make_sound(self):
        return "汪汪！"
    
    # 扩展父类的方法
    def get_info(self):
        base_info = super().get_info()
        return f"{base_info}，品种是 {self.breed}"

# 创建一个 Dog 实例
dog = Dog("小黑", "拉布拉多")
print(dog.name)  # 继承自父类的属性
print(dog.species)  # 通过父类的 __init__ 设置的属性
print(dog.breed)  # 子类的属性
print(dog.make_sound())  # 子类重写的方法
print(dog.get_info())  # 子类扩展的方法
```

### 多重继承

Python 支持多重继承，一个类可以继承多个父类：

```python
class Flyable:
    def fly(self):
        return "我可以飞行"

class Swimmable:
    def swim(self):
        return "我可以游泳"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        super().__init__(name, species="鸭子")
    
    def make_sound(self):
        return "嘎嘎！"

duck = Duck("唐老鸭")
print(duck.get_info())  # 来自 Animal
print(duck.make_sound())  # 重写 Animal 的方法
print(duck.fly())  # 来自 Flyable
print(duck.swim())  # 来自 Swimmable
```

### 方法解析顺序 (MRO)

当一个类继承自多个类时，Python 使用 C3 线性化算法来确定方法解析顺序：

```python
class A:
    def method(self):
        return "A.method"

class B(A):
    def method(self):
        return "B.method"

class C(A):
    def method(self):
        return "C.method"

class D(B, C):
    pass

print(D.__mro__)  # 打印方法解析顺序
# 输出: (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

d = D()
print(d.method())  # 输出: B.method，因为 B 在 MRO 中排在 C 前面
```

### isinstance() 和 issubclass() 函数

```python
# isinstance() 检查对象是否是类或其子类的实例
print(isinstance(dog, Dog))       # True
print(isinstance(dog, Animal))    # True
print(isinstance(dog, Flyable))   # False

# issubclass() 检查一个类是否是另一个类的子类
print(issubclass(Dog, Animal))    # True
print(issubclass(Duck, Flyable))  # True
print(issubclass(Dog, Flyable))   # False
```

## 多态

多态允许使用一个统一的接口来操作不同类型的对象：

```python
def make_animal_sound(animal):
    return animal.make_sound()

dog = Dog("小黑", "拉布拉多")
duck = Duck("唐老鸭")

print(make_animal_sound(dog))  # 输出：汪汪！
print(make_animal_sound(duck))  # 输出：嘎嘎！
```

## 特殊方法（魔术方法）

Python 类可以实现一些特殊方法（双下划线开头和结尾），以支持各种内置操作：

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 字符串表示
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    # 更详细的字符串表示，用于调试
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # 定义加法操作
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # 定义减法操作
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    # 定义乘法操作（标量乘法）
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # 定义长度
    def __len__(self):
        return int((self.x ** 2 + self.y ** 2) ** 0.5)
    
    # 定义相等性
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1)  # 输出：Vector(3, 4)
print(v1 + v2)  # 输出：Vector(4, 6)
print(v1 - v2)  # 输出：Vector(2, 2)
print(v1 * 2)  # 输出：Vector(6, 8)
print(len(v1))  # 输出：5
print(v1 == Vector(3, 4))  # 输出：True
```

常用的特殊方法：

| 方法 | 描述 | 对应操作 |
|------|------|----------|
| `__init__(self, ...)` | 初始化对象 | `x = MyClass()` |
| `__str__(self)` | 返回字符串表示 | `str(x)`, `print(x)` |
| `__repr__(self)` | 返回详细的字符串表示 | `repr(x)` |
| `__len__(self)` | 返回对象的长度 | `len(x)` |
| `__getitem__(self, key)` | 实现索引操作 | `x[key]` |
| `__setitem__(self, key, value)` | 实现索引赋值 | `x[key] = value` |
| `__delitem__(self, key)` | 实现索引删除 | `del x[key]` |
| `__iter__(self)` | 返回迭代器对象 | `for i in x` |
| `__next__(self)` | 返回下一个迭代值 | `next(x)` |
| `__contains__(self, item)` | 实现成员检查 | `item in x` |
| `__call__(self, ...)` | 使对象可调用 | `x()` |
| `__add__(self, other)` | 实现加法 | `x + y` |
| `__sub__(self, other)` | 实现减法 | `x - y` |
| `__mul__(self, other)` | 实现乘法 | `x * y` |
| `__truediv__(self, other)` | 实现除法 | `x / y` |
| `__eq__(self, other)` | 实现等于 | `x == y` |
| `__lt__(self, other)` | 实现小于 | `x < y` |
| `__gt__(self, other)` | 实现大于 | `x > y` |

## 抽象基类

抽象基类（Abstract Base Class，ABC）定义了一个接口，但不提供完整的实现。子类必须实现这些抽象方法。

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass
    
    def description(self):
        return f"这是一个图形，面积是 {self.area()}，周长是 {self.perimeter()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# 尝试实例化 Shape 会失败，因为它是抽象类
# shape = Shape()  # 会引发 TypeError

# 实例化具体子类
rectangle = Rectangle(4, 5)
circle = Circle(3)

print(rectangle.description())  # 输出：这是一个图形，面积是 20，周长是 18
print(circle.description())  # 输出：这是一个图形，面积是 28.27431，周长是 18.84954
```

## 元类

元类是创建类的类，它定义了类的行为。Python 中所有的类都是 `type` 的实例。

```python
# 定义一个简单的元类
class Meta(type):
    def __new__(cls, name, bases, attrs):
        # 将所有方法名转为大写
        uppercase_attrs = {
            key.upper() if not key.startswith('__') else key: value
            for key, value in attrs.items()
        }
        return super().__new__(cls, name, bases, uppercase_attrs)

# 使用自定义元类的类
class MyClass(metaclass=Meta):
    def hello(self):
        return "Hello, World!"
    
    def goodbye(self):
        return "Goodbye, World!"

obj = MyClass()
# 注意：方法名已经变成大写了
print(obj.HELLO())  # 输出：Hello, World!
print(obj.GOODBYE())  # 输出：Goodbye, World!
```

## 数据类

Python 3.7 引入了数据类（Data Classes），它们是主要用于存储数据的类，减少了样板代码：

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Rectangle:
    width: float
    height: float
    
    def area(self):
        return self.width * self.height

p = Point(1, 2)
print(p)  # 输出：Point(x=1, y=2)
print(p.x, p.y)  # 输出：1 2

r = Rectangle(4.0, 5.0)
print(r)  # 输出：Rectangle(width=4.0, height=5.0)
print(r.area())  # 输出：20.0
```

`@dataclass` 装饰器会自动生成 `__init__`、`__repr__` 和 `__eq__` 方法，使类的定义更简洁。

## 设计模式

面向对象编程中有许多常用的设计模式，以下是几个常见的例子：

### 单例模式

确保一个类只有一个实例，并提供全局访问点：

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 测试单例模式
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # 输出：True，表明 s1 和 s2 是同一个对象
```

### 工厂模式

创建对象的接口，让子类决定实例化哪个类：

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "汪汪！"

class Cat(Animal):
    def speak(self):
        return "喵喵！"

class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"不支持的动物类型：{animal_type}")

# 使用工厂创建对象
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")
print(dog.speak())  # 输出：汪汪！
print(cat.speak())  # 输出：喵喵！
```

### 观察者模式

定义对象间的一种一对多依赖关系，使得一个对象状态改变时，所有依赖于它的对象都会得到通知：

```python
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass
    
    def notify(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(self, *args, **kwargs)

class Observer:
    def update(self, subject, *args, **kwargs):
        pass

# 具体实现
class WeatherStation(Subject):
    def __init__(self):
        super().__init__()
        self._temperature = 0
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.notify(value)

class TemperatureDisplay(Observer):
    def update(self, subject, *args, **kwargs):
        if args:
            print(f"温度显示器：当前温度为 {args[0]} 度")

class TemperatureLogger(Observer):
    def update(self, subject, *args, **kwargs):
        if args:
            print(f"温度记录器：记录到温度变化为 {args[0]} 度")

# 测试观察者模式
station = WeatherStation()
display = TemperatureDisplay()
logger = TemperatureLogger()

station.attach(display)
station.attach(logger)

station.temperature = 25  
# 输出：
# 温度显示器：当前温度为 25 度
# 温度记录器：记录到温度变化为 25 度

station.detach(logger)
station.temperature = 30
# 输出：
# 温度显示器：当前温度为 30 度
```

## 最佳实践

1. **遵循单一职责原则**：每个类应该只负责一个明确的责任。

2. **使用合适的访问控制**：使用命名约定（单下划线前缀表示保护属性，双下划线前缀表示私有属性）来控制属性和方法的访问。

3. **选择组合而非继承**：当可以选择时，优先使用组合（将其他类作为成员变量）而不是继承，这样可以降低耦合度。

4. **不要过度设计**：根据实际需要使用面向对象特性，避免过度工程化。

5. **使用描述性命名**：为类、方法和属性选择清晰、描述性的名称。

6. **遵循 Pythonic 风格**：遵循 Python 的习惯和风格，例如使用属性装饰器而不是 getter/setter 方法。

7. **编写文档字符串**：为类和方法提供详细的文档字符串，说明它们的用途、参数和返回值。

## 下一步

现在您已经了解了 Python 面向对象编程的基础，接下来可以学习 [Python 文件操作](/intermediate/files)，探索如何使用 Python 处理文件和目录。 