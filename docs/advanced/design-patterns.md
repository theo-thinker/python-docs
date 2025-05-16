# Python 设计模式

设计模式是解决软件设计中常见问题的可复用方案。它们提供了经过验证的开发范式，有助于提高代码的可维护性、可读性和可扩展性。本章将介绍在 Python 中实现的常见设计模式。

## 设计模式基础

设计模式可以分为三大类：

1. **创建型模式**：处理对象创建机制，试图以适合当前情况的方式创建对象
2. **结构型模式**：关注类和对象的组合，形成更大的结构
3. **行为型模式**：关注对象之间的责任分配和通信

## 创建型设计模式

### 单例模式 (Singleton)

确保一个类只有一个实例，并提供对该实例的全局访问点。

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用单例
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # 输出: True
```

更现代的 Python 单例实现方式：

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # 只初始化一次
            self.value = 0
            self._initialized = True

# 使用装饰器实现单例
def singleton(cls):
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Logger:
    def __init__(self, name="Default"):
        self.name = name
    
    def log(self, message):
        print(f"[{self.name}] {message}")

# 测试
logger1 = Logger("App")
logger2 = Logger("System")  # 不会创建新实例
logger1.log("测试消息")
logger2.log("另一条消息")  # 仍使用名称 "App"
```

应用场景：数据库连接、全局配置、日志记录器等。

### 工厂模式 (Factory)

定义一个创建对象的接口，让子类决定实例化哪个类。

#### 简单工厂

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "汪汪!"

class Cat(Animal):
    def speak(self):
        return "喵喵!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"未知动物类型: {animal_type}")

# 使用简单工厂
factory = AnimalFactory()
dog = factory.create_animal("dog")
cat = factory.create_animal("cat")
print(dog.speak())  # 输出: 汪汪!
print(cat.speak())  # 输出: 喵喵!
```

#### 工厂方法

```python
from abc import ABC, abstractmethod

class Creator(ABC):
    @abstractmethod
    def factory_method(self):
        pass
    
    def operation(self):
        product = self.factory_method()
        return f"使用 {product.operation()}"

class ConcreteCreator1(Creator):
    def factory_method(self):
        return ConcreteProduct1()

class ConcreteCreator2(Creator):
    def factory_method(self):
        return ConcreteProduct2()

class Product(ABC):
    @abstractmethod
    def operation(self):
        pass

class ConcreteProduct1(Product):
    def operation(self):
        return "产品1"

class ConcreteProduct2(Product):
    def operation(self):
        return "产品2"

# 使用工厂方法
creator1 = ConcreteCreator1()
creator2 = ConcreteCreator2()
print(creator1.operation())  # 输出: 使用 产品1
print(creator2.operation())  # 输出: 使用 产品2
```

#### 抽象工厂

```python
from abc import ABC, abstractmethod

# 抽象产品
class Button(ABC):
    @abstractmethod
    def paint(self):
        pass

class Checkbox(ABC):
    @abstractmethod
    def paint(self):
        pass

# 具体产品
class WindowsButton(Button):
    def paint(self):
        return "渲染 Windows 按钮"

class WindowsCheckbox(Checkbox):
    def paint(self):
        return "渲染 Windows 复选框"

class MacButton(Button):
    def paint(self):
        return "渲染 Mac 按钮"

class MacCheckbox(Checkbox):
    def paint(self):
        return "渲染 Mac 复选框"

# 抽象工厂
class GUIFactory(ABC):
    @abstractmethod
    def create_button(self):
        pass
    
    @abstractmethod
    def create_checkbox(self):
        pass

# 具体工厂
class WindowsFactory(GUIFactory):
    def create_button(self):
        return WindowsButton()
    
    def create_checkbox(self):
        return WindowsCheckbox()

class MacFactory(GUIFactory):
    def create_button(self):
        return MacButton()
    
    def create_checkbox(self):
        return MacCheckbox()

# 客户端代码
def create_ui(factory):
    button = factory.create_button()
    checkbox = factory.create_checkbox()
    return button.paint(), checkbox.paint()

# 根据操作系统选择工厂
import platform
if platform.system() == "Windows":
    factory = WindowsFactory()
else:
    factory = MacFactory()

button_ui, checkbox_ui = create_ui(factory)
print(button_ui)
print(checkbox_ui)
```

应用场景：跨平台 UI 组件、数据库连接器等。

### 建造者模式 (Builder)

将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示。

```python
class Computer:
    def __init__(self):
        self.parts = []
    
    def add(self, part):
        self.parts.append(part)
    
    def list_parts(self):
        return f"电脑配置: {', '.join(self.parts)}"

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()
    
    def add_cpu(self, cpu):
        self.computer.add(f"CPU: {cpu}")
        return self
    
    def add_memory(self, memory):
        self.computer.add(f"内存: {memory}GB")
        return self
    
    def add_storage(self, storage):
        self.computer.add(f"存储: {storage}GB")
        return self
    
    def add_gpu(self, gpu):
        self.computer.add(f"显卡: {gpu}")
        return self
    
    def build(self):
        return self.computer

# 主管类，定义构建顺序
class Director:
    def construct_gaming_pc(self, builder):
        return builder.add_cpu("Intel i9").add_memory(32).add_storage(1000).add_gpu("RTX 3080").build()
    
    def construct_office_pc(self, builder):
        return builder.add_cpu("Intel i5").add_memory(16).add_storage(512).build()

# 使用建造者模式
director = Director()
builder = ComputerBuilder()

gaming_pc = director.construct_gaming_pc(builder)
print(gaming_pc.list_parts())

office_pc = director.construct_office_pc(ComputerBuilder())
print(office_pc.list_parts())

# 也可以直接使用建造者，进行链式调用
custom_pc = ComputerBuilder().add_cpu("AMD Ryzen 7").add_memory(64).add_storage(2000).add_gpu("RTX 3070").build()
print(custom_pc.list_parts())
```

应用场景：构建复杂的对象，如网络请求、配置对象、复杂文档等。

### 原型模式 (Prototype)

通过克隆现有实例来创建新的对象实例，而不是创建新实例。

```python
import copy

class Prototype:
    def __init__(self):
        self.objects = {}
    
    def register(self, name, obj):
        self.objects[name] = obj
    
    def unregister(self, name):
        del self.objects[name]
    
    def clone(self, name, **attrs):
        """深拷贝注册的对象，并更新属性"""
        obj = copy.deepcopy(self.objects.get(name))
        obj.__dict__.update(attrs)
        return obj

class Car:
    def __init__(self, name, brand, model, color, extras=None):
        self.name = name
        self.brand = brand
        self.model = model
        self.color = color
        self.extras = extras or []
    
    def __str__(self):
        return f"{self.name} ({self.brand} {self.model}, {self.color}, 配件: {', '.join(self.extras)})"

# 使用原型模式
car_prototype = Car(
    name="基本轿车",
    brand="通用品牌",
    model="基本款",
    color="白色",
    extras=["空调", "音响系统"]
)

prototype_manager = Prototype()
prototype_manager.register("基本轿车", car_prototype)

# 克隆并定制
sport_car = prototype_manager.clone(
    "基本轿车", 
    name="运动轿车", 
    color="红色", 
    extras=["空调", "音响系统", "运动套件", "涡轮增压"]
)

luxury_car = prototype_manager.clone(
    "基本轿车", 
    name="豪华轿车", 
    color="黑色", 
    extras=["空调", "高级音响系统", "真皮座椅", "自动泊车"]
)

print(car_prototype)
print(sport_car)
print(luxury_car)
```

应用场景：需要创建大量相似对象的地方，如游戏对象、GUI 组件的克隆等。

## 结构型设计模式

### 适配器模式 (Adapter)

将一个类的接口转换成客户希望的另一个接口，使原本由于接口不兼容而不能一起工作的类能一起工作。

```python
# 现有类，使用摄氏度
class CelsiusSensor:
    def get_temperature(self):
        return 25.0  # 返回摄氏度

# 目标接口，使用华氏度
class TemperatureSensor:
    def get_fahrenheit(self):
        pass

# 适配器类
class CelsiusToFahrenheitAdapter(TemperatureSensor):
    def __init__(self, celsius_sensor):
        self.celsius_sensor = celsius_sensor
    
    def get_fahrenheit(self):
        # 转换摄氏度到华氏度
        celsius = self.celsius_sensor.get_temperature()
        return celsius * 9/5 + 32

# 客户端代码，期望华氏度接口
def display_temperature(temp_sensor):
    print(f"当前温度: {temp_sensor.get_fahrenheit():.1f}°F")

# 使用适配器
celsius_sensor = CelsiusSensor()
adapter = CelsiusToFahrenheitAdapter(celsius_sensor)
display_temperature(adapter)  # 输出华氏度温度
```

还可以使用对象适配器和类适配器：

```python
# 对象适配器（通过组合）
class ObjectAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee
    
    def target_method(self):
        # 调用被适配对象的方法并进行转换
        return self.adaptee.specific_method() + " (已适配)"

# 类适配器（通过多重继承）
class SpecificClass:
    def specific_method(self):
        return "特定方法结果"

class TargetInterface:
    def target_method(self):
        pass

class ClassAdapter(SpecificClass, TargetInterface):
    def target_method(self):
        # 调用父类的方法并进行转换
        return self.specific_method() + " (已适配)"
```

应用场景：集成第三方库、API 接口适配、兼容旧系统等。

### 桥接模式 (Bridge)

将抽象部分与实现部分分离，使它们可以独立变化。

```python
from abc import ABC, abstractmethod

# 实现部分接口
class Device(ABC):
    @abstractmethod
    def is_enabled(self):
        pass
    
    @abstractmethod
    def enable(self):
        pass
    
    @abstractmethod
    def disable(self):
        pass
    
    @abstractmethod
    def get_volume(self):
        pass
    
    @abstractmethod
    def set_volume(self, volume):
        pass

# 具体实现类
class TV(Device):
    def __init__(self):
        self.enabled = False
        self.volume = 30
    
    def is_enabled(self):
        return self.enabled
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def get_volume(self):
        return self.volume
    
    def set_volume(self, volume):
        if volume > 100:
            self.volume = 100
        elif volume < 0:
            self.volume = 0
        else:
            self.volume = volume

class Radio(Device):
    def __init__(self):
        self.enabled = False
        self.volume = 20
    
    def is_enabled(self):
        return self.enabled
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def get_volume(self):
        return self.volume
    
    def set_volume(self, volume):
        if volume > 100:
            self.volume = 100
        elif volume < 0:
            self.volume = 0
        else:
            self.volume = volume

# 抽象部分
class RemoteControl:
    def __init__(self, device):
        self.device = device
    
    def toggle_power(self):
        if self.device.is_enabled():
            self.device.disable()
        else:
            self.device.enable()
    
    def volume_up(self):
        self.device.set_volume(self.device.get_volume() + 10)
    
    def volume_down(self):
        self.device.set_volume(self.device.get_volume() - 10)

# 扩展抽象部分
class AdvancedRemoteControl(RemoteControl):
    def mute(self):
        self.device.set_volume(0)

# 客户端代码
tv = TV()
basic_remote = RemoteControl(tv)
advanced_remote = AdvancedRemoteControl(tv)

basic_remote.toggle_power()
print(f"电视已开启: {tv.is_enabled()}")
basic_remote.volume_up()
print(f"电视音量: {tv.get_volume()}")

advanced_remote.mute()
print(f"电视音量（静音后）: {tv.get_volume()}")

# 使用相同的遥控器控制不同设备
radio = Radio()
radio_remote = AdvancedRemoteControl(radio)
radio_remote.toggle_power()
radio_remote.volume_up()
print(f"收音机已开启: {radio.is_enabled()}")
print(f"收音机音量: {radio.get_volume()}")
```

应用场景：多维度变化的系统、跨平台应用、驱动程序等。

### 组合模式 (Composite)

将对象组合成树形结构以表示"部分-整体"的层次结构，使客户端对单个对象和组合对象的使用具有一致性。

```python
from abc import ABC, abstractmethod

# 组件接口
class Component(ABC):
    def __init__(self, name):
        self.name = name
    
    def add(self, component):
        pass
    
    def remove(self, component):
        pass
    
    @abstractmethod
    def display(self, indent=""):
        pass
    
    @abstractmethod
    def get_size(self):
        pass

# 叶子节点
class File(Component):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def display(self, indent=""):
        print(f"{indent}- 文件: {self.name} ({self.size} KB)")
    
    def get_size(self):
        return self.size

# 组合节点
class Directory(Component):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, component):
        self.children.append(component)
    
    def remove(self, component):
        self.children.remove(component)
    
    def display(self, indent=""):
        print(f"{indent}+ 目录: {self.name} ({self.get_size()} KB)")
        for child in self.children:
            child.display(indent + "  ")
    
    def get_size(self):
        total_size = 0
        for child in self.children:
            total_size += child.get_size()
        return total_size

# 客户端代码
def main():
    # 创建文件系统结构
    root = Directory("root")
    
    docs = Directory("documents")
    docs.add(File("resume.pdf", 1024))
    docs.add(File("cover_letter.docx", 2048))
    
    photos = Directory("photos")
    vacation = Directory("vacation")
    vacation.add(File("beach.jpg", 5120))
    vacation.add(File("mountain.jpg", 4096))
    photos.add(vacation)
    photos.add(File("profile.png", 1024))
    
    root.add(docs)
    root.add(photos)
    root.add(File("config.ini", 512))
    
    # 显示整个文件系统
    root.display()
    
    # 只显示照片目录
    print("\n照片目录内容:")
    photos.display()
    
    # 计算大小
    print(f"\n文档目录大小: {docs.get_size()} KB")
    print(f"照片目录大小: {photos.get_size()} KB")
    print(f"根目录总大小: {root.get_size()} KB")

if __name__ == "__main__":
    main()
```

应用场景：文件系统、GUI 组件层次结构、组织结构等。

### 装饰器模式 (Decorator)

动态地给一个对象添加额外的职责，比子类化更加灵活。

```python
from abc import ABC, abstractmethod

# 组件接口
class Coffee(ABC):
    @abstractmethod
    def get_cost(self):
        pass
    
    @abstractmethod
    def get_description(self):
        pass

# 具体组件
class SimpleCoffee(Coffee):
    def get_cost(self):
        return 10.0
    
    def get_description(self):
        return "简单咖啡"

# 装饰器基类
class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self.coffee = coffee
    
    def get_cost(self):
        return self.coffee.get_cost()
    
    def get_description(self):
        return self.coffee.get_description()

# 具体装饰器
class MilkDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.coffee.get_cost() + 3.0
    
    def get_description(self):
        return self.coffee.get_description() + ", 加奶"

class SugarDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.coffee.get_cost() + 1.0
    
    def get_description(self):
        return self.coffee.get_description() + ", 加糖"

class WhipDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.coffee.get_cost() + 5.0
    
    def get_description(self):
        return self.coffee.get_description() + ", 加奶油"

# 客户端代码
def make_coffee():
    # 创建简单咖啡
    coffee = SimpleCoffee()
    print(f"咖啡: {coffee.get_description()}, 价格: {coffee.get_cost()}元")
    
    # 加牛奶
    coffee = MilkDecorator(coffee)
    print(f"咖啡: {coffee.get_description()}, 价格: {coffee.get_cost()}元")
    
    # 加糖
    coffee = SugarDecorator(coffee)
    print(f"咖啡: {coffee.get_description()}, 价格: {coffee.get_cost()}元")
    
    # 加奶油
    coffee = WhipDecorator(coffee)
    print(f"咖啡: {coffee.get_description()}, 价格: {coffee.get_cost()}元")
    
    # 双份牛奶
    coffee = MilkDecorator(coffee)
    print(f"咖啡: {coffee.get_description()}, 价格: {coffee.get_cost()}元")

if __name__ == "__main__":
    make_coffee()
```

注意：Python 的装饰器语法 `@decorator` 是受这种设计模式启发，但实现方式不同。

应用场景：UI 组件添加边框/滚动条、数据流增加加密/压缩、日志记录等。

### 外观模式 (Facade)

为子系统中的一组接口提供一个一致的高层接口，使子系统更容易使用。

```python
# 复杂子系统
class CPU:
    def freeze(self):
        print("冻结 CPU")
    
    def jump(self, address):
        print(f"跳转到地址: {address}")
    
    def execute(self):
        print("执行 CPU 指令")

class Memory:
    def load(self, address, data):
        print(f"从内存加载数据，地址: {address}, 数据: {data}")

class HardDrive:
    def read(self, sector, size):
        print(f"从硬盘读取数据，扇区: {sector}, 大小: {size}")
        return f"硬盘数据 {sector}"

# 外观类
class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
    
    def start(self):
        print("\n启动计算机...")
        self.cpu.freeze()
        self.memory.load(0, "引导数据")
        self.cpu.jump(0)
        self.cpu.execute()
    
    def shutdown(self):
        print("\n关闭计算机...")
        # 执行关机操作
    
    def open_browser(self):
        print("\n打开浏览器...")
        # 模拟打开浏览器的复杂操作
        self.cpu.freeze()
        self.memory.load(100, "浏览器程序")
        browser_data = self.hard_drive.read(2000, 1024)
        self.memory.load(200, browser_data)
        self.cpu.jump(100)
        self.cpu.execute()
    
    def open_text_editor(self):
        print("\n打开文本编辑器...")
        # 模拟打开文本编辑器的复杂操作
        self.cpu.freeze()
        self.memory.load(300, "文本编辑器程序")
        editor_data = self.hard_drive.read(3000, 512)
        self.memory.load(400, editor_data)
        self.cpu.jump(300)
        self.cpu.execute()

# 客户端代码
def main():
    computer = ComputerFacade()
    
    # 使用简单的接口操作复杂子系统
    computer.start()
    computer.open_browser()
    computer.open_text_editor()
    computer.shutdown()

if __name__ == "__main__":
    main()
```

应用场景：简化复杂系统接口、提供统一入口、封装第三方库等。 