# Python 变量与数据类型

变量是存储数据值的容器，Python 是一种动态类型语言，这意味着变量的类型可以随时改变。

## 变量声明与赋值

在 Python 中，变量不需要显式声明数据类型。当您为变量赋值时，变量会自动创建：

```python
# 变量赋值
name = "张三"
age = 25
height = 1.75
is_student = True

# 多个变量同时赋值
a, b, c = 1, 2, 3

# 多个变量赋相同的值
x = y = z = 0
```

## 变量命名规则

- 变量名只能包含字母、数字和下划线
- 变量名必须以字母或下划线开头
- 变量名区分大小写（`name`、`Name` 和 `NAME` 是三个不同的变量）
- 变量名不能使用 Python 关键字（如 `if`、`while` 等）

```python
# 有效的变量名
name = "John"
age2 = 30
_private = "私有变量"
first_name = "张"
lastName = "三"

# 无效的变量名
# 2name = "错误"  # 不能以数字开头
# my-name = "错误"  # 不能包含连字符
# class = "错误"  # 不能使用关键字
```

## Python 的数据类型

Python 有以下几种标准数据类型：

### 1. 数字（Number）

Python 支持多种数字类型：

```python
# 整数（Int）
x = 10
big_num = 1234567890123456789

# 浮点数（Float）
y = 10.5
scientific = 1.23e4  # 12300.0

# 复数（Complex）
z = 3 + 4j
```

### 2. 字符串（String）

字符串是由字符组成的序列：

```python
# 字符串创建
s1 = 'Hello'
s2 = "World"
s3 = '''多行
字符串'''

# 字符串索引（从0开始）
print(s1[0])  # 输出: H

# 字符串切片
print(s1[1:3])  # 输出: el

# 字符串拼接
full = s1 + " " + s2  # Hello World

# 字符串重复
repeat = s1 * 3  # HelloHelloHello

# 字符串方法
upper_case = s1.upper()  # HELLO
lower_case = s2.lower()  # world
replaced = s1.replace('H', 'J')  # Jello
```

### 3. 列表（List）

列表是有序、可变的集合：

```python
# 列表创建
fruits = ['苹果', '香蕉', '橙子']
mixed = [1, 'Hello', 3.14, True]

# 访问列表元素
print(fruits[0])  # 苹果

# 修改列表元素
fruits[1] = '梨'

# 列表切片
print(fruits[0:2])  # ['苹果', '梨']

# 列表方法
fruits.append('葡萄')  # 添加元素
fruits.remove('苹果')  # 删除元素
fruits.sort()  # 排序
length = len(fruits)  # 获取长度
```

### 4. 元组（Tuple）

元组是有序、不可变的集合：

```python
# 元组创建
coordinates = (10, 20)
person = ('张三', 25, '工程师')

# 访问元组元素
print(coordinates[0])  # 10

# 元组不能修改元素
# coordinates[0] = 15  # 这将引发错误

# 但可以连接元组
new_tuple = coordinates + (30, 40)  # (10, 20, 30, 40)

# 单元素元组需要逗号
single_item = (42,)  # 注意逗号
```

### 5. 字典（Dictionary）

字典是无序、可变的键值对集合：

```python
# 字典创建
person = {
    'name': '张三',
    'age': 25,
    'job': '工程师'
}

# 访问字典元素
print(person['name'])  # 张三

# 修改字典元素
person['age'] = 26

# 添加新键值对
person['city'] = '北京'

# 字典方法
keys = person.keys()  # 获取所有键
values = person.values()  # 获取所有值
person.pop('city')  # 删除指定键值对
```

### 6. 集合（Set）

集合是无序、不重复的元素集合：

```python
# 集合创建
fruits = {'苹果', '香蕉', '橙子'}
numbers = {1, 2, 3, 2, 1}  # 自动去重：{1, 2, 3}

# 集合方法
fruits.add('葡萄')  # 添加元素
fruits.remove('香蕉')  # 删除元素

# 集合运算
a = {1, 2, 3}
b = {3, 4, 5}
union = a | b  # 并集：{1, 2, 3, 4, 5}
intersection = a & b  # 交集：{3}
difference = a - b  # 差集：{1, 2}
```

### 7. 布尔（Boolean）

布尔类型只有两个值：`True` 和 `False`：

```python
is_valid = True
has_errors = False

# 布尔运算
result1 = True and False  # False
result2 = True or False  # True
result3 = not True  # False
```

## 类型转换

Python 提供了以下内置函数用于类型转换：

```python
# 字符串转整数
x = int("10")  # 10

# 整数转字符串
s = str(10)  # "10"

# 字符串转浮点数
y = float("10.5")  # 10.5

# 整数转浮点数
z = float(10)  # 10.0

# 转换为布尔值
b1 = bool(0)  # False
b2 = bool(1)  # True
b3 = bool("")  # False
b4 = bool("Hello")  # True

# 转换为列表
l1 = list("Hello")  # ['H', 'e', 'l', 'l', 'o']
l2 = list((1, 2, 3))  # [1, 2, 3]

# 转换为元组
t = tuple([1, 2, 3])  # (1, 2, 3)

# 转换为集合
s = set([1, 2, 2, 3])  # {1, 2, 3}
```

## 检查变量类型

`type()` 函数用于检查变量的类型：

```python
x = 10
y = "Hello"
z = [1, 2, 3]

print(type(x))  # <class 'int'>
print(type(y))  # <class 'str'>
print(type(z))  # <class 'list'>

# 使用 isinstance() 函数检查实例
print(isinstance(x, int))  # True
print(isinstance(y, str))  # True
print(isinstance(z, list))  # True
```

## 变量作用域

变量有以下几种作用域：

1. **局部作用域**：在函数内部定义的变量
2. **全局作用域**：在函数外部定义的变量
3. **嵌套作用域**：在嵌套函数中定义的变量

```python
# 全局变量
global_var = 10

def func():
    # 局部变量
    local_var = 20
    print(global_var)  # 可以访问全局变量
    print(local_var)

func()
print(global_var)  # 10
# print(local_var)  # 这会引发错误，因为局部变量只在函数内部有效

# 使用 global 关键字修改全局变量
def modify_global():
    global global_var
    global_var = 30

modify_global()
print(global_var)  # 30
```

## 常量

Python 没有内置的常量类型，但通常使用全大写名称的变量表示常量：

```python
PI = 3.14159
MAX_CONNECTIONS = 1000

# 这只是一种约定，实际上这些变量仍然可以被修改
```

## 下一步

现在您已经了解了 Python 的变量和数据类型，接下来可以学习 [Python 运算符](/basic/operators)。 