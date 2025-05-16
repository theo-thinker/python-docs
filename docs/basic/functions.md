# Python 函数

函数是组织好的、可重复使用的、用来实现特定功能的代码块。Python 提供了许多内置函数，同时也允许您创建自定义函数。

## 函数定义与调用

使用 `def` 关键字定义函数：

```python
# 定义函数
def greet():
    print("Hello, World!")

# 调用函数
greet()  # 输出: Hello, World!
```

## 函数参数

函数可以接受参数，这些参数在调用函数时传递给函数：

```python
# 带参数的函数
def greet(name):
    print(f"Hello, {name}!")

greet("张三")  # 输出: Hello, 张三!
```

### 位置参数

位置参数是按照定义顺序传递的参数：

```python
def describe_person(name, age, occupation):
    print(f"{name} 是一名 {age} 岁的 {occupation}")

describe_person("张三", 25, "工程师")  # 输出: 张三 是一名 25 岁的 工程师
```

### 关键字参数

关键字参数允许您通过参数名传递参数，无需考虑顺序：

```python
def describe_person(name, age, occupation):
    print(f"{name} 是一名 {age} 岁的 {occupation}")

describe_person(age=25, occupation="工程师", name="张三")  # 输出: 张三 是一名 25 岁的 工程师
```

### 默认参数值

函数参数可以有默认值，如果调用时未提供该参数，则使用默认值：

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("张三")  # 输出: Hello, 张三!
greet("李四", "Hi")  # 输出: Hi, 李四!
```

::: warning 注意
默认参数值应该是不可变对象，如字符串、数字或 `None`，避免使用可变对象（如列表、字典）作为默认值。
:::

```python
# 错误示例：使用可变对象作为默认值
def add_to_list(item, my_list=[]):  # 不推荐
    my_list.append(item)
    return my_list

print(add_to_list("a"))  # 输出: ['a']
print(add_to_list("b"))  # 输出: ['a', 'b'] (而不是预期的 ['b'])

# 正确示例
def add_to_list(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list

print(add_to_list("a"))  # 输出: ['a']
print(add_to_list("b"))  # 输出: ['b']
```

### 可变参数 (*args)

使用 `*args` 收集任意数量的位置参数到一个元组中：

```python
def sum_numbers(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

print(sum_numbers(1, 2))  # 输出: 3
print(sum_numbers(1, 2, 3, 4, 5))  # 输出: 15
```

### 关键字可变参数 (**kwargs)

使用 `**kwargs` 收集任意数量的关键字参数到一个字典中：

```python
def print_person_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_person_info(name="张三", age=25, city="北京", occupation="工程师")
# 输出:
# name: 张三
# age: 25
# city: 北京
# occupation: 工程师
```

### 参数顺序

在函数定义中，参数必须按照以下顺序排列：
1. 位置参数
2. 默认参数
3. 可变位置参数 (`*args`)
4. 关键字参数
5. 关键字可变参数 (`**kwargs`)

```python
def complex_function(a, b, c=10, *args, d=20, **kwargs):
    print(f"a: {a}, b: {b}, c: {c}, args: {args}, d: {d}, kwargs: {kwargs}")

complex_function(1, 2, 3, 4, 5, 6, d=30, e=40, f=50)
# 输出: a: 1, b: 2, c: 3, args: (4, 5, 6), d: 30, kwargs: {'e': 40, 'f': 50}
```

### 仅位置参数和仅关键字参数

Python 3.8 引入了更灵活的参数定义方式：

```python
# 使用 / 表示其之前的参数为仅位置参数（不能使用关键字）
# 使用 * 表示其之后的参数为仅关键字参数（必须使用关键字）
def advanced_function(a, b, /, c, *, d):
    print(f"a: {a}, b: {b}, c: {c}, d: {d}")

# 正确调用方式
advanced_function(1, 2, 3, d=4)
advanced_function(1, 2, c=3, d=4)

# 错误调用方式
# advanced_function(a=1, b=2, c=3, d=4)  # a, b 不能使用关键字
# advanced_function(1, 2, 3, 4)  # d 必须使用关键字
```

## 返回值

函数可以使用 `return` 语句返回值：

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 输出: 8
```

没有 `return` 语句或者 `return` 后没有表达式的函数会返回 `None`：

```python
def no_return():
    print("This function doesn't return anything")

result = no_return()
print(result)  # 输出: None
```

函数可以返回多个值（实际上是返回一个元组）：

```python
def get_min_max(numbers):
    return min(numbers), max(numbers)

min_val, max_val = get_min_max([1, 5, 3, 9, 2])
print(f"最小值: {min_val}, 最大值: {max_val}")  # 输出: 最小值: 1, 最大值: 9
```

## 变量作用域

变量的作用域决定了变量在代码中的可见性：

### 局部作用域

在函数内部定义的变量是局部变量，只能在函数内部访问：

```python
def func():
    local_var = 10  # 局部变量
    print(local_var)

func()  # 输出: 10
# print(local_var)  # 错误: 局部变量在函数外部不可见
```

### 全局作用域

在函数外部定义的变量是全局变量，可以在整个模块中访问：

```python
global_var = 10  # 全局变量

def func():
    print(global_var)  # 可以访问全局变量

func()  # 输出: 10
```

如果要在函数内部修改全局变量，需要使用 `global` 关键字：

```python
counter = 0  # 全局变量

def increment():
    global counter  # 声明 counter 是全局变量
    counter += 1
    print(counter)

increment()  # 输出: 1
increment()  # 输出: 2
```

### 嵌套作用域

在嵌套函数中，内部函数可以访问外部函数的变量：

```python
def outer():
    outer_var = "外部变量"
    
    def inner():
        print(outer_var)  # 可以访问外部函数的变量
    
    inner()

outer()  # 输出: 外部变量
```

如果要在内部函数修改外部函数的变量，需要使用 `nonlocal` 关键字：

```python
def counter():
    count = 0
    
    def increment():
        nonlocal count  # 声明 count 是外部函数的变量
        count += 1
        return count
    
    return increment

counter_func = counter()
print(counter_func())  # 输出: 1
print(counter_func())  # 输出: 2
```

## 递归函数

递归函数是调用自身的函数：

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # 输出: 120 (5 * 4 * 3 * 2 * 1)
```

::: warning 注意
Python 的默认递归深度限制为 1000。对于深度递归，可能需要考虑使用迭代方法或增加递归限制：
:::

```python
import sys
print(sys.getrecursionlimit())  # 默认通常是 1000
sys.setrecursionlimit(3000)  # 设置为更高的值
```

## 匿名函数 (lambda)

Lambda 函数是一种简洁的、匿名的单行函数：

```python
# 普通函数
def square(x):
    return x ** 2

# 等效的 lambda 函数
square_lambda = lambda x: x ** 2

print(square(5))       # 输出: 25
print(square_lambda(5))  # 输出: 25
```

Lambda 函数通常与内置函数如 `map()`、`filter()` 和 `sorted()` 一起使用：

```python
# 使用 map() 对列表中的每个元素应用函数
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # 输出: [1, 4, 9, 16, 25]

# 使用 filter() 过滤列表中的元素
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # 输出: [2, 4]

# 使用 sorted() 自定义排序
people = [
    {"name": "张三", "age": 30},
    {"name": "李四", "age": 25},
    {"name": "王五", "age": 35}
]
sorted_people = sorted(people, key=lambda person: person["age"])
print(sorted_people)  # 按年龄排序
```

## 高阶函数

高阶函数是接受其他函数作为参数或返回函数的函数：

### 函数作为参数

```python
def apply(func, value):
    return func(value)

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

print(apply(square, 3))  # 输出: 9
print(apply(cube, 3))    # 输出: 27
```

### 返回函数

```python
def get_math_func(operation):
    if operation == "add":
        def add(x, y):
            return x + y
        return add
    elif operation == "multiply":
        def multiply(x, y):
            return x * y
        return multiply
    else:
        def default(x, y):
            return None
        return default

add_func = get_math_func("add")
multiply_func = get_math_func("multiply")

print(add_func(3, 5))       # 输出: 8
print(multiply_func(3, 5))  # 输出: 15
```

## 闭包

闭包是一个函数，它记住其外部作用域中的值，即使外部作用域已经执行完毕：

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor  # factor 是外部函数的变量
    return multiplier

# 创建两个闭包
double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 输出: 10
print(triple(5))  # 输出: 15
```

## 装饰器

装饰器是修改其他函数的功能的函数，这将在 [装饰器](/intermediate/decorators) 章节中详细讨论。

## 实际应用示例

### 计算器函数

```python
def calculator(num1, num2, operation="+"):
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y if y != 0 else "错误：除数不能为零"
    }
    
    if operation not in operations:
        return "不支持的操作"
    
    return operations[operation](num1, num2)

print(calculator(10, 5, "+"))  # 输出: 15
print(calculator(10, 5, "-"))  # 输出: 5
print(calculator(10, 5, "*"))  # 输出: 50
print(calculator(10, 5, "/"))  # 输出: 2.0
print(calculator(10, 0, "/"))  # 输出: 错误：除数不能为零
```

### 缓存函数结果（记忆化）

```python
def memoize(func):
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试效率
import time

start = time.time()
print(fibonacci(35))  # 使用缓存，很快
end = time.time()
print(f"计算耗时: {end - start:.6f} 秒")
```

### 创建API请求函数

```python
import requests

def create_api_caller(base_url, headers=None):
    """创建一个API调用函数"""
    
    def call_api(endpoint, method="GET", params=None, data=None):
        url = f"{base_url}/{endpoint}"
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=data,
            headers=headers
        )
        return response.json()
    
    return call_api

# 使用示例
github_api = create_api_caller("https://api.github.com")
user_info = github_api("users/python")
print(f"Python 组织的公开仓库数: {user_info.get('public_repos')}")
```

## 函数设计最佳实践

1. **函数应专注于单一职责**：每个函数应该只做一件事，并且做好。
2. **保持函数简短**：函数体应该简洁明了，避免过长的函数。
3. **使用描述性命名**：函数名应该清晰描述其功能。
4. **添加文档字符串**：使用文档字符串描述函数的功能、参数和返回值。
5. **参数数量适中**：避免过多的参数，通常不超过5个。
6. **使用异常而非返回错误代码**：当出现错误时，抛出异常而不是返回特殊值。
7. **保持纯函数**：尽可能让函数是纯函数（给定相同输入总是返回相同输出，且无副作用）。

## 下一步

现在您已经了解了 Python 的函数，接下来可以学习 [Python 模块与包](/basic/modules)。 