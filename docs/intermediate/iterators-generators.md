# Python 迭代器与生成器

迭代器和生成器是 Python 中非常强大的特性，它们让您能够高效地处理数据流和序列，尤其适合处理大型数据集。本章将详细介绍迭代器和生成器的概念、用法及最佳实践。

## 迭代器基础

### 什么是迭代器

迭代器是一个实现了迭代器协议的对象，它提供了一种访问容器（如列表或字典）中元素的方式，而不需要了解底层容器的具体实现。

迭代器协议包含两个方法：
- `__iter__()` 方法，返回迭代器对象自身
- `__next__()` 方法，返回序列中的下一个元素，如果没有更多元素则抛出 `StopIteration` 异常

### 可迭代对象和迭代器的区别

- **可迭代对象（Iterable）**：实现了 `__iter__()` 方法的对象，能够一次返回其元素的对象。例如：列表、元组、字典、集合、字符串等。
- **迭代器（Iterator）**：实现了 `__iter__()` 和 `__next__()` 方法的对象。

```python
# 可迭代对象的例子
iterable_list = [1, 2, 3]  # 列表是可迭代对象
iterable_dict = {'a': 1, 'b': 2}  # 字典是可迭代对象
iterable_str = "hello"  # 字符串是可迭代对象

# 使用 iter() 函数从可迭代对象创建迭代器
iterator = iter(iterable_list)
print(type(iterator))  # <class 'list_iterator'>

# 使用 next() 函数获取下一个元素
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3

try:
    print(next(iterator))  # 抛出 StopIteration 异常
except StopIteration:
    print("迭代器已经到达末尾")

# for 循环自动处理 StopIteration 异常
for value in iterable_list:
    print(value)  # 依次打印 1, 2, 3
```

### 创建自定义迭代器

通过实现 `__iter__()` 和 `__next__()` 方法，您可以创建自己的迭代器：

```python
class CountDown:
    """从指定数字倒数到零的迭代器"""
    
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

# 使用自定义迭代器
countdown = CountDown(3)
for num in countdown:
    print(num)  # 依次打印 3, 2, 1, 0
```

### 内置迭代器函数

Python 提供了许多内置函数来处理迭代器：

```python
# 使用 enumerate 获取索引和值
for i, value in enumerate(['a', 'b', 'c']):
    print(f"索引 {i}: {value}")

# 使用 zip 并行迭代多个迭代器
for name, age in zip(['Alice', 'Bob', 'Charlie'], [25, 30, 35]):
    print(f"{name} 今年 {age} 岁")

# 使用 map 对迭代器中的每个元素应用函数
squared = map(lambda x: x**2, [1, 2, 3, 4])
print(list(squared))  # [1, 4, 9, 16]

# 使用 filter 筛选元素
evens = filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
print(list(evens))  # [2, 4, 6]
```

## 生成器基础

### 什么是生成器

生成器是一种特殊类型的迭代器，它使用 `yield` 语句而不是 `return` 语句返回结果。每次调用 `next()` 时，生成器会从上次离开的地方继续执行，直到遇到下一个 `yield` 语句。

生成器的最大优势是它们可以 "惰性求值"（按需生成值），这使得它们在处理大型数据集时非常高效。

### 生成器函数

使用 `yield` 语句的函数被称为生成器函数。当调用生成器函数时，它不会立即执行，而是返回一个生成器对象：

```python
def countdown(n):
    """简单的生成器函数，从n倒数到0"""
    print("开始倒数！")
    while n >= 0:
        yield n
        n -= 1
    print("倒数结束！")

# 创建生成器对象
generator = countdown(3)
print(type(generator))  # <class 'generator'>

# 使用 next() 获取值
print(next(generator))  # 输出：开始倒数！ 3
print(next(generator))  # 输出：2
print(next(generator))  # 输出：1
print(next(generator))  # 输出：0

try:
    print(next(generator))  # 输出：倒数结束！然后抛出 StopIteration
except StopIteration:
    print("生成器已经到达末尾")

# 使用 for 循环遍历生成器
for value in countdown(3):
    print(value)  # 依次打印：开始倒数！ 3, 2, 1, 0, 倒数结束！
```

### 生成器表达式

生成器表达式是创建生成器的一种简洁方式，语法类似于列表推导式，但使用圆括号而不是方括号：

```python
# 列表推导式 - 创建完整列表
squares_list = [x**2 for x in range(5)]
print(squares_list)  # [0, 1, 4, 9, 16]
print(type(squares_list))  # <class 'list'>

# 生成器表达式 - 创建生成器
squares_generator = (x**2 for x in range(5))
print(squares_generator)  # <generator object <genexpr> at 0x...>
print(type(squares_generator))  # <class 'generator'>

# 使用生成器表达式
for square in squares_generator:
    print(square)  # 依次打印 0, 1, 4, 9, 16

# 生成器表达式与列表推导式的内存比较
import sys
list_comp = [x for x in range(10000)]
gen_exp = (x for x in range(10000))
print(f"列表大小: {sys.getsizeof(list_comp)} 字节")
print(f"生成器大小: {sys.getsizeof(gen_exp)} 字节")
```

### 多 yield 语句和状态维护

生成器函数中可以有多个 `yield` 语句，函数的状态（包括局部变量和执行位置）会在每次调用之间保持：

```python
def fibonacci(n):
    """生成斐波那契数列的前n个数"""
    a, b = 0, 1
    count = 0
    
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# 使用生成器生成斐波那契数列
for fib in fibonacci(10):
    print(fib, end=' ')  # 输出：0 1 1 2 3 5 8 13 21 34
```

## 高级迭代器技巧

### itertools 模块

`itertools` 模块提供了许多高效处理迭代器的函数：

```python
import itertools

# 无限迭代器
# count - 从起始值开始无限计数
for i in itertools.islice(itertools.count(10, 2), 5):
    print(i, end=' ')  # 输出：10 12 14 16 18

print("\n----")

# cycle - 无限循环某个可迭代对象
for i in itertools.islice(itertools.cycle('ABC'), 7):
    print(i, end=' ')  # 输出：A B C A B C A

print("\n----")

# repeat - 无限或有限次数重复某个元素
for i in itertools.repeat('Hello', 3):
    print(i, end=' ')  # 输出：Hello Hello Hello

print("\n----")

# 终止于最短输入序列的迭代器
# chain - 连接多个可迭代对象
for i in itertools.chain('ABC', [1, 2, 3]):
    print(i, end=' ')  # 输出：A B C 1 2 3

print("\n----")

# compress - 根据选择器筛选元素
for i in itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 0]):
    print(i, end=' ')  # 输出：A C E

print("\n----")

# 排列组合迭代器
# combinations - 生成指定长度的所有组合
for combo in itertools.combinations('ABC', 2):
    print(''.join(combo), end=' ')  # 输出：AB AC BC

print("\n----")

# permutations - 生成指定长度的所有排列
for perm in itertools.permutations('ABC', 2):
    print(''.join(perm), end=' ')  # 输出：AB AC BA BC CA CB

print("\n----")

# product - 生成多个可迭代对象的笛卡尔积
for prod in itertools.product('AB', [1, 2]):
    print(''.join(str(x) for x in prod), end=' ')  # 输出：A1 A2 B1 B2
```

### 迭代器链

可以使用内置的 `iter()` 函数和 `itertools` 模块创建复杂的迭代器链：

```python
import itertools

# 创建多个迭代器
iter1 = iter([1, 2, 3])
iter2 = iter([4, 5, 6])
iter3 = iter([7, 8, 9])

# 将多个迭代器链接起来
chained = itertools.chain(iter1, iter2, iter3)

# 使用链接的迭代器
for num in chained:
    print(num, end=' ')  # 输出：1 2 3 4 5 6 7 8 9
```

### 自定义迭代器的更高级用法

结合装饰器和类方法，可以创建更复杂的迭代器：

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def reset(self):
        """重置迭代器"""
        self.index = 0
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value
    
    def filter(self, predicate):
        """返回一个只包含满足条件的元素的新迭代器"""
        class FilteredIterator:
            def __init__(self, source, predicate):
                self.source = source
                self.predicate = predicate
            
            def __iter__(self):
                return self
            
            def __next__(self):
                while True:
                    value = next(self.source)
                    if self.predicate(value):
                        return value
        
        return FilteredIterator(iter(self), predicate)
    
    def map(self, func):
        """返回一个应用函数到每个元素的新迭代器"""
        class MappedIterator:
            def __init__(self, source, func):
                self.source = source
                self.func = func
            
            def __iter__(self):
                return self
            
            def __next__(self):
                value = next(self.source)
                return self.func(value)
        
        return MappedIterator(iter(self), func)

# 使用自定义迭代器
processor = DataProcessor([1, 2, 3, 4, 5, 6])

# 链式操作：过滤偶数并对结果求平方
result = processor.filter(lambda x: x % 2 == 0).map(lambda x: x**2)

for value in result:
    print(value, end=' ')  # 输出：4 16 36
```

## 高级生成器技巧

### send() 方法和双向通信

生成器不仅可以生成值，还可以接收值。`send()` 方法允许向生成器发送值，该值将成为 `yield` 表达式的结果：

```python
def echo_generator():
    """回声生成器 - 回显接收到的值"""
    print("生成器启动！")
    value = yield "准备好接收值了！"
    while True:
        print(f"收到: {value}")
        value = yield f"回声: {value}"

# 创建生成器对象
gen = echo_generator()

# 启动生成器（第一次必须使用 next() 或 gen.send(None)）
print(next(gen))  # 输出：生成器启动！ 准备好接收值了！

# 向生成器发送值并接收结果
print(gen.send("Hello"))  # 输出：收到: Hello 回声: Hello
print(gen.send("World"))  # 输出：收到: World 回声: World
```

### throw() 和 close() 方法

生成器还具有 `throw()` 方法（抛出异常）和 `close()` 方法（关闭生成器）：

```python
def catching_generator():
    """可以捕获异常的生成器"""
    while True:
        try:
            value = yield "等待输入或异常..."
            print(f"接收到值: {value}")
        except ValueError:
            print("捕获到 ValueError 异常！")
        except GeneratorExit:
            print("生成器被关闭！")
            break

# 创建并启动生成器
gen = catching_generator()
print(next(gen))  # 输出：等待输入或异常...

# 发送正常值
print(gen.send("正常数据"))  # 输出：接收到值: 正常数据 等待输入或异常...

# 抛出异常
print(gen.throw(ValueError))  # 输出：捕获到 ValueError 异常！ 等待输入或异常...

# 关闭生成器
gen.close()  # 输出：生成器被关闭！
```

### 子生成器和委托

在 Python 3.3 及以上版本中，`yield from` 语法允许一个生成器委托其部分操作给另一个生成器：

```python
def sub_generator():
    """子生成器 - 生成一些值并处理一些输入"""
    yield "子生成器就绪"
    value = yield "子生成器等待输入"
    yield f"子生成器收到: {value}"
    return "子生成器完成"  # 返回值将成为 yield from 表达式的值

def main_generator():
    """主生成器 - 委托部分工作给子生成器"""
    print("主生成器开始")
    result = yield from sub_generator()
    print(f"子生成器返回: {result}")
    yield "主生成器恢复"

# 使用主生成器
gen = main_generator()
print(next(gen))  # 输出：主生成器开始 子生成器就绪
print(next(gen))  # 输出：子生成器等待输入
print(gen.send("代理数据"))  # 输出：子生成器收到: 代理数据
print(next(gen))  # 输出：子生成器返回: 子生成器完成 主生成器恢复
```

### 协程和异步编程

生成器和 `yield from` 语法是 Python 异步编程的基础。在 Python 3.5 之前，协程是使用生成器实现的：

```python
def simple_coroutine():
    """简单的协程示例"""
    print("协程开始")
    x = yield
    print(f"收到 x = {x}")
    y = yield
    print(f"收到 y = {y}")
    return x + y

# 使用协程
coro = simple_coroutine()
next(coro)  # 启动协程，执行到第一个 yield
coro.send(10)  # 发送 x = 10
try:
    coro.send(20)  # 发送 y = 20，协程计算 x + y 并结束
except StopIteration as e:
    result = e.value  # 获取协程的返回值
    print(f"协程返回: {result}")  # 输出：协程返回: 30
```

在 Python 3.5 及以后的版本中，可以使用 `async/await` 语法实现更现代的协程：

```python
import asyncio

async def hello_world():
    """异步函数（协程）"""
    print("Hello")
    await asyncio.sleep(1)
    print("World")
    return "完成"

# 运行异步函数
async def main():
    result = await hello_world()
    print(f"结果: {result}")

# 在 Python 3.7+ 中执行
asyncio.run(main())
```

## 实际应用示例

### 无限数据流处理

生成器特别适合处理无限或非常大的数据流：

```python
def read_large_file(file_path, chunk_size=1024):
    """分块读取大文件的生成器"""
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def line_reader(file_path):
    """按行读取文件的生成器"""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# 示例用法
try:
    import os
    
    # 创建示例文件
    with open('example.txt', 'w') as f:
        f.write('\n'.join(f"这是第 {i} 行" for i in range(1, 1001)))
    
    # 使用生成器读取并处理文件
    for i, line in enumerate(line_reader('example.txt')):
        if i < 5:  # 只处理前5行
            print(line)
        else:
            break
    
    # 计算文件行数
    line_count = sum(1 for _ in line_reader('example.txt'))
    print(f"文件总行数: {line_count}")
    
    # 清理示例文件
    os.remove('example.txt')
except Exception as e:
    print(f"示例代码出错: {e}")
```

### 数据管道和转换

使用生成器可以创建高效的数据处理管道：

```python
def read_csv(file_path):
    """读取 CSV 文件的行"""
    with open(file_path, 'r') as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

def filter_data(records, predicate):
    """筛选符合条件的记录"""
    for record in records:
        if predicate(record):
            yield record

def transform_data(records, transformer):
    """转换记录"""
    for record in records:
        yield transformer(record)

def group_by(records, key_func):
    """按键函数对记录分组"""
    groups = {}
    for record in records:
        key = key_func(record)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    # 返回分组后的结果
    for key, group in groups.items():
        yield (key, group)

# 示例：数据处理管道
try:
    import os
    
    # 创建示例 CSV 文件
    with open('people.csv', 'w') as f:
        f.write('name,age,city\n')
        f.write('Alice,25,Beijing\n')
        f.write('Bob,30,Shanghai\n')
        f.write('Charlie,35,Beijing\n')
        f.write('David,28,Shanghai\n')
    
    # 数据处理管道
    people = read_csv('people.csv')
    adults = filter_data(people, lambda p: int(p['age']) >= 30)
    simplified = transform_data(adults, lambda p: {'name': p['name'], 'city': p['city']})
    
    print("30岁及以上的人:")
    for person in simplified:
        print(f"{person['name']} ({person['city']})")
    
    # 重新读取并按城市分组
    people = read_csv('people.csv')
    by_city = group_by(people, lambda p: p['city'])
    
    print("\n按城市分组:")
    for city, group in by_city:
        names = [p['name'] for p in group]
        print(f"{city}: {', '.join(names)}")
    
    # 清理示例文件
    os.remove('people.csv')
except Exception as e:
    print(f"示例代码出错: {e}")
```

### 内存优化计算

使用生成器可以大幅减少内存使用，尤其是在处理大型数据集时：

```python
def calculate_running_average():
    """计算运行平均值的生成器"""
    total = 0.0
    count = 0
    average = 0.0
    
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count

def memory_efficient_statistics(data_source):
    """计算大型数据集的统计信息，无需一次性加载全部数据"""
    # 初始化统计变量
    count = 0
    total = 0
    minimum = float('inf')
    maximum = float('-inf')
    
    # 使用运行平均值生成器
    avg_gen = calculate_running_average()
    next(avg_gen)  # 启动生成器
    
    # 处理每个数据项
    for value in data_source:
        count += 1
        total += value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
        average = avg_gen.send(value)
    
    return {
        'count': count,
        'total': total,
        'average': average,
        'minimum': minimum,
        'maximum': maximum
    }

# 示例：处理大量数据
def large_data_source(n):
    """模拟大型数据源"""
    import random
    for _ in range(n):
        yield random.randint(1, 1000)

# 计算统计信息
stats = memory_efficient_statistics(large_data_source(1000000))
print("统计信息:")
for key, value in stats.items():
    print(f"{key}: {value}")
```

## 最佳实践和性能考虑

### 迭代器和生成器的优缺点

**优点**：
- 内存效率高：一次只处理一个元素
- 延迟计算：只在需要时才生成值
- 无限序列：可以表示无限序列
- 代码易读性：简化了复杂的数据处理逻辑

**缺点**：
- 一次性消耗：迭代器一旦被消耗就不能重置
- 不支持索引：不能直接通过索引访问元素
- 无法获取长度：必须遍历整个迭代器才能知道长度
- 调试困难：难以查看迭代器中的所有元素

### 性能优化技巧

1. **避免在循环中创建生成器**：

```python
# 不好的做法：在循环中创建生成器
for i in range(100):
    data = (x for x in range(10))  # 每次循环创建新的生成器
    process(data)

# 好的做法：创建一个生成器工厂函数
def create_data():
    return (x for x in range(10))

for i in range(100):
    process(create_data())
```

2. **利用 itertools 函数优化性能**：

```python
import itertools

# 不好的做法：自己实现循环逻辑
def manually_combined():
    result = []
    for a in 'ABC':
        for b in [1, 2, 3]:
            result.append((a, b))
    return result

# 好的做法：使用 itertools
def efficiently_combined():
    return itertools.product('ABC', [1, 2, 3])
```

3. **结合生成器表达式和生成器函数**：

```python
def process_large_file(file_path):
    # 使用生成器读取文件
    lines = (line.strip() for line in open(file_path))
    
    # 过滤空行
    non_empty_lines = (line for line in lines if line)
    
    # 解析并处理
    for line in non_empty_lines:
        # 处理每行...
        yield processed_result
```

### 迭代器和生成器的使用建议

1. **何时使用列表推导式和生成器表达式**：
   - 如果需要多次遍历数据，使用列表推导式
   - 如果只需要遍历一次或数据量很大，使用生成器表达式

2. **重置迭代器**：如果需要多次遍历同一个数据集，可以：
   - 将迭代器转换为列表（如果数据量不大）
   - 创建一个迭代器工厂函数，每次需要时返回新的迭代器
   - 实现一个带有 `reset()` 方法的自定义迭代器

3. **警惕迭代器的消耗**：
   - 迭代器消耗后不能重复使用
   - 在多处使用同一迭代器时要小心，先将其转换为列表或创建多个独立迭代器

## 下一步

现在您已经了解了 Python 的迭代器和生成器，接下来可以探索 [Python 装饰器](/intermediate/decorators)，学习如何使用这一强大功能来扩展和修改函数和类的行为。 