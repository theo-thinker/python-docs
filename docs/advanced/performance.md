# Python 性能优化

Python 因其简洁易用的语法和丰富的生态系统而深受开发者喜爱，但在性能要求较高的场景下，可能需要一些优化技巧。本章将介绍常见的 Python 性能优化方法和工具。

## 性能优化的基本原则

在开始优化之前，请牢记以下原则：

1. **先测量，后优化**：在优化前确定性能瓶颈所在，避免过早优化
2. **重视算法和数据结构**：选择合适的算法和数据结构往往比代码微优化更重要
3. **权衡可读性和性能**：不要为了微小的性能提升牺牲代码可读性和可维护性
4. **了解 Python 的优化限制**：理解 GIL 等 Python 特性带来的限制

## 性能分析工具

### 时间测量

最基本的性能分析方法是使用 `time` 模块测量代码运行时间：

```python
import time

def measure_time(func):
    """简单的函数运行时间测量装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 运行时间: {end_time - start_time:.6f} 秒")
        return result
    return wrapper

@measure_time
def slow_function():
    """一个耗时的函数"""
    time.sleep(1)
    
slow_function()  # 输出: slow_function 运行时间: 1.000xxx 秒
```

### timeit 模块

`timeit` 模块是 Python 标准库提供的用于精确测量小段代码执行时间的工具：

```python
import timeit

# 测量列表推导式与循环的性能差异
list_comp_time = timeit.timeit(
    '[x**2 for x in range(1000)]',
    number=10000
)

loop_time = timeit.timeit(
    '''
result = []
for x in range(1000):
    result.append(x**2)
''',
    number=10000
)

print(f"列表推导式: {list_comp_time:.6f} 秒")
print(f"循环: {loop_time:.6f} 秒")
print(f"列表推导式比循环快 {loop_time / list_comp_time:.2f} 倍")
```

### cProfile 和 profile 模块

这些模块可以提供更详细的性能数据，包括函数调用次数、每次调用的时间等：

```python
import cProfile
import pstats
import io

def profile_func(func, *args, **kwargs):
    """对函数进行性能分析"""
    pr = cProfile.Profile()
    pr.enable()
    
    result = func(*args, **kwargs)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # 打印前 10 个结果
    print(s.getvalue())
    
    return result

def fibonacci_recursive(n):
    """递归计算斐波那契数列（效率低）"""
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 性能分析递归斐波那契函数
profile_func(fibonacci_recursive, 30)
```

### line_profiler：逐行性能分析

`line_profiler` 可以提供每行代码的执行时间：

```python
# 需要先安装: pip install line_profiler

# 在实际代码中，使用 @profile 装饰器标记要分析的函数
# 然后使用命令行工具 kernprof 运行: kernprof -l script.py

@profile  # 只有在 kernprof 中运行时才有效
def process_data(data):
    result = []
    for item in data:
        # 一些计算密集型操作
        processed = item ** 2
        result.append(processed)
    
    # 再处理一遍结果
    total = sum(result)
    return total / len(result) if result else 0

# 运行函数
data = list(range(10000))
process_data(data)
```

### memory_profiler：内存分析

`memory_profiler` 用于分析代码的内存使用情况：

```python
# 需要先安装: pip install memory_profiler

# 同样，使用 @profile 装饰器标记要分析的函数
# 然后用 python -m memory_profiler script.py 运行

@profile
def memory_intensive_function():
    # 创建一个大列表
    big_list = [i for i in range(1000000)]
    
    # 处理数据
    result = sum(big_list)
    
    # 创建另一个列表
    another_list = [result] * 1000000
    
    return len(another_list)

# 运行函数
memory_intensive_function()
```

## 代码优化技巧

### 使用合适的数据结构

选择正确的数据结构对性能至关重要：

```python
# 列表操作时间复杂度
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # O(1)，常数时间
5 in my_list  # O(n)，线性时间

# 字典操作时间复杂度
my_dict = {i: i**2 for i in range(1, 6)}
my_dict[6] = 36  # O(1)，常数时间
5 in my_dict  # O(1)，常数时间

# 集合操作时间复杂度
my_set = {1, 2, 3, 4, 5}
my_set.add(6)  # O(1)，常数时间
5 in my_set  # O(1)，常数时间

# 根据场景选择正确的数据结构
def find_duplicates_list(items):
    duplicates = []
    seen = []
    for item in items:
        if item in seen and item not in duplicates:  # O(n) 操作
            duplicates.append(item)
        seen.append(item)
    return duplicates

def find_duplicates_set(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:  # O(1) 操作
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

# 测量时间差异
import random
data = [random.randint(0, 1000) for _ in range(10000)]

# 使用 timeit 比较两个函数的性能
print(timeit.timeit(lambda: find_duplicates_list(data), number=10))
print(timeit.timeit(lambda: find_duplicates_set(data), number=10))
```

### 列表推导式和生成器表达式

列表推导式通常比等效的 for 循环更快，而生成器表达式更节省内存：

```python
# 列表推导式
squares_list = [x**2 for x in range(1000)]  # 立即创建完整列表

# 生成器表达式
squares_gen = (x**2 for x in range(1000))  # 创建生成器，按需计算值

# 使用生成器处理大数据
def process_large_file(filename):
    # 改进前: 一次读取整个文件到内存
    # with open(filename) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         process_line(line)
    
    # 改进后: 使用生成器逐行处理
    with open(filename) as f:
        for line in f:  # 文件对象是可迭代的，逐行产生数据
            yield line.strip()

# 比较内存使用
import sys
list_size = sys.getsizeof([x for x in range(1000000)])
gen_size = sys.getsizeof((x for x in range(1000000)))
print(f"列表占用内存: {list_size} 字节")
print(f"生成器占用内存: {gen_size} 字节")
print(f"内存比例: {list_size / gen_size:.2f} 倍")
```

### 局部变量优化

局部变量访问比全局变量和属性访问更快：

```python
import math

# 未优化版本
def calculate_distances1(points):
    distances = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            # 重复访问全局变量和属性
            distances.append(math.sqrt((points[i][0] - points[j][0])**2 + 
                                      (points[i][1] - points[j][1])**2))
    return distances

# 优化版本
def calculate_distances2(points):
    # 局部变量引用
    local_sqrt = math.sqrt
    distances = []
    point_count = len(points)
    
    for i in range(point_count - 1):
        point_i = points[i]
        point_i_0 = point_i[0]
        point_i_1 = point_i[1]
        
        for j in range(i + 1, point_count):
            point_j = points[j]
            dx = point_i_0 - point_j[0]
            dy = point_i_1 - point_j[1]
            distances.append(local_sqrt(dx*dx + dy*dy))
            
    return distances

# 测试性能差异
points = [(i, i) for i in range(100)]
print(timeit.timeit(lambda: calculate_distances1(points), number=10))
print(timeit.timeit(lambda: calculate_distances2(points), number=10))
```

### 减少函数调用开销

函数调用在 Python 中有一定开销，对于频繁调用的简单函数，可以考虑内联代码：

```python
# 使用函数
def is_prime_func(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_primes_with_func(limit):
    count = 0
    for num in range(2, limit):
        if is_prime_func(num):  # 函数调用
            count += 1
    return count

# 内联代码
def count_primes_inline(limit):
    count = 0
    for num in range(2, limit):
        # 内联判断素数的代码
        is_prime = True
        if num < 2:
            is_prime = False
        else:
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
        if is_prime:
            count += 1
    return count

# 比较性能
print(timeit.timeit(lambda: count_primes_with_func(1000), number=10))
print(timeit.timeit(lambda: count_primes_inline(1000), number=10))
```

### 使用内置函数和模块

Python 的内置函数通常用 C 语言实现，性能更好：

```python
import operator

numbers = list(range(10000))

# 手动计算总和
def manual_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# 使用内置 sum 函数
def builtin_sum(numbers):
    return sum(numbers)

# 手动找最大值
def manual_max(numbers):
    max_val = float('-inf')
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

# 使用内置 max 函数
def builtin_max(numbers):
    return max(numbers)

# 比较性能
print("求和:")
print(timeit.timeit(lambda: manual_sum(numbers), number=1000))
print(timeit.timeit(lambda: builtin_sum(numbers), number=1000))

print("求最大值:")
print(timeit.timeit(lambda: manual_max(numbers), number=1000))
print(timeit.timeit(lambda: builtin_max(numbers), number=1000))
```

### 减少循环中的计算

将循环不变量移到循环外，可以显著提高性能：

```python
import math

def calculate_area1(radius_list):
    areas = []
    for r in radius_list:
        areas.append(math.pi * r * r)  # 每次循环都访问 math.pi
    return areas

def calculate_area2(radius_list):
    areas = []
    pi = math.pi  # 循环外提取常量
    for r in radius_list:
        areas.append(pi * r * r)
    return areas

# 更进一步优化
def calculate_area3(radius_list):
    pi = math.pi
    # 使用列表推导式
    return [pi * r * r for r in radius_list]

# 比较性能
radius_list = list(range(1, 10001))
print(timeit.timeit(lambda: calculate_area1(radius_list), number=100))
print(timeit.timeit(lambda: calculate_area2(radius_list), number=100))
print(timeit.timeit(lambda: calculate_area3(radius_list), number=100))
```

### 字符串优化

字符串连接和格式化有不同的性能特性：

```python
def str_concat_with_plus(n):
    result = ""
    for i in range(n):
        result = result + str(i) + ", "  # 低效
    return result

def str_concat_with_join(n):
    return ", ".join(str(i) for i in range(n))  # 高效

def str_format_with_percent(data):
    return "%s is %d years old and %.2f tall" % (data['name'], data['age'], data['height'])

def str_format_with_method(data):
    return "{} is {} years old and {:.2f} tall".format(data['name'], data['age'], data['height'])

def str_format_with_fstring(data):
    return f"{data['name']} is {data['age']} years old and {data['height']:.2f} tall"

# 比较字符串连接性能
print("字符串连接:")
print(timeit.timeit(lambda: str_concat_with_plus(1000), number=100))
print(timeit.timeit(lambda: str_concat_with_join(1000), number=100))

# 比较格式化性能
data = {'name': '张三', 'age': 30, 'height': 1.75}
print("字符串格式化:")
print(timeit.timeit(lambda: str_format_with_percent(data), number=100000))
print(timeit.timeit(lambda: str_format_with_method(data), number=100000))
print(timeit.timeit(lambda: str_format_with_fstring(data), number=100000))
```

## 代码编译和原生扩展

### 使用 PyPy

PyPy 是 Python 的一种替代解释器，使用 JIT 编译技术提供更好的性能：

```
# 比较 CPython 和 PyPy 性能（命令行操作）

# 使用 CPython 运行
python benchmark_script.py

# 使用 PyPy 运行
pypy benchmark_script.py
```

### 使用 Cython

Cython 将 Python 代码转换为 C 代码，然后编译成扩展模块：

```python
# 常规 Python 代码 (保存为 example.py)
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

# Cython 版本 (保存为 example_cy.pyx)
"""
def fibonacci_cy(int n):
    cdef int i
    cdef long a = 0, b = 1
    for i in range(n):
        a, b = b, a + b
    return a
"""

# 编译 Cython 代码的 setup.py
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("example_cy.pyx")
)
"""

# 编译: python setup.py build_ext --inplace
# 然后可以导入并使用优化后的函数:
"""
import example
import example_cy
import timeit

n = 1000000
print(timeit.timeit(lambda: example.fibonacci(n), number=10))
print(timeit.timeit(lambda: example_cy.fibonacci_cy(n), number=10))
"""
```

### Numba 即时编译

Numba 使用 LLVM 为数值计算提供即时编译：

```python
# 需要安装: pip install numba

import numpy as np
from numba import jit
import timeit

# 常规 Python 函数
def sum_of_squares(arr):
    result = 0.0
    for i in range(arr.shape[0]):
        result += arr[i] * arr[i]
    return result

# Numba 优化版本
@jit(nopython=True)
def sum_of_squares_numba(arr):
    result = 0.0
    for i in range(arr.shape[0]):
        result += arr[i] * arr[i]
    return result

# 创建大型数组
arr = np.random.random(10000000)

# 预热 JIT 编译器
sum_of_squares_numba(arr[:100])

# 比较性能
print("Python:", timeit.timeit(lambda: sum_of_squares(arr), number=10))
print("Numba:", timeit.timeit(lambda: sum_of_squares_numba(arr), number=10))
```

### 使用 C 扩展

对于性能要求极高的部分，可以用 C 语言编写扩展模块：

```c
// example.c - 一个简单的 C 扩展示例
/*
#include <Python.h>

static PyObject* fibonacci(PyObject* self, PyObject* args) {
    long n;
    if (!PyArg_ParseTuple(args, "l", &n))
        return NULL;
    
    long a = 0, b = 1, temp;
    for (long i = 0; i < n; i++) {
        temp = a;
        a = b;
        b = temp + b;
    }
    
    return PyLong_FromLong(a);
}

static PyMethodDef ExampleMethods[] = {
    {"fibonacci", fibonacci, METH_VARARGS, "Calculate the nth Fibonacci number."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    "example",
    NULL,
    -1,
    ExampleMethods
};

PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&examplemodule);
}
*/

// setup.py - 编译 C 扩展
/*
from setuptools import setup, Extension

module = Extension('example', sources=['example.c'])

setup(
    name='example',
    version='1.0',
    ext_modules=[module]
)
*/

// 编译: python setup.py build_ext --inplace
```

## 并行和并发

### 多线程

使用 `threading` 模块处理 I/O 密集型任务：

```python
import threading
import time
import requests

def download_url(url):
    """下载URL内容"""
    response = requests.get(url)
    return response.text

def sequential_downloads(urls):
    """顺序下载多个URL"""
    start = time.time()
    for url in urls:
        download_url(url)
    print(f"顺序下载耗时: {time.time() - start:.2f} 秒")

def threaded_downloads(urls):
    """多线程下载多个URL"""
    start = time.time()
    threads = []
    
    # 创建线程
    for url in urls:
        thread = threading.Thread(target=download_url, args=(url,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print(f"多线程下载耗时: {time.time() - start:.2f} 秒")

# 测试URL列表
urls = [
    "https://www.python.org",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.google.com",
    "https://www.wikipedia.org"
]

# 比较顺序和多线程下载
sequential_downloads(urls)
threaded_downloads(urls)
```

### 多进程

使用 `multiprocessing` 模块处理 CPU 密集型任务：

```python
import multiprocessing
import time

def cpu_intensive_task(n):
    """CPU密集型计算（计算第n个斐波那契数）"""
    if n <= 1:
        return n
    return cpu_intensive_task(n-1) + cpu_intensive_task(n-2)

def sequential_processing(numbers):
    """顺序处理多个任务"""
    start = time.time()
    results = []
    for n in numbers:
        results.append(cpu_intensive_task(n))
    print(f"顺序处理耗时: {time.time() - start:.2f} 秒")
    return results

def parallel_processing(numbers):
    """多进程并行处理多个任务"""
    start = time.time()
    
    # 创建进程池
    with multiprocessing.Pool() as pool:
        # 使用进程池映射任务
        results = pool.map(cpu_intensive_task, numbers)
    
    print(f"并行处理耗时: {time.time() - start:.2f} 秒")
    return results

# 测试数据
numbers = [30, 31, 32, 33, 34]

# 比较顺序和并行处理
sequential_results = sequential_processing(numbers)
parallel_results = parallel_processing(numbers)

# 验证结果一致性
print(f"结果一致: {sequential_results == parallel_results}")
```

### 异步 IO

使用 `asyncio` 模块高效处理 I/O 任务：

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    """异步获取URL内容"""
    async with session.get(url) as response:
        return await response.text()

async def download_all(urls):
    """并发下载多个URL"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def async_downloads(urls):
    """使用异步IO下载多个URL"""
    start = time.time()
    
    # 获取事件循环
    loop = asyncio.get_event_loop()
    
    # 运行异步任务
    results = loop.run_until_complete(download_all(urls))
    
    print(f"异步下载耗时: {time.time() - start:.2f} 秒")
    return results

# 测试URL列表
urls = [
    "https://www.python.org",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.google.com",
    "https://www.wikipedia.org"
]

# 运行异步下载测试
# async_downloads(urls)
```

## 性能优化案例研究

### 代码重构：从慢到快

我们将通过一个实际例子展示性能优化的完整过程：

```python
import time
import string
import random

# 初始版本：低效的文本分析函数
def analyze_text_v1(text):
    """分析文本，返回字符频率和单词频率"""
    # 字符频率
    char_freq = {}
    for char in text:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    # 单词频率
    words = text.split()
    word_freq = {}
    for word in words:
        # 清理标点符号
        cleaned_word = ""
        for char in word:
            if char not in string.punctuation:
                cleaned_word += char.lower()
        
        if cleaned_word:
            if cleaned_word in word_freq:
                word_freq[cleaned_word] += 1
            else:
                word_freq[cleaned_word] = 1
    
    return char_freq, word_freq

# 第一次优化：使用Counter和更好的写法
from collections import Counter

def analyze_text_v2(text):
    """使用Counter优化的文本分析函数"""
    # 字符频率
    char_freq = Counter(text)
    
    # 单词频率
    # 清理并分割文本
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    words = clean_text.split()
    word_freq = Counter(words)
    
    return char_freq, word_freq

# 第二次优化：预处理和列表推导式
def analyze_text_v3(text):
    """进一步优化的文本分析函数"""
    # 字符频率
    char_freq = Counter(text)
    
    # 预处理标点符号转换表
    translator = str.maketrans('', '', string.punctuation)
    
    # 使用列表推导式提取单词
    word_freq = Counter(
        text.translate(translator).lower().split()
    )
    
    return char_freq, word_freq

# 生成测试数据
def generate_test_text(size=1000000):
    """生成指定大小的随机文本用于测试"""
    words = ['python', 'programming', 'performance', 'optimization', 
             'algorithm', 'language', 'developer', 'code', 'function', 'class']
    
    return ' '.join(random.choice(words) for _ in range(size // 8))

# 测试性能
test_text = generate_test_text()

for version, func in [
    ("v1 (初始版本)", analyze_text_v1),
    ("v2 (使用Counter)", analyze_text_v2),
    ("v3 (预处理优化)", analyze_text_v3)
]:
    start = time.time()
    char_freq, word_freq = func(test_text)
    end = time.time()
    print(f"{version} 耗时: {end - start:.4f} 秒")
    print(f"  识别出 {len(char_freq)} 个不同字符")
    print(f"  识别出 {len(word_freq)} 个不同单词")
```

### NumPy 和向量化

数值计算使用 NumPy 可大幅提升性能：

```python
import numpy as np
import timeit

# 普通 Python 计算
def python_vector_dot(v1, v2):
    """计算两个向量的点积"""
    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

# NumPy 向量化计算
def numpy_vector_dot(v1, v2):
    """使用 NumPy 计算点积"""
    return np.dot(v1, v2)

# 测试数据
size = 1000000
v1 = [random.random() for _ in range(size)]
v2 = [random.random() for _ in range(size)]
v1_np = np.array(v1)
v2_np = np.array(v2)

# 比较性能
print("计算100万元素向量点积:")
python_time = timeit.timeit(lambda: python_vector_dot(v1, v2), number=1)
numpy_time = timeit.timeit(lambda: numpy_vector_dot(v1_np, v2_np), number=1)

print(f"Python: {python_time:.6f} 秒")
print(f"NumPy: {numpy_time:.6f} 秒")
print(f"NumPy 比 Python 快 {python_time / numpy_time:.2f} 倍")
```

## 优化实践指南

### 优化工作流程

性能优化应该遵循以下工作流程：

1. **明确目标**：确定需要满足的性能需求
2. **分析瓶颈**：使用性能分析工具找出瓶颈
3. **逐步优化**：从影响最大的瓶颈开始，进行渐进式优化
4. **验证结果**：每次优化后测量性能提升
5. **平衡取舍**：权衡性能和代码可维护性

### 常见优化陷阱

在优化过程中应避免以下陷阱：

1. **过早优化**：在没有确定瓶颈前就进行优化
2. **基准测试不当**：测试环境与生产环境差异过大
3. **只关注微优化**：忽略算法和架构层面的改进
4. **牺牲可读性**：为了微小的性能提升使代码难以理解
5. **过于依赖特定版本**：优化可能在 Python 版本更新后失效

### 持续优化策略

对于长期项目，持续优化的策略包括：

1. **建立性能基准**：定期运行性能测试并记录结果
2. **自动化性能测试**：将性能测试纳入 CI/CD 流程
3. **设置性能预算**：为关键操作设定最大允许执行时间
4. **性能监控**：在生产环境监控实际性能
5. **定期审查**：定期回顾代码库，寻找新的优化机会

## Python 3.13 性能特性

Python 3.13 引入了一些新的性能改进：

1. **更快的字典和集合**：内部实现优化
2. **C API 改进**：更高效的 C 扩展接口
3. **内存分配优化**：更高效的内存使用
4. **JIT 编译支持进展**：为未来的 JIT 编译器做准备
5. **类型注解优化**：运行时类型检查性能改进

例如，使用 Python 3.13 的新特性：

```python
# Python 3.13 的新 F-string 语法优化
number = 123456789
formatted = f"{number=:,}"  # Python 3.13 允许在 = 后直接使用格式说明符

# Python 3.13 类型参数语法提供更好的静态类型检查
def identity[T](x: T) -> T:
    return x
```

## 下一步

现在您已经了解了 Python 性能优化的关键技术，接下来可以探索特定领域的优化，例如网络应用性能、数据库交互、机器学习等。请查看 [Python 设计模式](/advanced/design-patterns) 了解如何设计高效且可维护的代码结构。 