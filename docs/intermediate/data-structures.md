# Python 数据结构

Python 提供了多种内置数据结构，帮助开发者高效地存储和管理数据。除了基本的列表和字典外，Python 还提供了更多专用的数据结构，以及通过 `collections` 模块支持的高级数据结构。

## 列表（List）进阶

列表是 Python 中最常用的数据结构之一，下面介绍一些进阶用法：

### 列表推导式

列表推导式提供了一种简洁的方式创建列表：

```python
# 基本列表推导式
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# 多层列表推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 创建二维矩阵
matrix = [[i+j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

### 列表切片进阶

```python
# 基本切片：[start:stop:step]
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[2:8:2])  # [2, 4, 6]

# 反向切片
print(numbers[::-1])  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(numbers[7:2:-1])  # [7, 6, 5, 4, 3]

# 使用切片复制列表
copy_of_numbers = numbers[:]
```

### 列表操作技巧

```python
# 合并列表
list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged = list1 + list2  # [1, 2, 3, 4, 5, 6]

# 使用 * 重复列表
repeated = list1 * 3  # [1, 2, 3, 1, 2, 3, 1, 2, 3]

# 使用 zip 并行处理多个列表
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# 使用 enumerate 获取索引和值
for i, name in enumerate(names):
    print(f"Index {i}: {name}")

# 使用 any 和 all
print(any(x > 5 for x in ages))  # True
print(all(x > 30 for x in ages))  # False
```

## 字典（Dictionary）进阶

### 字典推导式

```python
# 创建平方映射
squares = {x: x**2 for x in range(6)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 从序列创建字典
names = ['Alice', 'Bob', 'Charlie']
name_lengths = {name: len(name) for name in names}
print(name_lengths)  # {'Alice': 5, 'Bob': 3, 'Charlie': 7}

# 带条件的字典推导式
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
print(even_squares)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}
```

### 字典操作技巧

```python
# 合并字典（Python 3.9+）
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = dict1 | dict2  # {'a': 1, 'b': 3, 'c': 4}

# 更新字典（Python 3.9+）
dict1 |= dict2  # 现在 dict1 是 {'a': 1, 'b': 3, 'c': 4}

# 使用 get 方法安全地获取值
user = {'name': 'Alice', 'age': 25}
print(user.get('email', 'Not provided'))  # 'Not provided'

# 使用 setdefault 设置默认值
result = user.setdefault('email', 'alice@example.com')
print(result)  # 'alice@example.com'
print(user)    # {'name': 'Alice', 'age': 25, 'email': 'alice@example.com'}

# 使用 items, keys, values 方法
for key, value in user.items():
    print(f"{key}: {value}")

# 使用 pop 和 popitem
age = user.pop('age')  # 移除并返回 'age' 的值
last_item = user.popitem()  # 移除并返回最后添加的键值对
```

## 集合（Set）进阶

### 集合推导式

```python
# 从列表创建集合
numbers = [1, 2, 2, 3, 4, 4, 5]
unique_numbers = {x for x in numbers}
print(unique_numbers)  # {1, 2, 3, 4, 5}

# 带条件的集合推导式
even_numbers = {x for x in range(10) if x % 2 == 0}
print(even_numbers)  # {0, 2, 4, 6, 8}
```

### 集合操作

```python
# 集合运算
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# 并集
union = set1 | set2  # {1, 2, 3, 4, 5, 6, 7, 8}
union_alt = set1.union(set2)  # 同上

# 交集
intersection = set1 & set2  # {4, 5}
intersection_alt = set1.intersection(set2)  # 同上

# 差集
difference = set1 - set2  # {1, 2, 3}
difference_alt = set1.difference(set2)  # 同上

# 对称差集
symmetric_diff = set1 ^ set2  # {1, 2, 3, 6, 7, 8}
symmetric_diff_alt = set1.symmetric_difference(set2)  # 同上

# 检查子集和超集
print(set1.issubset({1, 2, 3, 4, 5, 6}))  # True
print(set1.issuperset({1, 2, 3}))  # True
```

## 元组（Tuple）进阶

### 命名元组

```python
from collections import namedtuple

# 创建命名元组类型
Person = namedtuple('Person', ['name', 'age', 'city'])

# 创建命名元组实例
alice = Person('Alice', 25, 'Beijing')
bob = Person(name='Bob', age=30, city='Shanghai')

# 访问属性
print(alice.name)    # 'Alice'
print(alice[0])      # 'Alice'（也支持索引访问）
print(bob.city)      # 'Shanghai'

# 解包
name, age, city = alice
print(name, age, city)  # Alice 25 Beijing

# 转换为字典
alice_dict = alice._asdict()
print(alice_dict)  # {'name': 'Alice', 'age': 25, 'city': 'Beijing'}

# 创建新实例，只更改部分字段
alice_older = alice._replace(age=26)
print(alice_older)  # Person(name='Alice', age=26, city='Beijing')
```

## collections 模块中的数据结构

### defaultdict

```python
from collections import defaultdict

# 创建带默认值的字典
int_dict = defaultdict(int)  # 默认值为 0
int_dict['a'] += 1
int_dict['b'] += 2
print(int_dict)  # defaultdict(<class 'int'>, {'a': 1, 'b': 2})

# 用于计数
words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
counter = defaultdict(int)
for word in words:
    counter[word] += 1
print(counter)  # defaultdict(<class 'int'>, {'apple': 3, 'banana': 2, 'orange': 1})

# 用于分组
animals = [('dog', 'Fido'), ('cat', 'Felix'), ('dog', 'Buddy'), ('fish', 'Nemo')]
pets_by_type = defaultdict(list)
for animal_type, name in animals:
    pets_by_type[animal_type].append(name)
print(pets_by_type)  # defaultdict(<class 'list'>, {'dog': ['Fido', 'Buddy'], 'cat': ['Felix'], 'fish': ['Nemo']})
```

### OrderedDict

在 Python 3.7 以后，常规字典已保持插入顺序，但 OrderedDict 仍有一些特殊用途：

```python
from collections import OrderedDict

# 创建有序字典
ordered = OrderedDict([('first', 1), ('second', 2), ('third', 3)])

# 移动项到末尾
ordered.move_to_end('first')
print(ordered)  # OrderedDict([('second', 2), ('third', 3), ('first', 1)])

# 反向迭代
for key in reversed(ordered):
    print(key, ordered[key])

# 比较顺序
dict1 = OrderedDict([('a', 1), ('b', 2)])
dict2 = OrderedDict([('b', 2), ('a', 1)])
print(dict1 == dict2)  # False（顺序不同）

regular_dict1 = {'a': 1, 'b': 2}
regular_dict2 = {'b': 2, 'a': 1}
print(regular_dict1 == regular_dict2)  # True（普通字典只比较内容）
```

### Counter

```python
from collections import Counter

# 计数
colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
color_counts = Counter(colors)
print(color_counts)  # Counter({'blue': 3, 'red': 2, 'green': 1})

# 常见操作
print(color_counts['red'])    # 2
print(color_counts['yellow'])  # 0（不存在的元素返回0）

# 最常见的元素
print(color_counts.most_common(2))  # [('blue', 3), ('red', 2)]

# 更新计数
color_counts.update(['red', 'yellow'])
print(color_counts)  # Counter({'blue': 3, 'red': 3, 'green': 1, 'yellow': 1})

# 减去计数
color_counts.subtract(['red', 'red', 'blue'])
print(color_counts)  # Counter({'blue': 2, 'red': 1, 'green': 1, 'yellow': 1})

# Counter 算术
counter1 = Counter(['a', 'b', 'a', 'c'])
counter2 = Counter(['a', 'd', 'e'])
print(counter1 + counter2)  # Counter({'a': 3, 'b': 1, 'c': 1, 'd': 1, 'e': 1})
print(counter1 - counter2)  # Counter({'b': 1, 'c': 1, 'a': 1})
```

### deque（双端队列）

```python
from collections import deque

# 创建双端队列
dq = deque(['a', 'b', 'c'])
print(dq)  # deque(['a', 'b', 'c'])

# 添加元素
dq.append('d')        # 在右侧添加
dq.appendleft('z')    # 在左侧添加
print(dq)  # deque(['z', 'a', 'b', 'c', 'd'])

# 移除元素
right_elem = dq.pop()        # 从右侧移除
left_elem = dq.popleft()     # 从左侧移除
print(right_elem, left_elem)  # d z
print(dq)  # deque(['a', 'b', 'c'])

# 旋转
dq.rotate(1)    # 向右旋转1步
print(dq)       # deque(['c', 'a', 'b'])
dq.rotate(-2)   # 向左旋转2步
print(dq)       # deque(['b', 'c', 'a'])

# 限制大小的双端队列
limited_dq = deque(maxlen=3)
for i in range(5):
    limited_dq.append(i)
    print(limited_dq)  # 最终结果: deque([2, 3, 4], maxlen=3)
```

### ChainMap

```python
from collections import ChainMap

# 创建多个字典的视图
defaults = {'theme': 'Default', 'language': 'English', 'id': 1}
user_settings = {'theme': 'Dark'}

# 将字典链接在一起，优先级从左到右
settings = ChainMap(user_settings, defaults)

print(settings['theme'])     # 'Dark'（从 user_settings 获取）
print(settings['language'])  # 'English'（从 defaults 获取）

# 更新映射
user_settings['language'] = 'Chinese'
print(settings['language'])  # 'Chinese'（现在从 user_settings 获取）

# 添加新的映射
env_settings = {'theme': 'Light', 'id': 2}
settings = ChainMap(env_settings, user_settings, defaults)
print(settings['theme'])  # 'Light'（从 env_settings 获取）
print(settings['id'])     # 2（从 env_settings 获取）
```

## 高级数据结构

### 堆（优先队列）

```python
import heapq

# 创建堆
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(numbers)  # 将列表转换为堆
print(numbers)  # [1, 1, 2, 3, 5, 9, 4, 6]（堆的内部表示）

# 添加元素
heapq.heappush(numbers, 0)
print(numbers)  # [0, 1, 2, 3, 1, 9, 4, 6, 5]

# 移除最小元素
smallest = heapq.heappop(numbers)
print(smallest)  # 0
print(numbers)   # [1, 1, 2, 3, 5, 9, 4, 6]

# 获取最小的 n 个元素
smallest_3 = heapq.nsmallest(3, numbers)
print(smallest_3)  # [1, 1, 2]

# 获取最大的 n 个元素
largest_3 = heapq.nlargest(3, numbers)
print(largest_3)  # [9, 6, 5]

# 处理复杂对象
people = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 20},
    {'name': 'Charlie', 'age': 30}
]
youngest = heapq.nsmallest(2, people, key=lambda x: x['age'])
print(youngest)  # [{'name': 'Bob', 'age': 20}, {'name': 'Alice', 'age': 25}]
```

### bisect 模块（二分查找）

```python
import bisect

# 有序列表
sorted_numbers = [1, 3, 5, 7, 9]

# 查找插入点
position = bisect.bisect(sorted_numbers, 4)
print(position)  # 2（4应该插入的位置）

# 插入元素并保持有序
bisect.insort(sorted_numbers, 4)
print(sorted_numbers)  # [1, 3, 4, 5, 7, 9]

# 左侧插入点（相等时插入到左侧）
position = bisect.bisect_left(sorted_numbers, 4)
print(position)  # 2

# 右侧插入点（相等时插入到右侧）
position = bisect.bisect_right(sorted_numbers, 4)
print(position)  # 3

# 查找范围
def find_range(numbers, x):
    """查找数值在有序列表中的范围"""
    i = bisect.bisect_left(numbers, x)
    j = bisect.bisect_right(numbers, x)
    return i, j

start, end = find_range(sorted_numbers, 4)
print(f"值4在位置{start}到{end-1}之间出现")  # 值4在位置2到2之间出现
```

### array 模块

```python
import array

# 创建类型化数组
int_array = array.array('i', [1, 2, 3, 4, 5])  # 'i'表示有符号整数
float_array = array.array('f', [1.1, 2.2, 3.3])  # 'f'表示浮点数

# 基本操作
int_array.append(6)
int_array.extend([7, 8, 9])
int_array.pop()  # 移除并返回最后一个元素
print(int_array)  # array('i', [1, 2, 3, 4, 5, 6, 7, 8])

# 转换为字节和从字节转换
byte_data = int_array.tobytes()
new_array = array.array('i')
new_array.frombytes(byte_data)
print(new_array)  # array('i', [1, 2, 3, 4, 5, 6, 7, 8])

# 数组类型码
"""
'b': 有符号字符, 'B': 无符号字符
'h': 有符号短整数, 'H': 无符号短整数
'i': 有符号整数, 'I': 无符号整数
'l': 有符号长整数, 'L': 无符号长整数
'q': 有符号长长整数, 'Q': 无符号长长整数
'f': 浮点数, 'd': 双精度浮点数
"""
```

## 实际应用示例

### LRU缓存实现

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return -1
        # 将访问的元素移到末尾（最近使用的位置）
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # 如果键已存在，先移除它（后面会重新添加到末尾）
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # 如果超出容量，移除最不常用的项（字典头部）
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# 使用示例
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))       # 返回 1
cache.put(3, 3)           # 淘汰 key 2
print(cache.get(2))       # 返回 -1 (未找到)
cache.put(4, 4)           # 淘汰 key 1
print(cache.get(1))       # 返回 -1 (未找到)
print(cache.get(3))       # 返回 3
print(cache.get(4))       # 返回 4
```

### 单词频率统计

```python
from collections import Counter
import re

def word_frequency(text):
    """统计文本中单词出现的频率"""
    # 将文本转换为小写并提取单词
    words = re.findall(r'\b\w+\b', text.lower())
    # 统计频率
    return Counter(words)

# 示例文本
text = """
Python is a programming language that lets you work quickly
and integrate systems more effectively. Python is powerful... and fast;
plays well with others; runs everywhere; is friendly & easy to learn;
is Open.
"""

frequencies = word_frequency(text)
print("Most common words:")
for word, count in frequencies.most_common(5):
    print(f"{word}: {count}")

# 单词长度分布
length_dist = Counter([len(word) for word in frequencies.keys()])
print("\nWord length distribution:")
for length, count in sorted(length_dist.items()):
    print(f"{length} letters: {count} words")
```

### 图数据结构

```python
from collections import defaultdict, deque

class Graph:
    def __init__(self, directed=False):
        self.graph = defaultdict(set)
        self.directed = directed
    
    def add_edge(self, u, v):
        self.graph[u].add(v)
        if not self.directed:
            self.graph[v].add(u)
    
    def bfs(self, start):
        """广度优先搜索"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")
            
            for neighbor in sorted(self.graph[vertex]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def dfs(self, start, visited=None):
        """深度优先搜索"""
        if visited is None:
            visited = set()
        
        visited.add(start)
        print(start, end=" ")
        
        for neighbor in sorted(self.graph[start]):
            if neighbor not in visited:
                self.dfs(neighbor, visited)

# 使用示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(4, 0)

print("BFS starting from vertex 0:")
g.bfs(0)  # 输出: 0 1 2 4 3

print("\nDFS starting from vertex 0:")
g.dfs(0)  # 输出: 0 1 2 3 4
```

## 内存优化技巧

### 使用生成器代替列表

```python
# 使用列表（存储所有值）
def get_squares_list(n):
    return [x**2 for x in range(n)]

# 使用生成器（按需生成值）
def get_squares_generator(n):
    return (x**2 for x in range(n))

import sys
list_version = get_squares_list(1000000)
gen_version = get_squares_generator(1000000)

print(f"列表大小: {sys.getsizeof(list_version)} 字节")
print(f"生成器大小: {sys.getsizeof(gen_version)} 字节")

# 使用生成器
for square in get_squares_generator(5):
    print(square)
```

### 使用 __slots__

```python
class PersonNormal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class PersonWithSlots:
    __slots__ = ['name', 'age']  # 限制只能有这些属性
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

import sys
p1 = PersonNormal('Alice', 25)
p2 = PersonWithSlots('Alice', 25)

print(f"普通对象大小: {sys.getsizeof(p1.__dict__)} 字节")
print(f"使用 __slots__ 的对象没有 __dict__")
print(f"对象本身大小比较:")
print(f"普通对象: {sys.getsizeof(p1)} 字节")
print(f"slots对象: {sys.getsizeof(p2)} 字节")
```

### 使用 bytes 和 bytearray

```python
# 字符串占用内存
text = "Hello" * 1000
print(f"字符串大小: {sys.getsizeof(text)} 字节")

# bytes 是不可变的
byte_data = b"Hello" * 1000
print(f"bytes 大小: {sys.getsizeof(byte_data)} 字节")

# bytearray 是可变的
byte_array = bytearray(b"Hello" * 1000)
print(f"bytearray 大小: {sys.getsizeof(byte_array)} 字节")

# 使用 bytearray 修改数据
byte_array[0] = 74  # 'J' 的 ASCII 码
print(byte_array[:5])  # bytearray(b'Jello')
```

## 最佳实践

1. **选择合适的数据结构**：根据需要执行的操作类型选择最合适的数据结构。
   - 需要快速查找：字典或集合
   - 需要有序数据：列表或 OrderedDict
   - 需要唯一值：集合
   - 需要固定大小的容器：元组
   - 需要先进先出队列：deque
   - 需要优先队列：heapq

2. **使用推导式**：当需要创建列表、字典或集合时，推导式通常比显式循环更简洁、更高效。

3. **利用生成器**：处理大量数据时，使用生成器可以显著降低内存使用量。

4. **使用内置函数**：优先使用内置函数（如 `map`、`filter`、`sorted`）而不是自己实现。

5. **理解可变性和不可变性**：了解每种数据结构的可变性特性，避免不必要的复制或意外的修改。

## 下一步

现在您已经掌握了 Python 的高级数据结构，接下来可以学习 [Python 迭代器与生成器](/intermediate/iterators-generators)，进一步提升您的 Python 编程能力。 