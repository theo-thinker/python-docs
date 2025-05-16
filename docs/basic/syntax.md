# Python 基本语法

Python 语法简洁明了，设计注重可读性，这使得学习和使用起来都相对容易。本章将介绍 Python 的基本语法规则。

## Python 程序结构

Python 程序由模块组成，模块包含语句，语句包含表达式，表达式构建并处理对象。

一个简单的 Python 程序示例：

```python
# 这是一个简单的 Python 程序
print("Hello, World!")  # 输出文本到控制台
```

## 代码缩进

Python 使用缩进来表示代码块，而不是像其他语言使用的花括号 `{}`。缩进的空格数可以自由选择，但必须保持一致。通常使用 4 个空格作为标准缩进。

```python
# 正确的缩进示例
if True:
    print("条件为真")
    if True:
        print("嵌套条件也为真")
# 下面的代码回到了第一层

# 错误的缩进示例（会引发 IndentationError）
if True:
print("这行代码缺少缩进")
```

::: warning 注意
缩进错误是 Python 初学者最常见的语法错误之一。务必保持一致的缩进风格！
:::

## 多行语句

Python 语句通常以换行符结束，但可以使用反斜杠 `\` 来继续上一行：

```python
total = 1 + \
        2 + \
        3
print(total)  # 输出: 6
```

在括号 `()`, 方括号 `[]` 或花括号 `{}` 内的语句不需要使用反斜杠即可跨行：

```python
days = ['Monday', 'Tuesday', 'Wednesday', 
        'Thursday', 'Friday', 'Saturday', 'Sunday']
```

## 注释

Python 支持单行和多行注释：

```python
# 这是单行注释

'''
这是多行注释
可以跨越多行
Python 实际上是使用三引号字符串作为多行注释
'''

"""
这也是多行注释
使用双引号也是可以的
"""
```

## 引号

Python 可以使用单引号 `'` 或双引号 `"` 来表示字符串：

```python
str1 = 'Hello'
str2 = "World"
```

三引号用于表示多行字符串：

```python
paragraph = """这是一个多行字符串。
它可以跨越多行。
非常方便！"""

print(paragraph)
```

## 空行

Python 中的空行是语句的一部分，用于增加代码的可读性。函数之间或类的方法之间空一行，类与函数之间空两行。

## 等待用户输入

```python
input("\n按回车键继续...")
```

## 同一行多条语句

Python 允许在同一行放置多个语句，用分号 `;` 分隔：

```python
a = 1; b = 2; c = 3
```

这种做法不推荐，因为它降低了代码的可读性。

## 代码组

缩进相同的一组语句构成一个代码块，称为代码组。如 if、while、def 和 class 这样的复合语句，首行以关键字开始，以冒号 `:` 结束：

```python
if expression:
    statement1
    statement2
    ...
    statementN
else:
    statement1
    statement2
    ...
    statementN
```

## 命令行参数

Python 可以通过 `sys` 模块的 `argv` 获取命令行参数：

```python
import sys

# 打印所有命令行参数
for i in sys.argv:
    print(i)

# 打印参数个数
print("参数个数:", len(sys.argv))
```

## 标准数据类型

Python 有五个标准数据类型：
- Number（数字）
- String（字符串）
- List（列表）
- Tuple（元组）
- Dictionary（字典）

这些数据类型将在[变量与数据类型](/basic/variables)章节详细介绍。

## print() 函数

`print()` 函数是 Python 中最常用的输出函数：

```python
# 基本使用
print("Hello, World!")

# 多个值，用空格分隔
print("姓名:", "张三", "年龄:", 25)

# 使用 sep 参数修改分隔符
print("姓名:", "张三", "年龄:", 25, sep='|')

# 使用 end 参数修改结尾字符（默认是换行符 \n）
print("Hello", end=' ')
print("World")  # 输出: Hello World
```

## 下一步

现在您已经了解了 Python 的基本语法规则，接下来可以学习 [Python 变量与数据类型](/basic/variables)。 