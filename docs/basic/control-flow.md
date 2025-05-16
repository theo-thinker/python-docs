# Python 控制流

控制流是编程中的基本概念，用于控制程序的执行顺序。Python 提供了多种控制流语句，包括条件语句和循环语句。

## 条件语句

条件语句允许您根据条件执行不同的代码块。

### if 语句

基本的条件判断使用 `if` 语句：

```python
age = 20

if age >= 18:
    print("您是成年人")
```

### if-else 语句

当条件为假时，执行 `else` 部分：

```python
age = 16

if age >= 18:
    print("您是成年人")
else:
    print("您是未成年人")
```

### if-elif-else 语句

多条件判断使用 `elif`（else if 的缩写）：

```python
score = 85

if score >= 90:
    grade = "优秀"
elif score >= 80:
    grade = "良好"
elif score >= 70:
    grade = "中等"
elif score >= 60:
    grade = "及格"
else:
    grade = "不及格"

print(f"您的成绩等级是：{grade}")  # 输出：您的成绩等级是：良好
```

### 嵌套条件语句

条件语句可以嵌套：

```python
age = 25
income = 30000

if age >= 18:
    print("您是成年人")
    if income >= 20000:
        print("您需要缴纳个人所得税")
    else:
        print("您的收入低于起征点，无需缴税")
else:
    print("您是未成年人，无需缴税")
```

### 条件表达式（三元运算符）

Python 支持简洁的条件表达式：

```python
age = 20
status = "成年" if age >= 18 else "未成年"
print(status)  # 输出：成年
```

## 循环语句

循环语句允许您多次执行代码块。

### while 循环

当条件为真时，重复执行代码块：

```python
# 打印 1 到 5
count = 1
while count <= 5:
    print(count)
    count += 1
```

### for 循环

遍历可迭代对象中的元素：

```python
# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Python":
    print(char)

# 使用 range() 函数
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

# 指定 range() 的起始和结束
for i in range(2, 6):  # 2, 3, 4, 5
    print(i)

# 指定 range() 的步长
for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
    print(i)
```

### 嵌套循环

循环可以嵌套：

```python
# 打印乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()  # 换行
```

输出：
```
1×1=1
1×2=2   2×2=4
1×3=3   2×3=6   3×3=9
1×4=4   2×4=8   3×4=12  4×4=16
1×5=5   2×5=10  3×5=15  4×5=20  5×5=25
1×6=6   2×6=12  3×6=18  4×6=24  5×6=30  6×6=36
1×7=7   2×7=14  3×7=21  4×7=28  5×7=35  6×7=42  7×7=49
1×8=8   2×8=16  3×8=24  4×8=32  5×8=40  6×8=48  7×8=56  8×8=64
1×9=9   2×9=18  3×9=27  4×9=36  5×9=45  6×9=54  7×9=63  8×9=72  9×9=81
```

## 控制流修改语句

### break 语句

`break` 语句用于提前退出循环：

```python
# 找到第一个能被 3 整除的数
for i in range(1, 10):
    if i % 3 == 0:
        print(f"找到了：{i}")
        break  # 找到后立即退出循环
```

### continue 语句

`continue` 语句用于跳过当前循环的剩余部分，直接进入下一次循环：

```python
# 打印不能被 3 整除的数
for i in range(1, 10):
    if i % 3 == 0:
        continue  # 跳过能被 3 整除的数
    print(i)  # 输出：1, 2, 4, 5, 7, 8
```

### else 子句

循环可以有 `else` 子句，当循环正常完成（未被 `break` 语句中断）时执行：

```python
# 检查列表中是否存在偶数
numbers = [1, 3, 5, 7, 9]
for num in numbers:
    if num % 2 == 0:
        print("找到偶数")
        break
else:
    print("没有偶数")  # 循环正常完成，打印此消息
```

```python
# 查找质数
num = 17
for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
        print(f"{num} 不是质数")
        break
else:
    print(f"{num} 是质数")
```

## pass 语句

`pass` 是空操作语句，用作占位符：

```python
def function_not_implemented_yet():
    pass  # 稍后实现此函数

if age < 18:
    pass  # 暂时不处理未成年人的情况
else:
    # 处理成年人的情况
    process_adult()
```

## 实际应用示例

### 简单的登录系统

```python
correct_username = "admin"
correct_password = "password123"
max_attempts = 3
attempts = 0

while attempts < max_attempts:
    username = input("请输入用户名: ")
    password = input("请输入密码: ")
    
    if username == correct_username and password == correct_password:
        print("登录成功！")
        break
    else:
        attempts += 1
        remaining = max_attempts - attempts
        if remaining > 0:
            print(f"用户名或密码错误，还有 {remaining} 次尝试机会")
        else:
            print("尝试次数过多，账户已锁定")
```

### 猜数字游戏

```python
import random

target = random.randint(1, 100)
guess_count = 0
max_guesses = 10

print("我想了一个 1 到 100 之间的数字，请猜猜是多少？")

while guess_count < max_guesses:
    try:
        guess = int(input(f"请输入你的猜测 (还剩 {max_guesses - guess_count} 次机会): "))
        guess_count += 1
        
        if guess < target:
            print("太小了！")
        elif guess > target:
            print("太大了！")
        else:
            print(f"恭喜你猜对了！答案是 {target}，你用了 {guess_count} 次猜中。")
            break
    except ValueError:
        print("请输入有效的数字！")
else:
    print(f"很遗憾，你没有猜中。正确答案是 {target}。")
```

### 列表推导式

列表推导式是一种创建列表的简洁方法，结合了循环和条件判断：

```python
# 生成 1 到 10 的平方列表
squares = [x**2 for x in range(1, 11)]
print(squares)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 生成偶数列表
evens = [x for x in range(1, 11) if x % 2 == 0]
print(evens)  # [2, 4, 6, 8, 10]

# 嵌套列表推导式
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 生成字典
squared_dict = {x: x**2 for x in range(1, 6)}
print(squared_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

## 下一步

现在您已经了解了 Python 的控制流语句，接下来可以学习 [Python 函数](/basic/functions)。 