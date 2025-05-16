# Python 正则表达式

正则表达式（Regular Expression，简称 regex）是一种强大的文本模式匹配和操作工具。在 Python 中，`re` 模块提供了对正则表达式的完整支持，使您能够轻松地进行复杂的字符串搜索、替换和提取。

## 正则表达式基础

### 导入正则表达式模块

使用 Python 的正则表达式功能，首先需要导入 `re` 模块：

```python
import re
```

### 基本匹配模式

以下是一些基本的正则表达式模式：

| 模式 | 描述 | 示例 |
|------|------|------|
| 字面字符 | 匹配文本中的实际字符 | `python` 匹配 "python" |
| `.` | 匹配任意单个字符（除了换行符） | `a.b` 匹配 "acb", "aab", "a@b" 等 |
| `^` | 匹配字符串开头 | `^Hello` 匹配以 "Hello" 开头的字符串 |
| `$` | 匹配字符串结尾 | `world$` 匹配以 "world" 结尾的字符串 |
| `*` | 匹配前一个字符 0 次或多次 | `a*` 匹配 "", "a", "aa", "aaa" 等 |
| `+` | 匹配前一个字符 1 次或多次 | `a+` 匹配 "a", "aa", "aaa" 等 |
| `?` | 匹配前一个字符 0 次或 1 次 | `colou?r` 匹配 "color" 和 "colour" |
| `{n}` | 匹配前一个字符恰好 n 次 | `a{3}` 匹配 "aaa" |
| `{n,}` | 匹配前一个字符至少 n 次 | `a{2,}` 匹配 "aa", "aaa" 等 |
| `{n,m}` | 匹配前一个字符 n 到 m 次 | `a{2,4}` 匹配 "aa", "aaa", "aaaa" |
| `[]` | 字符集，匹配方括号中的任意字符 | `[abc]` 匹配 "a", "b" 或 "c" |
| `[^]` | 否定字符集，匹配不在方括号中的任意字符 | `[^abc]` 匹配除了 "a", "b", "c" 以外的任意字符 |
| `\` | 转义特殊字符或表示特殊序列 | `\.` 匹配 "." 字符 |
| `\d` | 匹配任意数字，等同于 `[0-9]` | `\d+` 匹配一个或多个数字 |
| `\D` | 匹配任意非数字，等同于 `[^0-9]` | `\D+` 匹配一个或多个非数字字符 |
| `\w` | 匹配字母、数字或下划线，等同于 `[a-zA-Z0-9_]` | `\w+` 匹配一个或多个单词字符 |
| `\W` | 匹配非字母、数字或下划线，等同于 `[^a-zA-Z0-9_]` | `\W+` 匹配一个或多个非单词字符 |
| `\s` | 匹配任意空白字符（空格、制表符、换行符等） | `\s+` 匹配一个或多个空白字符 |
| `\S` | 匹配任意非空白字符 | `\S+` 匹配一个或多个非空白字符 |
| `\b` | 匹配单词边界 | `\bword\b` 匹配完整的单词 "word" |
| `\B` | 匹配非单词边界 | `\Bword\B` 匹配 "word" 但它必须在其他字符之间 |
| `|` | 或操作符 | `cat|dog` 匹配 "cat" 或 "dog" |
| `()` | 分组，捕获匹配的子字符串 | `(ab)+` 匹配 "ab", "abab", "ababab" 等 |
| `(?:)` | 非捕获分组 | `(?:ab)+` 类似 `(ab)+` 但不捕获匹配的文本 |

## Python 中的 re 模块函数

### re.search()

在字符串中搜索匹配模式的第一个位置：

```python
import re

text = "Python is awesome, isn't it?"
match = re.search(r"is", text)
if match:
    print(f"找到匹配：{match.group()}")
    print(f"起始位置：{match.start()}")
    print(f"结束位置：{match.end()}")
    print(f"匹配位置元组：{match.span()}")
```

### re.match()

从字符串的开头开始匹配模式：

```python
result = re.match(r"Python", text)  # 匹配成功，因为 text 以 "Python" 开头
print(result.group())  # 输出: Python

result = re.match(r"awesome", text)  # 匹配失败，因为 text 不以 "awesome" 开头
print(result)  # 输出: None
```

### re.findall()

返回字符串中所有匹配的子字符串列表：

```python
matches = re.findall(r"\w+", text)
print(matches)  # 输出: ['Python', 'is', 'awesome', 'isn', 't', 'it']
```

### re.finditer()

返回迭代器，每个迭代器元素是一个匹配对象：

```python
for match in re.finditer(r"\w+", text):
    print(f"{match.group()} 位于位置 {match.span()}")
```

### re.sub()

替换匹配的子字符串：

```python
result = re.sub(r"awesome", "great", text)
print(result)  # 输出: Python is great, isn't it?

# 使用函数进行替换
def capitalize(match):
    return match.group(0).upper()

result = re.sub(r"\b\w+\b", capitalize, text)
print(result)  # 输出: PYTHON IS AWESOME, ISN'T IT?
```

### re.split()

根据模式分割字符串：

```python
words = re.split(r"\s+", text)
print(words)  # 输出: ['Python', 'is', 'awesome,', "isn't", 'it?']

# 指定最大分割次数
words = re.split(r"\s+", text, maxsplit=2)
print(words)  # 输出: ['Python', 'is', "awesome, isn't it?"]
```

## 编译正则表达式

如果需要多次使用同一个正则表达式，可以先编译它以提高性能：

```python
pattern = re.compile(r"\d+")

text = "有 42 只猫和 31 只狗"
matches = pattern.findall(text)
print(matches)  # 输出: ['42', '31']

substitution = pattern.sub("XX", text)
print(substitution)  # 输出: 有 XX 只猫和 XX 只狗
```

编译后的正则表达式对象具有与 `re` 模块相同的方法（`search`、`match`、`findall` 等）。

## 分组和捕获

### 基本分组

使用圆括号 `()` 创建捕获组：

```python
text = "我的电话号码是 123-456-7890"
pattern = re.compile(r"(\d{3})-(\d{3})-(\d{4})")
match = pattern.search(text)

if match:
    print(f"完整匹配: {match.group(0)}")  # 输出: 123-456-7890
    print(f"第一组: {match.group(1)}")   # 输出: 123
    print(f"第二组: {match.group(2)}")   # 输出: 456
    print(f"第三组: {match.group(3)}")   # 输出: 7890
    print(f"所有组: {match.groups()}")   # 输出: ('123', '456', '7890')
```

### 命名分组

使用 `(?P<name>pattern)` 语法创建命名捕获组：

```python
pattern = re.compile(r"(?P<area>\d{3})-(?P<prefix>\d{3})-(?P<line>\d{4})")
match = pattern.search(text)

if match:
    print(f"区号: {match.group('area')}")    # 输出: 123
    print(f"前缀: {match.group('prefix')}")  # 输出: 456
    print(f"行号: {match.group('line')}")    # 输出: 7890
    print(f"命名组字典: {match.groupdict()}")  # 输出: {'area': '123', 'prefix': '456', 'line': '7890'}
```

### 非捕获分组

使用 `(?:pattern)` 创建非捕获组，这些组不会保存在结果中：

```python
text = "abc123def456"
pattern = re.compile(r"(?:\d+)([a-z]+)")
match = pattern.search(text)

if match:
    print(match.group(0))  # 输出: 123def
    print(match.group(1))  # 输出: def
    print(match.groups())  # 输出: ('def',)
```

## 贪婪与非贪婪匹配

默认情况下，量词 (`*`, `+`, `?`, `{n,m}`) 是贪婪的，它们会尽可能多地匹配字符：

```python
text = "<div>内容1</div><div>内容2</div>"

# 贪婪匹配（尽可能多地匹配）
pattern = re.compile(r"<div>.*</div>")
match = pattern.search(text)
print(match.group())  # 输出: <div>内容1</div><div>内容2</div>

# 非贪婪匹配（尽可能少地匹配）
pattern = re.compile(r"<div>.*?</div>")
match = pattern.search(text)
print(match.group())  # 输出: <div>内容1</div>
```

## 前向断言和后向断言

### 正向前瞻断言

`(?=...)` 匹配模式之前的位置，但不消耗字符：

```python
text = "价格: $10, €20, ¥30"

# 匹配后面跟着货币单位的数字
pattern = re.compile(r"\d+(?=\$|\€|\¥)")
matches = pattern.findall(text)
print(matches)  # 输出：[]（注意：这里找不到匹配，因为货币符号位于数字前面）

# 正确的用法：匹配货币符号后面的数字
pattern = re.compile(r"(?<=\$|\€|\¥)\d+")
matches = pattern.findall(text)
print(matches)  # 输出: ['10', '20', '30']
```

### 负向前瞻断言

`(?!...)` 匹配后面不是指定模式的位置：

```python
# 匹配不后跟感叹号的字母序列
text = "Hello! Hi. Hey!"
pattern = re.compile(r"\b\w+(?![!])\b")
matches = pattern.findall(text)
print(matches)  # 输出: ['Hi']
```

### 正向后顾断言

`(?<=...)` 匹配前面是指定模式的位置：

```python
# 匹配前面有货币符号的数字
text = "价格: $10, €20, ¥30"
pattern = re.compile(r"(?<=\$|\€|\¥)\d+")
matches = pattern.findall(text)
print(matches)  # 输出: ['10', '20', '30']
```

### 负向后顾断言

`(?<!...)` 匹配前面不是指定模式的位置：

```python
# 匹配前面不是字母的数字
text = "a1 b2 3c 45"
pattern = re.compile(r"(?<![a-zA-Z])\d+")
matches = pattern.findall(text)
print(matches)  # 输出: ['3', '45']
```

## 标志（Flags）

`re` 模块提供了多个标志，可以修改正则表达式的行为：

```python
# re.IGNORECASE 或 re.I：忽略大小写
pattern = re.compile(r"python", re.IGNORECASE)
text = "Python 是一种编程语言"
match = pattern.search(text)
print(match.group())  # 输出: Python

# re.MULTILINE 或 re.M：允许 ^ 和 $ 匹配每行的开始和结束
text = """第一行
第二行
第三行"""
pattern = re.compile(r"^第.*$", re.MULTILINE)
matches = pattern.findall(text)
print(matches)  # 输出: ['第一行', '第二行', '第三行']

# re.DOTALL 或 re.S：让 . 匹配包括换行符在内的所有字符
text = "这是第一行\n这是第二行"
pattern = re.compile(r".*", re.DOTALL)
match = pattern.search(text)
print(match.group())  # 输出: 这是第一行\n这是第二行

# re.VERBOSE 或 re.X：允许正则表达式中的注释和空白
pattern = re.compile(r"""
    \d{3}    # 区号
    [-\s]?   # 可选的分隔符
    \d{3}    # 前缀
    [-\s]?   # 可选的分隔符
    \d{4}    # 行号
""", re.VERBOSE)
text = "联系我：123-456-7890"
match = pattern.search(text)
print(match.group())  # 输出: 123-456-7890

# 可以使用 | 操作符组合多个标志
pattern = re.compile(r"^python", re.IGNORECASE | re.MULTILINE)
```

## 常见正则表达式示例

### 电子邮件地址验证

```python
email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
emails = ["user@example.com", "invalid@.com", "name@domain.co.uk", "@missing.com"]

for email in emails:
    if email_pattern.match(email):
        print(f"{email} 是有效的电子邮件地址")
    else:
        print(f"{email} 不是有效的电子邮件地址")
```

### URL 提取

```python
text = """
访问我们的网站 https://www.example.com 或 http://subdomain.example.co.uk/path?query=term
也可以查看 ftp://files.example.org
"""

url_pattern = re.compile(r"(https?|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9._/?=&%]*)?")
urls = url_pattern.findall(text)
for url in urls:
    print(url[0] + url[1])  # 使用 findall 返回所有分组，需要手动拼接
```

更简单的方法是使用非捕获组：

```python
url_pattern = re.compile(r"(?:https?|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9._/?=&%]*)?")
urls = url_pattern.findall(text)
for url in urls:
    print(url)
```

### 手机号码格式化

```python
phone_numbers = ["13812345678", "1381234567", "138-1234-5678"]
phone_pattern = re.compile(r"1[3-9]\d{9}|1[3-9]\d{1}-\d{4}-\d{4}")

for number in phone_numbers:
    if phone_pattern.fullmatch(number):
        # 格式化为标准格式
        formatted = re.sub(r"1([3-9]\d)(\d{4})(\d{4})", r"1\1-\2-\3", number)
        formatted = re.sub(r"1([3-9]\d)-(\d{4})-(\d{4})", r"1\1-\2-\3", formatted)
        print(f"{number} 是有效的手机号码，标准格式: {formatted}")
    else:
        print(f"{number} 不是有效的手机号码")
```

### 提取 HTML 标签

```python
html = "<html><head><title>网页标题</title></head><body><p>这是一个<b>示例</b>段落</p></body></html>"

# 提取所有 HTML 标签
tag_pattern = re.compile(r"<[^>]+>")
tags = tag_pattern.findall(html)
print(tags)  # 输出: ['<html>', '<head>', '<title>', '</title>', '</head>', '<body>', '<p>', '<b>', '</b>', '</p>', '</body>', '</html>']

# 提取 HTML 标签及其内容
content_pattern = re.compile(r"<([a-z]+)>(.*?)</\1>")
contents = content_pattern.findall(html)
for tag, content in contents:
    print(f"{tag}: {content}")
```

### 密码强度检查

```python
def check_password_strength(password):
    # 检查长度（至少 8 个字符）
    if len(password) < 8:
        return "密码太短，至少需要 8 个字符"
    
    # 检查是否包含数字
    if not re.search(r"\d", password):
        return "密码应包含至少一个数字"
    
    # 检查是否包含小写字母
    if not re.search(r"[a-z]", password):
        return "密码应包含至少一个小写字母"
    
    # 检查是否包含大写字母
    if not re.search(r"[A-Z]", password):
        return "密码应包含至少一个大写字母"
    
    # 检查是否包含特殊字符
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "密码应包含至少一个特殊字符"
    
    return "密码强度良好"

# 测试密码强度
passwords = ["pass", "password", "Password1", "Password1!", "p@$$w0rd"]
for password in passwords:
    print(f"{password}: {check_password_strength(password)}")
```

## 性能考虑

1. **编译正则表达式**：如果要重复使用一个正则表达式，先编译它可以提高性能。

2. **避免过度回溯**：复杂的正则表达式可能导致回溯过多，降低性能。例如，`(a+)+b` 对于类似 "aaaaaaaaaaaaX" 的输入可能导致灾难性回溯。

3. **使用非捕获组**：如果不需要捕获结果，使用非捕获组 `(?:...)`。

4. **尽量具体**：使正则表达式尽可能具体，避免过于通用的模式。

5. **使用原始字符串**：在 Python 中，总是使用原始字符串（以 `r` 开头）来定义正则表达式，避免反斜杠转义问题。

## 正则表达式调试工具

Python 的 `re` 模块提供了 `DEBUG` 标志，可以显示编译后的正则表达式的详细信息：

```python
pattern = re.compile(r"\d+", re.DEBUG)
```

此外，有许多在线工具可以帮助测试和调试正则表达式，例如：
- regex101.com
- regexr.com
- debuggex.com

## 最佳实践

1. **保持简单**：复杂的正则表达式难以理解和维护。如果需要，将一个复杂的正则表达式分解为多个简单的正则表达式。

2. **添加注释**：对于复杂的正则表达式，使用 `re.VERBOSE` 标志并添加注释。

3. **测试极端情况**：测试边界条件、空字符串和特殊字符。

4. **考虑安全性**：处理用户输入时，避免正则表达式注入攻击（例如，限制回溯次数）。

5. **不要过度依赖**：有些问题使用正则表达式并不是最佳解决方案，考虑使用专门的解析库（如 HTML 解析器、CSV 解析器等）。

## 下一步

现在您已经掌握了 Python 正则表达式的基础知识，接下来可以深入学习 [Python 日期与时间](/intermediate/datetime) 处理。 