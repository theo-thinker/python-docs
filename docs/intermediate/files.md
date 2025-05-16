# Python 文件操作

文件操作是编程中的基础任务之一，几乎所有程序都需要读取或写入文件。Python 提供了简单而强大的文件处理机制，让您能够轻松地操作各种类型的文件。

## 文件基础

### 打开和关闭文件

使用 `open()` 函数打开文件，它返回一个文件对象。完成操作后，必须使用 `close()` 方法关闭文件。

```python
# 打开文件
file = open('example.txt', 'r')  # 'r' 表示读取模式

# 操作文件
content = file.read()
print(content)

# 关闭文件
file.close()
```

### 文件打开模式

`open()` 函数的第二个参数指定文件的打开模式：

| 模式 | 描述 |
|------|------|
| `'r'` | 读取模式（默认） |
| `'w'` | 写入模式（会覆盖已有内容） |
| `'a'` | 追加模式（在文件末尾追加内容） |
| `'x'` | 独占创建模式（如果文件已存在则失败） |
| `'b'` | 二进制模式（与其他模式结合使用，如 `'rb'`） |
| `'t'` | 文本模式（默认，与其他模式结合使用，如 `'rt'`） |
| `'+'` | 读写模式（与其他模式结合使用，如 `'r+'`） |

示例：

```python
# 写入模式 - 创建新文件或覆盖已有文件
file = open('example.txt', 'w')
file.write('Hello, World!')
file.close()

# 追加模式 - 在文件末尾添加内容
file = open('example.txt', 'a')
file.write('\nAppended line.')
file.close()

# 二进制模式 - 处理图像等二进制文件
file = open('image.jpg', 'rb')
image_data = file.read()
file.close()
```

## 使用 with 语句（上下文管理器）

使用 `with` 语句可以自动处理文件的关闭，即使发生异常也能确保文件被正确关闭。这是推荐的文件操作方式：

```python
# 使用 with 语句打开文件
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
# 文件在 with 块结束时自动关闭
```

## 文件读取操作

### 读取整个文件

```python
with open('example.txt', 'r', encoding='utf-8') as file:
    content = file.read()  # 读取整个文件内容
    print(content)
```

### 逐行读取

```python
# 方法1：使用 readlines() 方法
with open('example.txt', 'r') as file:
    lines = file.readlines()  # 返回包含所有行的列表
    for line in lines:
        print(line.strip())  # strip() 移除行尾的换行符

# 方法2：直接遍历文件对象（内存效率更高）
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())
```

### 读取指定字节数

```python
with open('example.txt', 'r') as file:
    chunk = file.read(5)  # 读取前5个字符
    print(chunk)
```

### 文件指针操作

```python
with open('example.txt', 'r') as file:
    # 读取一部分
    first_chunk = file.read(10)
    print(first_chunk)
    
    # 获取当前文件指针位置
    position = file.tell()
    print(f"当前位置: {position}")
    
    # 将文件指针移动到指定位置
    file.seek(0)  # 回到文件开头
    print(f"重新定位后的位置: {file.tell()}")
    
    # 再次读取
    content = file.read()
    print(content)
```

## 文件写入操作

### 写入字符串

```python
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write('这是第一行\n')
    file.write('这是第二行\n')
```

### 写入多行

```python
lines = ['第一行\n', '第二行\n', '第三行\n']

with open('output.txt', 'w') as file:
    file.writelines(lines)  # writelines() 不会自动添加换行符
```

### 格式化写入

```python
data = {'name': '张三', 'age': 30, 'city': '北京'}

with open('formatted.txt', 'w', encoding='utf-8') as file:
    for key, value in data.items():
        file.write(f"{key}: {value}\n")
```

## 文件和目录操作

Python 的 `os` 和 `shutil` 模块提供了丰富的文件和目录操作功能：

```python
import os
import shutil

# 获取当前工作目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 列出目录内容
files = os.listdir('.')
print(f"目录内容: {files}")

# 检查路径是否存在
if os.path.exists('example.txt'):
    print('文件存在')

# 获取文件信息
file_size = os.path.getsize('example.txt')
mod_time = os.path.getmtime('example.txt')
print(f"文件大小: {file_size} 字节")
print(f"修改时间: {mod_time}")

# 创建目录
os.mkdir('new_folder')  # 创建单个目录
os.makedirs('path/to/nested/folder', exist_ok=True)  # 创建多级目录

# 删除文件
os.remove('unwanted_file.txt')

# 删除目录
os.rmdir('empty_folder')  # 只能删除空目录
shutil.rmtree('folder_with_contents')  # 删除目录及其内容

# 重命名文件或目录
os.rename('old_name.txt', 'new_name.txt')

# 移动文件或目录
shutil.move('file.txt', 'new_location/file.txt')

# 复制文件
shutil.copy('source.txt', 'destination.txt')  # 复制文件
shutil.copy2('source.txt', 'destination.txt')  # 复制文件及其元数据

# 复制目录
shutil.copytree('source_folder', 'destination_folder')
```

## 使用 pathlib 模块

Python 3.4 引入了 `pathlib` 模块，它提供了一种面向对象的方式来处理文件路径：

```python
from pathlib import Path

# 创建路径对象
path = Path('example.txt')
dir_path = Path('my_directory')

# 检查路径是否存在
if path.exists():
    print(f"{path} 存在")

# 路径信息
print(f"文件名: {path.name}")
print(f"后缀: {path.suffix}")
print(f"父目录: {path.parent}")
print(f"绝对路径: {path.absolute()}")

# 路径组合
new_path = Path('documents') / 'reports' / 'annual.pdf'
print(new_path)

# 创建目录
dir_path.mkdir(exist_ok=True)
Path('nested/folders').mkdir(parents=True, exist_ok=True)

# 列出目录内容
for item in Path('.').iterdir():
    print(item)

# 查找文件
py_files = list(Path('.').glob('*.py'))
all_py_files = list(Path('.').rglob('*.py'))  # 递归查找

# 读写文件
text = path.read_text(encoding='utf-8')  # 读取文本
path.write_text('新内容', encoding='utf-8')  # 写入文本

# 二进制文件操作
binary_data = Path('image.jpg').read_bytes()
Path('new_image.jpg').write_bytes(binary_data)

# 文件操作
if path.exists():
    path.unlink()  # 删除文件

# 重命名文件
path.rename('new_name.txt')
```

## CSV 文件操作

CSV（逗号分隔值）是一种常见的数据交换格式，Python 的 `csv` 模块提供了简单的处理方法：

```python
import csv

# 写入 CSV 文件
data = [
    ['姓名', '年龄', '城市'],
    ['张三', '30', '北京'],
    ['李四', '25', '上海'],
    ['王五', '35', '广州']
]

with open('data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# 读取 CSV 文件
with open('data.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# 使用字典操作 CSV
dict_data = [
    {'姓名': '张三', '年龄': 30, '城市': '北京'},
    {'姓名': '李四', '年龄': 25, '城市': '上海'},
    {'姓名': '王五', '年龄': 35, '城市': '广州'}
]

with open('dict_data.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['姓名', '年龄', '城市']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    writer.writerows(dict_data)

with open('dict_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # 每行都是一个字典
```

## JSON 文件操作

JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，Python 的 `json` 模块提供了操作 JSON 数据的函数：

```python
import json

# Python 对象
data = {
    'name': '张三',
    'age': 30,
    'city': '北京',
    'skills': ['Python', 'Java', 'SQL'],
    'is_employee': True,
    'address': {
        'street': '中关村大街',
        'zipcode': '100080'
    }
}

# 写入 JSON 文件
with open('data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
    # ensure_ascii=False 确保中文字符正确输出
    # indent=4 使输出格式美观

# 读取 JSON 文件
with open('data.json', 'r', encoding='utf-8') as file:
    loaded_data = json.load(file)
    print(loaded_data)

# 字符串与 JSON 转换
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)

obj = json.loads(json_str)
print(obj['name'])
```

## 二进制文件操作

Python 可以处理任何类型的二进制文件，如图像、音频或自定义格式的文件：

```python
# 读取二进制文件
with open('image.jpg', 'rb') as file:
    binary_data = file.read()
    print(f"文件大小: {len(binary_data)} 字节")

# 写入二进制文件
with open('copy_image.jpg', 'wb') as file:
    file.write(binary_data)

# 处理大文件 - 分块读取
chunk_size = 4096  # 4KB 块
with open('large_file.bin', 'rb') as source:
    with open('large_file_copy.bin', 'wb') as target:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:  # 到达文件末尾
                break
            target.write(chunk)
```

## 临时文件

`tempfile` 模块用于创建临时文件和目录：

```python
import tempfile
import os

# 创建临时文件
with tempfile.NamedTemporaryFile(delete=False) as temp:
    print(f"临时文件名: {temp.name}")
    temp.write(b'This is some temporary data')

# 使用临时文件
with open(temp.name, 'rb') as file:
    data = file.read()
    print(data)

# 删除临时文件
os.unlink(temp.name)

# 创建临时目录
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"临时目录: {temp_dir}")
    # 使用临时目录
    temp_file_path = os.path.join(temp_dir, 'temp_file.txt')
    with open(temp_file_path, 'w') as file:
        file.write('Temporary file in a temporary directory')
    
    # 读取临时文件
    with open(temp_file_path, 'r') as file:
        print(file.read())
    
    # 临时目录在 with 块结束时自动删除
```

## 文件压缩和解压

Python 提供了多种处理压缩文件的模块：

```python
import zipfile
import tarfile
import gzip
import shutil

# 创建 ZIP 文件
with zipfile.ZipFile('archive.zip', 'w') as zipf:
    zipf.write('file1.txt')
    zipf.write('file2.txt')
    # 添加目录中的所有文件
    for root, _, files in os.walk('my_folder'):
        for file in files:
            zipf.write(os.path.join(root, file))

# 解压 ZIP 文件
with zipfile.ZipFile('archive.zip', 'r') as zipf:
    # 列出压缩文件内容
    print(zipf.namelist())
    # 解压所有文件
    zipf.extractall('extracted_folder')
    # 解压单个文件
    zipf.extract('file1.txt', 'specific_folder')

# 创建 TAR 文件
with tarfile.open('archive.tar.gz', 'w:gz') as tarf:
    tarf.add('file1.txt')
    tarf.add('my_folder')

# 解压 TAR 文件
with tarfile.open('archive.tar.gz', 'r:gz') as tarf:
    # 列出压缩文件内容
    print(tarf.getnames())
    # 解压所有文件
    tarf.extractall('extracted_folder')

# 使用 gzip 压缩单个文件
with open('file.txt', 'rb') as f_in:
    with gzip.open('file.txt.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# 解压 gzip 文件
with gzip.open('file.txt.gz', 'rb') as f_in:
    with open('file_uncompressed.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```

## 文件监控

使用 `watchdog` 库（需要安装：`pip install watchdog`）可以监控文件系统的变化：

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"创建了文件: {event.src_path}")
    
    def on_deleted(self, event):
        if not event.is_directory:
            print(f"删除了文件: {event.src_path}")
    
    def on_modified(self, event):
        if not event.is_directory:
            print(f"修改了文件: {event.src_path}")
    
    def on_moved(self, event):
        if not event.is_directory:
            print(f"移动了文件: {event.src_path} -> {event.dest_path}")

path = "."  # 监控当前目录
handler = MyHandler()
observer = Observer()
observer.schedule(handler, path, recursive=True)
observer.start()

try:
    print(f"开始监控目录: {path}")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
```

## 文件安全和最佳实践

1. **异常处理**：始终使用 `try-except` 块或 `with` 语句处理文件操作，确保文件正确关闭。

2. **路径安全**：
   ```python
   import os.path

   # 不安全的路径拼接
   # unsafe_path = base_dir + '/' + user_input  # 不要这样做!

   # 安全的路径拼接
   safe_path = os.path.join(base_dir, user_input)
   
   # 确保路径不会超出允许的范围
   safe_path = os.path.normpath(safe_path)
   if not safe_path.startswith(base_dir):
       raise ValueError("路径操作不允许!")
   ```

3. **文件锁定**：在多进程环境中使用文件锁：
   ```python
   import fcntl
   import time

   with open('shared_file.txt', 'w') as file:
       try:
           fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
           # 操作文件...
           time.sleep(10)  # 模拟长时间操作
           file.write('Process 1 wrote this\n')
       except IOError:
           print("无法获得文件锁")
       finally:
           fcntl.flock(file, fcntl.LOCK_UN)  # 释放锁
   ```

4. **性能考虑**：
   - 对于大文件，使用分块读取而不是一次读取整个文件
   - 在适当的情况下使用二进制模式
   - 考虑使用 `mmap` 模块进行内存映射文件操作

5. **编码处理**：明确指定编码，尤其是处理非ASCII文本时：
   ```python
   with open('file.txt', 'r', encoding='utf-8') as file:
       content = file.read()
   ```

## 下一步

现在您已经了解了 Python 的文件操作，接下来可以学习 [Python 正则表达式](/intermediate/regex)，它可以帮助您在文件处理中完成更复杂的模式匹配和文本处理。 