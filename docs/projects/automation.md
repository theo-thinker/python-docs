# Python 自动化脚本

Python 是自动化任务的理想选择，凭借其简洁的语法和丰富的库，能够轻松实现各种自动化需求。本章介绍如何使用 Python 创建自动化脚本，提高工作效率。

## 自动化基础

### 为什么选择 Python 进行自动化？

Python 在自动化领域具有以下优势：

1. **简洁易读的语法**：代码简单直观
2. **丰富的标准库**：内置多种实用功能
3. **大量第三方库**：几乎所有自动化任务都有对应的库
4. **跨平台兼容性**：可在 Windows、MacOS 和 Linux 上运行
5. **活跃的社区支持**：丰富的教程和解决方案

### 自动化应用场景

Python 自动化可应用于多种场景：

- **文件操作**：批量重命名、移动、复制文件
- **数据处理**：提取、转换、加载数据
- **网络请求**：API 调用、网页抓取
- **定时任务**：定期备份、报告生成
- **系统管理**：监控资源、管理服务
- **测试自动化**：单元测试、UI 测试
- **办公自动化**：Excel 处理、PDF 生成、邮件发送

### 准备工作

在开始编写自动化脚本前，需要做好以下准备：

1. **安装 Python**：推荐使用 Python 3.13 或更高版本
2. **设置虚拟环境**：隔离项目依赖
3. **熟悉基本语法**：掌握 Python 基础知识
4. **了解相关库**：针对具体任务选择合适的库

```bash
# 创建虚拟环境
python -m venv automation_env

# 激活虚拟环境
# Windows
automation_env\Scripts\activate
# MacOS/Linux
source automation_env/bin/activate

# 安装必要的库
pip install requests pandas openpyxl python-docx pillow pyautogui schedule
```

## 文件操作自动化

文件操作是最常见的自动化任务之一，Python 提供了多种处理文件的方式。

### 基本文件操作

```python
import os
import shutil

# 列出目录内容
files = os.listdir("/path/to/directory")
print(files)

# 检查文件是否存在
if os.path.exists("/path/to/file.txt"):
    print("文件存在")

# 创建目录
os.makedirs("/path/to/new/directory", exist_ok=True)

# 复制文件
shutil.copy2("/path/to/source.txt", "/path/to/destination.txt")

# 移动文件
shutil.move("/path/to/source.txt", "/path/to/new/location.txt")

# 删除文件
os.remove("/path/to/file.txt")

# 删除目录
shutil.rmtree("/path/to/directory")
```

### 批量重命名文件

```python
import os

def batch_rename(directory, prefix, extension=None):
    """批量重命名指定目录下的文件"""
    for count, filename in enumerate(os.listdir(directory)):
        # 如果指定了扩展名过滤，则跳过不匹配的文件
        if extension and not filename.endswith(extension):
            continue
            
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1]
        
        # 创建新文件名
        new_name = f"{prefix}_{count:03d}{ext}"
        
        # 构建完整路径
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        
        # 重命名文件
        os.rename(src, dst)
        print(f"重命名: {filename} -> {new_name}")

# 示例：重命名所有 JPG 文件
batch_rename("/path/to/photos", "vacation", ".jpg")
```

### 遍历目录树

```python
import os

def explore_directory(directory):
    """递归遍历目录树"""
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}[目录]: {os.path.basename(root)}/")
        
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}[文件]: {file}")

# 示例：遍历项目目录
explore_directory("/path/to/project")
```

### 查找和过滤文件

```python
import os
import fnmatch
import re

def find_files(directory, pattern):
    """查找匹配指定模式的文件"""
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

# 使用通配符查找
python_files = find_files("/path/to/project", "*.py")
print(f"找到 {len(python_files)} 个 Python 文件")

# 使用正则表达式查找文件内容
def find_in_files(directory, pattern, file_ext="*.txt"):
    """在文件内容中查找匹配的模式"""
    matches = []
    regex = re.compile(pattern)
    
    for file_path in find_files(directory, file_ext):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append((file_path, line_num, line.strip()))
        except (UnicodeDecodeError, IOError):
            # 跳过无法读取的文件
            pass
            
    return matches

# 示例：在所有 Python 文件中查找 "import" 语句
results = find_in_files("/path/to/project", r"import\s+(\w+)", "*.py")
for file_path, line_num, line in results[:10]:  # 只显示前 10 个结果
    print(f"{file_path}:{line_num}: {line}")
```

### 文件监控

```python
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"创建文件: {event.src_path}")
    
    def on_deleted(self, event):
        if not event.is_directory:
            print(f"删除文件: {event.src_path}")
    
    def on_modified(self, event):
        if not event.is_directory:
            print(f"修改文件: {event.src_path}")
    
    def on_moved(self, event):
        if not event.is_directory:
            print(f"移动文件: {event.src_path} -> {event.dest_path}")

def monitor_directory(path):
    """监控指定目录的文件变化"""
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    
    try:
        print(f"开始监控目录: {path}")
        print("按 Ctrl+C 停止监控")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 示例：监控下载目录
# 需要先安装 watchdog 库：pip install watchdog
# monitor_directory("/path/to/downloads")
```

## 数据处理自动化

数据处理是自动化脚本的关键应用之一，Python 提供了强大的数据处理工具。

### 使用 pandas 处理表格数据

```python
import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv("data.csv")

# 查看数据基本信息
print(df.info())
print(df.describe())

# 数据清洗
df = df.dropna()  # 删除缺失值
df = df[df['age'] > 0]  # 过滤数据

# 数据转换
df['name'] = df['name'].str.upper()  # 转换为大写
df['birth_year'] = 2023 - df['age']  # 创建新列

# 数据分组和聚合
result = df.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'age': 'mean'
})

# 保存结果
result.to_csv("result.csv")
df.to_excel("processed_data.xlsx", index=False)
```

### Excel 文件处理

```python
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

def process_excel(input_file, output_file):
    """处理 Excel 文件并添加格式"""
    # 使用 pandas 读取数据
    df = pd.read_excel(input_file)
    
    # 数据处理
    df['总分'] = df['语文'] + df['数学'] + df['英语']
    df['平均分'] = df['总分'] / 3
    df['是否及格'] = df['平均分'] >= 60
    
    # 使用 openpyxl 添加格式
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "成绩单"
    
    # 添加数据
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    
    # 添加标题格式
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
    
    # 添加条件格式
    for row in range(2, ws.max_row + 1):
        # 如果不及格，设置为红色
        if ws.cell(row=row, column=df.columns.get_loc('是否及格') + 1).value == False:
            ws.cell(row=row, column=df.columns.get_loc('平均分') + 1).font = Font(color="FF0000")
    
    # 调整列宽
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # 保存结果
    wb.save(output_file)
    print(f"处理完成，结果已保存至 {output_file}")

# 示例：处理成绩单
# process_excel("学生成绩.xlsx", "成绩单报告.xlsx")
```

### 处理 CSV 和 JSON 数据

```python
import csv
import json
import pandas as pd

# 读取 CSV 文件
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# 转换为 JSON
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 读取 JSON 文件
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 合并多个 CSV 文件
def merge_csv_files(file_list, output_file):
    """合并多个 CSV 文件为一个"""
    # 使用 pandas 合并
    dfs = [pd.read_csv(file) for file in file_list]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"已合并 {len(file_list)} 个文件到 {output_file}")

# 示例：合并月度报表
# files = ['jan.csv', 'feb.csv', 'mar.csv']
# merge_csv_files(files, 'q1_report.csv')
```

### 数据清洗转换

```python
import pandas as pd
import re

def clean_data(df):
    """清洗数据框中的数据"""
    # 创建副本，避免修改原始数据
    cleaned_df = df.copy()
    
    # 删除重复行
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 处理缺失值
    cleaned_df['age'] = cleaned_df['age'].fillna(cleaned_df['age'].median())
    cleaned_df['income'] = cleaned_df['income'].fillna(0)
    
    # 删除包含过多缺失值的行
    cleaned_df = cleaned_df.dropna(thresh=len(cleaned_df.columns) - 2)
    
    # 数据类型转换
    cleaned_df['age'] = cleaned_df['age'].astype(int)
    
    # 文本清洗
    cleaned_df['name'] = cleaned_df['name'].str.strip()
    
    # 使用正则表达式清洗电话号码
    def clean_phone(phone):
        if pd.isna(phone):
            return None
        # 只保留数字
        digits = re.sub(r'\D', '', str(phone))
        # 格式化为 xxx-xxxx-xxxx
        if len(digits) == 11:
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        return digits
    
    cleaned_df['phone'] = cleaned_df['phone'].apply(clean_phone)
    
    return cleaned_df

# 示例：清洗客户数据
# df = pd.read_csv('customers.csv')
# clean_df = clean_data(df)
# clean_df.to_csv('clean_customers.csv', index=False)
```

## 网络请求自动化

网络自动化可以让您与网站和在线服务进行交互，实现数据抓取、API 调用等功能。

### 基本 HTTP 请求

```python
import requests

# 发送 GET 请求
response = requests.get('https://api.github.com/events')
print(f"状态码: {response.status_code}")
print(f"响应头: {response.headers['content-type']}")

# 解析 JSON 响应
events = response.json()
for event in events[:3]:  # 显示前 3 个事件
    print(f"类型: {event['type']}, 作者: {event.get('actor', {}).get('login')}")

# 发送带参数的 GET 请求
params = {
    'q': 'python',
    'sort': 'stars',
    'order': 'desc'
}
response = requests.get('https://api.github.com/search/repositories', params=params)
repos = response.json()
print(f"找到 {repos['total_count']} 个仓库")

# 发送 POST 请求
data = {
    'name': 'John',
    'email': 'john@example.com'
}
response = requests.post('https://httpbin.org/post', data=data)
print(response.json())

# 处理请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json'
}
response = requests.get('https://api.github.com/users/octocat', headers=headers)
print(response.json().get('name'))

# 处理身份验证
response = requests.get('https://api.github.com/user', auth=('username', 'password'))
# 更安全的方法是使用 token
# response = requests.get('https://api.github.com/user', headers={'Authorization': f'token {token}'})
```

### 网页内容抓取

```python
import requests
from bs4 import BeautifulSoup

def scrape_quotes():
    """抓取名人名言网站的内容"""
    url = 'http://quotes.toscrape.com/'
    response = requests.get(url)
    
    # 检查请求是否成功
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return []
    
    # 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 查找所有引言
    quotes = []
    for quote in soup.select('.quote'):
        text = quote.select_one('.text').get_text()
        author = quote.select_one('.author').get_text()
        tags = [tag.get_text() for tag in quote.select('.tag')]
        
        quotes.append({
            'text': text,
            'author': author,
            'tags': tags
        })
    
    return quotes

# 示例：获取并显示名人名言
# quotes = scrape_quotes()
# for i, quote in enumerate(quotes, 1):
#     print(f"{i}. {quote['text']} - {quote['author']}")
#     print(f"   标签: {', '.join(quote['tags'])}")
#     print()
```

### API 数据获取与处理

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_weather_forecast(city, api_key):
    """获取城市的天气预报"""
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric',  # 摄氏度
        'lang': 'zh_cn'     # 中文
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        data = response.json()
        
        # 提取有用的信息
        forecasts = []
        for item in data['list']:
            time = datetime.fromtimestamp(item['dt'])
            temp = item['main']['temp']
            feels_like = item['main']['feels_like']
            weather = item['weather'][0]['description']
            wind_speed = item['wind']['speed']
            
            forecasts.append({
                'time': time,
                'temperature': temp,
                'feels_like': feels_like,
                'weather': weather,
                'wind_speed': wind_speed
            })
        
        # 创建数据框
        df = pd.DataFrame(forecasts)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

# 示例：获取北京的天气预报
# api_key = "你的API密钥"  # 需要在 OpenWeatherMap 网站注册获取
# weather_df = get_weather_forecast('Beijing', api_key)
# if weather_df is not None:
#     print(weather_df.head())
#     # 保存为 CSV
#     weather_df.to_csv('beijing_weather.csv', index=False)
```

### 自动化下载文件

```python
import requests
import os
from tqdm import tqdm

def download_file(url, output_path=None):
    """下载文件并显示进度条"""
    # 如果没有指定输出路径，使用 URL 中的文件名
    if output_path is None:
        output_path = os.path.basename(url)
    
    # 创建目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 发送请求并获取响应
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 确保请求成功
    
    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))
    
    # 显示下载进度条
    print(f"下载 {url} 到 {output_path}")
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)
    
    # 下载文件
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    print(f"下载完成: {output_path}")
    return output_path

# 示例：下载 Python 文档
# download_file('https://docs.python.org/3/archives/python-3.9.5-docs-pdf-a4.zip', 'python-docs.zip')
```

## 任务调度自动化

将脚本安排为自动运行的任务，是自动化的重要环节。

### 使用 schedule 库进行简单调度

```python
import schedule
import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduled_tasks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('scheduler')

def job():
    """要执行的任务"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"执行计划任务，当前时间: {now}")
    # 这里可以放置您的实际任务代码

# 定义不同的调度计划
def setup_schedule():
    # 每 10 秒执行一次
    schedule.every(10).seconds.do(job)
    
    # 每小时执行一次
    schedule.every().hour.do(job)
    
    # 每天特定时间执行
    schedule.every().day.at("10:30").do(job)
    
    # 每周一执行
    schedule.every().monday.do(job)
    
    # 特定时间执行（带标签）
    schedule.every().day.at("12:00").do(job).tag('backup')
    
    logger.info("调度任务已设置")

# 运行调度程序
def run_scheduler():
    logger.info("启动调度程序...")
    setup_schedule()
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("调度程序已停止")

# 示例：运行调度程序
# if __name__ == "__main__":
#     run_scheduler()
```

### 取消计划任务

```python
# 清除所有任务
schedule.clear()

# 清除特定标签的任务
schedule.clear('backup')

# 获取所有任务
jobs = schedule.get_jobs()
```

### 使用系统任务调度器

#### 在 Windows 上使用 Task Scheduler

```python
import subprocess

def create_windows_task(task_name, script_path, schedule_time):
    """创建 Windows 计划任务"""
    # 构建命令
    cmd = f'schtasks /create /tn "{task_name}" /tr "python {script_path}" /sc DAILY /st {schedule_time} /f'
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"已创建计划任务: {task_name}")
    except subprocess.CalledProcessError as e:
        print(f"创建计划任务失败: {e}")

# 示例：创建每日备份任务
# create_windows_task("DailyBackup", "C:\\scripts\\backup.py", "23:00")
```

#### 在 Linux/macOS 上使用 cron

```python
import subprocess
import getpass

def create_cron_job(schedule, command):
    """创建 cron 任务"""
    # 获取当前 crontab
    user = getpass.getuser()
    try:
        # 获取现有 crontab
        process = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = process.stdout
    except subprocess.CalledProcessError:
        current_crontab = ""
    
    # 添加新任务
    new_crontab = current_crontab
    if not new_crontab.endswith('\n'):
        new_crontab += '\n'
    new_crontab += f"{schedule} {command}\n"
    
    # 写入 crontab
    try:
        process = subprocess.run(['crontab', '-'], input=new_crontab, text=True)
        print(f"已为用户 {user} 创建 cron 任务")
    except subprocess.CalledProcessError as e:
        print(f"创建 cron 任务失败: {e}")

# 示例：创建定时备份任务 (每天 23:00 运行)
# create_cron_job("0 23 * * *", "/usr/bin/python3 /home/user/scripts/backup.py")
```

### 使用 APScheduler 进行高级调度

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)

def task_function():
    print(f"执行任务时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def apscheduler_demo():
    """APScheduler 示例"""
    # 创建调度器
    scheduler = BlockingScheduler()
    
    # 添加任务
    # 1. 间隔调度 - 每 5 秒执行一次
    scheduler.add_job(task_function, 'interval', seconds=5, id='interval_job')
    
    # 2. 定时调度 - 每天特定时间执行
    scheduler.add_job(task_function, 'cron', hour=12, minute=0, id='daily_job')
    
    # 3. 使用 CronTrigger - 更复杂的调度
    scheduler.add_job(
        task_function,
        CronTrigger(day_of_week='mon-fri', hour=9, minute=30),
        id='workday_job'
    )
    
    # 4. 一次性执行
    scheduler.add_job(
        task_function,
        'date',
        run_date=datetime(2023, 12, 31, 23, 59, 59),
        id='new_year_job'
    )
    
    # 启动调度器
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()

# 示例：在后台运行调度器
def background_scheduler_demo():
    scheduler = BackgroundScheduler()
    scheduler.add_job(task_function, 'interval', seconds=10)
    scheduler.start()
    
    print("后台调度器已启动. 按 Ctrl+C 停止...")
    try:
        # 主程序可以继续执行其他任务
        while True:
            time.sleep(2)
            print("主程序继续运行...")
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("调度器已停止")

# background_scheduler_demo()
```

## 办公自动化

自动化办公任务可以大幅提高工作效率，Python 提供了丰富的库来处理各类办公文档。

### 自动化 Excel 操作

```python
import pandas as pd
import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

def create_sales_report(data_file, output_file):
    """创建销售报表并添加图表"""
    # 读取数据
    df = pd.read_excel(data_file)
    
    # 按月份和产品分组汇总
    monthly_sales = df.pivot_table(
        index='Month', 
        columns='Product',
        values='Sales',
        aggfunc='sum'
    )
    
    # 创建工作簿
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "销售报表"
    
    # 添加数据
    for r in dataframe_to_rows(monthly_sales, index=True, header=True):
        ws.append(r)
    
    # 创建图表
    chart = BarChart()
    chart.title = "月度销售报表"
    chart.x_axis.title = "月份"
    chart.y_axis.title = "销售额"
    
    # 设置图表数据范围
    data = Reference(ws, min_col=2, min_row=1, max_col=ws.max_column, max_row=ws.max_row)
    categories = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)
    
    # 添加图表到工作表
    ws.add_chart(chart, "A15")
    
    # 保存结果
    wb.save(output_file)
    print(f"销售报表已创建并保存为 {output_file}")

# 示例：创建销售报表
# create_sales_report("sales_data.xlsx", "sales_report.xlsx")
```

### 自动化 Word 文档

```python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_report_template(company_name, report_title, output_file):
    """创建 Word 报告模板"""
    # 创建文档
    doc = Document()
    
    # 添加标题
    title = doc.add_heading(report_title, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 添加公司信息
    company_para = doc.add_paragraph()
    company_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    company_run = company_para.add_run(company_name)
    company_run.bold = True
    company_run.font.size = Pt(14)
    
    # 添加日期
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run("报告日期: ").bold = True
    date_para.add_run("XXXX年XX月XX日")
    
    # 添加目录项
    doc.add_heading("目录", level=1)
    for i, section in enumerate(["摘要", "引言", "方法", "结果", "讨论", "结论"]):
        doc.add_paragraph(f"{i+1}. {section}", style="List Number")
    
    # 添加各节内容
    for section in ["摘要", "引言", "方法", "结果", "讨论", "结论"]:
        doc.add_heading(section, level=1)
        doc.add_paragraph("在这里添加" + section + "内容。" * 5)
    
    # 添加表格示例
    doc.add_heading("示例表格", level=2)
    table = doc.add_table(rows=4, cols=3)
    table.style = "Table Grid"
    
    # 设置表头
    header_cells = table.rows[0].cells
    header_cells[0].text = "项目"
    header_cells[1].text = "描述"
    header_cells[2].text = "结果"
    
    # 填充表格数据
    for i in range(1, 4):
        row_cells = table.rows[i].cells
        row_cells[0].text = f"项目 {i}"
        row_cells[1].text = f"这是项目 {i} 的描述"
        row_cells[2].text = f"结果 {i}"
    
    # 添加图片位置
    doc.add_heading("示例图表", level=2)
    doc.add_paragraph("在下方插入图表:")
    
    # 添加页眉和页脚
    sections = doc.sections
    for section in sections:
        header = section.header
        header.paragraphs[0].text = f"{company_name} - {report_title}"
        
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.text = "页码: "
        footer_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # 保存文档
    doc.save(output_file)
    print(f"报告模板已创建并保存为 {output_file}")

# 示例：创建报告模板
# create_report_template("ABC科技有限公司", "年度研究报告", "report_template.docx")
```

### 自动化 PDF 处理

```python
from PyPDF2 import PdfReader, PdfWriter
import os

def merge_pdfs(input_files, output_file):
    """合并多个PDF文件"""
    pdf_writer = PdfWriter()
    
    # 添加每个PDF的所有页面
    for file in input_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
    
    # 保存合并后的PDF
    with open(output_file, 'wb') as f:
        pdf_writer.write(f)
    
    print(f"已将 {len(input_files)} 个PDF文件合并为 {output_file}")

def split_pdf(input_file, output_folder):
    """将PDF拆分为单页文件"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取PDF
    pdf_reader = PdfReader(input_file)
    
    # 拆分为单页
    for i, page in enumerate(pdf_reader.pages):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        
        # 创建输出文件名
        output_file = os.path.join(output_folder, f"page_{i+1}.pdf")
        
        # 保存单页文件
        with open(output_file, 'wb') as f:
            pdf_writer.write(f)
    
    print(f"已将 {input_file} 拆分为 {len(pdf_reader.pages)} 个单页文件，保存在 {output_folder}")

# 示例：合并PDF
# merge_pdfs(["file1.pdf", "file2.pdf", "file3.pdf"], "merged.pdf")

# 示例：拆分PDF
# split_pdf("document.pdf", "pages")
```

### 自动化邮件发送

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

def send_email(sender, password, recipient, subject, body, attachments=None, smtp_server="smtp.gmail.com", smtp_port=587):
    """发送电子邮件"""
    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient if isinstance(recipient, str) else ", ".join(recipient)
    msg['Subject'] = subject
    
    # 添加正文
    msg.attach(MIMEText(body, 'html'))
    
    # 添加附件
    if attachments:
        for file in attachments:
            with open(file, 'rb') as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(file))
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file)}"'
                msg.attach(part)
    
    # 连接到SMTP服务器
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 启用TLS加密
        server.login(sender, password)  # 登录
        
        # 发送邮件
        if isinstance(recipient, str):
            recipients = [recipient]
        else:
            recipients = recipient
            
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        
        print(f"邮件已成功发送给 {msg['To']}")
        return True
    except Exception as e:
        print(f"发送邮件时出错: {e}")
        return False

# 示例：发送带附件的邮件
# send_email(
#     sender="your_email@gmail.com",
#     password="your_password",  # 对于Gmail，请使用应用专用密码
#     recipient=["recipient1@example.com", "recipient2@example.com"],
#     subject="每月报告",
#     body="<h3>每月报告</h3><p>请查看附件中的报告。</p>",
#     attachments=["report.xlsx", "chart.pdf"]
# )
```

### 自动化图像处理

```python
from PIL import Image, ImageFilter, ImageEnhance
import os

def batch_process_images(input_folder, output_folder, operations=None, format='JPEG'):
    """批量处理图像"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 默认操作：调整大小、提高对比度、锐化
    if operations is None:
        operations = [
            lambda img: img.resize((800, 600), Image.LANCZOS),
            lambda img: ImageEnhance.Contrast(img).enhance(1.2),
            lambda img: img.filter(ImageFilter.SHARPEN)
        ]
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    files = [f for f in os.listdir(input_folder) 
             if os.path.splitext(f)[1].lower() in image_extensions]
    
    # 处理每个图像
    for file in files:
        input_path = os.path.join(input_folder, file)
        
        # 修改文件扩展名为目标格式
        filename = os.path.splitext(file)[0]
        output_path = os.path.join(output_folder, f"{filename}.{format.lower()}")
        
        try:
            # 打开图像
            img = Image.open(input_path)
            
            # 应用操作
            for operation in operations:
                img = operation(img)
            
            # 保存处理后的图像
            img.save(output_path, format)
            print(f"已处理 {file} 并保存为 {os.path.basename(output_path)}")
        
        except Exception as e:
            print(f"处理 {file} 时出错: {e}")
    
    print(f"批处理完成。已处理 {len(files)} 个文件。")

# 示例：批量处理图像
# batch_process_images(
#     "photos", 
#     "processed_photos",
#     operations=[
#         lambda img: img.resize((1024, 768), Image.LANCZOS),
#         lambda img: ImageEnhance.Brightness(img).enhance(1.1),
#         lambda img: ImageEnhance.Contrast(img).enhance(1.2),
#         lambda img: img.filter(ImageFilter.SHARPEN)
#     ],
#     format='PNG'
# )
```

### 自动化 GUI 操作

```python
import pyautogui
import time

# 设置安全区域，防止鼠标失控
pyautogui.FAILSAFE = True
# 操作间隔时间，防止操作过快
pyautogui.PAUSE = 0.5

def automate_data_entry(data_list):
    """自动化重复的数据输入任务"""
    print("准备开始自动化数据输入...")
    print("请在5秒内将光标放在第一个输入框上...")
    time.sleep(5)
    
    for data_item in data_list:
        # 输入数据
        pyautogui.typewrite(str(data_item["name"]))
        pyautogui.press('tab')
        pyautogui.typewrite(str(data_item["email"]))
        pyautogui.press('tab')
        pyautogui.typewrite(str(data_item["phone"]))
        
        # 点击提交按钮
        pyautogui.press('tab')
        pyautogui.press('enter')
        
        # 等待页面刷新
        time.sleep(2)
        
        print(f"已输入: {data_item['name']}")

def take_screenshot(output_file):
    """截取屏幕截图"""
    screenshot = pyautogui.screenshot()
    screenshot.save(output_file)
    print(f"截图已保存为 {output_file}")

def find_and_click_image(image_path, confidence=0.9):
    """查找并点击屏幕上的图像"""
    try:
        # 查找图像位置
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            # 获取中心点
            center = pyautogui.center(location)
            # 点击
            pyautogui.click(center)
            print(f"已点击位于 {center} 的图像")
            return True
        else:
            print(f"未找到图像: {image_path}")
            return False
    except Exception as e:
        print(f"查找图像时出错: {e}")
        return False

# 示例：自动化数据输入
# data = [
#     {"name": "张三", "email": "zhang@example.com", "phone": "13800000001"},
#     {"name": "李四", "email": "li@example.com", "phone": "13800000002"},
#     {"name": "王五", "email": "wang@example.com", "phone": "13800000003"}
# ]
# automate_data_entry(data)

# 示例：查找并点击图像
# find_and_click_image("submit_button.png")
```

## 系统自动化

Python 可以用于自动化各种系统任务和管理操作。

### 自动化系统监控

```python
import psutil
import platform
import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    filename='system_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_system_info():
    """获取系统信息"""
    info = {}
    info['platform'] = platform.system()
    info['platform-release'] = platform.release()
    info['platform-version'] = platform.version()
    info['architecture'] = platform.machine()
    info['processor'] = platform.processor()
    return info

def monitor_system_resources(interval=60, duration=3600):
    """监控系统资源使用情况"""
    end_time = time.time() + duration
    
    print(f"开始监控系统资源，间隔：{interval}秒，持续时间：{duration}秒")
    logging.info("开始系统监控")
    
    while time.time() < end_time:
        # 获取当前时间
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 收集资源使用情况
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 记录日志
        log_message = (f"时间: {now}, CPU: {cpu_percent}%, "
                       f"内存: {memory.percent}%, 磁盘: {disk.percent}%")
        logging.info(log_message)
        print(log_message)
        
        # 检查资源使用是否超过阈值
        if cpu_percent > 90:
            alert = f"警告: CPU 使用率过高: {cpu_percent}%"
            logging.warning(alert)
            print(alert)
        
        if memory.percent > 90:
            alert = f"警告: 内存使用率过高: {memory.percent}%"
            logging.warning(alert)
            print(alert)
        
        if disk.percent > 90:
            alert = f"警告: 磁盘使用率过高: {disk.percent}%"
            logging.warning(alert)
            print(alert)
        
        # 等待下一个监控间隔
        time.sleep(interval)
    
    logging.info("系统监控结束")
    print("系统监控结束")

# 示例：监控系统资源
# monitor_system_resources(interval=10, duration=60)  # 每10秒监控一次，持续60秒
```

## Python 3.13 新特性在自动化中的应用

Python 3.13 引入了一些新特性，可以提升自动化脚本的开发效率和运行性能。

### f-string 格式化增强

```python
import time

def performance_test():
    """测试 Python 3.13 f-string 性能优化"""
    data = {
        'name': 'Python 自动化',
        'version': 3.13,
        'features': ['更快的 f-string', '类型提示改进', '模式匹配增强']
    }
    
    start = time.time()
    for i in range(100000):
        result = f"项目: {data['name']}, 版本: {data['version']}, 特性数量: {len(data['features'])}"
    end = time.time()
    
    print(f"f-string 格式化 100,000 次用时: {end - start:.4f} 秒")

# 示例：性能测试
# performance_test()
```

### 优化的文件操作

```python
import pathlib

def path_operations():
    """演示 Python 3.13 优化的路径操作"""
    # 创建路径
    base_dir = pathlib.Path('/path/to/project')
    
    # 路径组合
    config_file = base_dir / 'config' / 'settings.json'
    
    # 检查路径是否存在
    if not config_file.exists():
        print(f"配置文件不存在: {config_file}")
    
    # 创建目录
    data_dir = base_dir / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取所有 Python 文件
    python_files = list(base_dir.glob('**/*.py'))
    print(f"找到 {len(python_files)} 个 Python 文件")
    
    # 读取文件内容
    if config_file.exists():
        content = config_file.read_text(encoding='utf-8')
        print(f"配置文件内容: {content[:100]}...")
    
    # 重命名文件
    old_file = data_dir / 'old.txt'
    new_file = data_dir / 'new.txt'
    if old_file.exists():
        old_file.rename(new_file)
        print(f"已重命名: {old_file} -> {new_file}")

# 示例：路径操作
# path_operations()
```

## 最佳实践

开发自动化脚本时，请遵循以下最佳实践：

1. **模块化设计**：将功能拆分为可重用的模块和函数
2. **错误处理**：使用 try-except 处理可能的异常
3. **日志记录**：添加适当的日志便于排查问题
4. **配置分离**：将配置与代码分离，便于调整
5. **测试验证**：先小范围测试，确认无误后再大规模应用
6. **版本控制**：使用 Git 等工具管理脚本版本
7. **文档注释**：添加良好的注释和文档，便于他人使用

### 常见问题的解决方案

- **脚本运行缓慢**：使用性能分析工具找出瓶颈，考虑多线程或异步处理
- **依赖问题**：使用虚拟环境和 requirements.txt 管理依赖
- **编码问题**：始终指定文件编码（如 UTF-8）
- **路径问题**：使用 os.path 或 pathlib 处理跨平台路径
- **权限问题**：检查文件和目录权限，使用 sudo 或管理员权限运行（如需要）

## 进一步学习

要深入学习 Python 自动化，可以探索以下领域：

1. **Web 自动化**：Selenium, Playwright
2. **API 自动化**：Requests, FastAPI
3. **测试自动化**：Pytest, Robot Framework
4. **CI/CD 自动化**：GitHub Actions, Jenkins
5. **容器自动化**：Docker, Kubernetes

## 结语

Python 自动化脚本可以显著提高工作效率，减少重复性工作。随着经验积累，您将能够自动化越来越复杂的任务，为自己和团队节省宝贵的时间。
