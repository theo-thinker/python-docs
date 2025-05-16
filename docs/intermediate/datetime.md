# Python 日期与时间

日期和时间是编程中的常见数据类型，Python 提供了多种处理日期、时间、时区和时间间隔的模块。本章将介绍 Python 中处理日期和时间的主要方法。

## datetime 模块

`datetime` 模块是 Python 处理日期和时间的主要模块，它提供了多个类来表示和操作日期和时间。

### datetime 类

`datetime` 类表示日期和时间的组合：

```python
from datetime import datetime

# 获取当前日期和时间
now = datetime.now()
print(f"当前日期和时间: {now}")  # 例如: 2023-07-14 15:30:45.123456

# 创建指定的 datetime 对象
dt = datetime(2023, 7, 14, 15, 30, 45)
print(dt)  # 输出: 2023-07-14 15:30:45

# 获取日期和时间的各个部分
print(f"年: {dt.year}")    # 2023
print(f"月: {dt.month}")   # 7
print(f"日: {dt.day}")     # 14
print(f"时: {dt.hour}")    # 15
print(f"分: {dt.minute}")  # 30
print(f"秒: {dt.second}")  # 45
print(f"微秒: {dt.microsecond}")  # 0
```

### date 类

`date` 类表示日期（年、月、日）：

```python
from datetime import date

# 获取当前日期
today = date.today()
print(f"今天: {today}")  # 例如: 2023-07-14

# 创建指定的 date 对象
d = date(2023, 7, 14)
print(d)  # 输出: 2023-07-14

# 获取日期的各个部分
print(f"年: {d.year}")   # 2023
print(f"月: {d.month}")  # 7
print(f"日: {d.day}")    # 14

# 获取星期几 (0 表示星期一，6 表示星期日)
print(f"星期几: {d.weekday()}")  # 0-6
print(f"ISO 星期几: {d.isoweekday()}")  # 1-7 (1 表示星期一)

# 从 datetime 获取 date
dt = datetime.now()
d = dt.date()
print(f"当前日期: {d}")
```

### time 类

`time` 类表示一天中的时间（时、分、秒、微秒）：

```python
from datetime import time

# 创建时间对象
t = time(15, 30, 45, 123456)
print(t)  # 输出: 15:30:45.123456

# 获取时间的各个部分
print(f"时: {t.hour}")    # 15
print(f"分: {t.minute}")  # 30
print(f"秒: {t.second}")  # 45
print(f"微秒: {t.microsecond}")  # 123456

# 从 datetime 获取 time
dt = datetime.now()
t = dt.time()
print(f"当前时间: {t}")
```

### timedelta 类

`timedelta` 类表示两个日期、时间或日期时间之间的差异：

```python
from datetime import datetime, timedelta

# 创建 timedelta
delta = timedelta(days=5, hours=2, minutes=30)
print(delta)  # 输出: 5 days, 2:30:00

# 日期和时间的加减法
now = datetime.now()
future = now + delta
past = now - delta

print(f"当前: {now}")
print(f"5天2小时30分钟后: {future}")
print(f"5天2小时30分钟前: {past}")

# 计算两个日期之间的差异
dt1 = datetime(2023, 7, 14, 12, 0, 0)
dt2 = datetime(2023, 7, 20, 14, 30, 0)
diff = dt2 - dt1
print(f"相差: {diff}")  # 输出: 6 days, 2:30:00
print(f"相差的天数: {diff.days}")  # 6
print(f"相差的总秒数: {diff.total_seconds()}")  # 536400.0

# 使用 timedelta 创建各种时间间隔
one_day = timedelta(days=1)
one_week = timedelta(weeks=1)
one_hour = timedelta(hours=1)
ten_minutes = timedelta(minutes=10)
half_second = timedelta(milliseconds=500)
```

## 日期和时间的格式化

### 日期时间的字符串表示

`datetime` 对象可以转换为不同格式的字符串：

```python
from datetime import datetime

now = datetime.now()

# 使用 strftime() 方法格式化日期时间
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"格式化的日期时间: {formatted}")  # 例如: 2023-07-14 15:30:45

# 各种格式化选项
print(now.strftime("%Y年%m月%d日 %H时%M分%S秒"))  # 例如: 2023年07月14日 15时30分45秒
print(now.strftime("%A, %B %d, %Y"))  # 例如: Friday, July 14, 2023
print(now.strftime("%a, %b %d, %Y"))  # 例如: Fri, Jul 14, 2023
print(now.strftime("%Y-%m-%d"))      # 例如: 2023-07-14
print(now.strftime("%H:%M:%S"))      # 例如: 15:30:45
print(now.strftime("%I:%M %p"))      # 例如: 03:30 PM
```

### 格式化指令

以下是一些常用的格式化指令：

| 指令 | 意义 | 示例 |
|------|------|------|
| `%Y` | 四位数年份 | 2023 |
| `%y` | 两位数年份 | 23 |
| `%m` | 月份（01-12） | 07 |
| `%d` | 日（01-31） | 14 |
| `%H` | 24小时制小时（00-23） | 15 |
| `%I` | 12小时制小时（01-12） | 03 |
| `%M` | 分钟（00-59） | 30 |
| `%S` | 秒（00-59） | 45 |
| `%f` | 微秒（000000-999999） | 123456 |
| `%p` | AM/PM | PM |
| `%a` | 星期的缩写 | Fri |
| `%A` | 星期的完整名称 | Friday |
| `%b` | 月份的缩写 | Jul |
| `%B` | 月份的完整名称 | July |
| `%j` | 一年中的第几天（001-366） | 195 |
| `%U` | 一年中的第几周（00-53，周日为一周的第一天） | 28 |
| `%W` | 一年中的第几周（00-53，周一为一周的第一天） | 28 |
| `%c` | 本地日期和时间表示 | Fri Jul 14 15:30:45 2023 |
| `%x` | 本地日期表示 | 07/14/23 |
| `%X` | 本地时间表示 | 15:30:45 |
| `%%` | 字面的 % 字符 | % |

### 解析日期和时间字符串

使用 `strptime()` 方法可以将字符串解析为 `datetime` 对象：

```python
from datetime import datetime

# 解析日期时间字符串
date_string = "2023-07-14 15:30:45"
dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
print(dt)  # 输出: 2023-07-14 15:30:45

# 解析不同格式的字符串
date_string = "14/07/2023"
dt = datetime.strptime(date_string, "%d/%m/%Y")
print(dt)  # 输出: 2023-07-14 00:00:00

date_string = "July 14, 2023"
dt = datetime.strptime(date_string, "%B %d, %Y")
print(dt)  # 输出: 2023-07-14 00:00:00

date_string = "7/14/23 3:30 PM"
dt = datetime.strptime(date_string, "%m/%d/%y %I:%M %p")
print(dt)  # 输出: 2023-07-14 15:30:00
```

## 时区处理

Python 提供了多种处理时区的方法，包括 `datetime` 模块的 `tzinfo` 类和 `pytz` 第三方库。

### 使用 datetime.timezone

在 Python 3.2+ 中，可以使用内置的 `timezone` 类表示简单的 UTC 偏移时区：

```python
from datetime import datetime, timezone, timedelta

# 创建 UTC 时间
dt_utc = datetime.now(timezone.utc)
print(f"UTC 时间: {dt_utc}")

# 创建带有固定偏移的时区
tz_shanghai = timezone(timedelta(hours=8))  # UTC+8
dt_shanghai = datetime.now(tz_shanghai)
print(f"上海时间: {dt_shanghai}")

# 转换时区
dt_nyc = dt_shanghai.astimezone(timezone(timedelta(hours=-4)))  # UTC-4
print(f"纽约时间: {dt_nyc}")
```

### 使用 pytz 库

`pytz` 库提供了更完整的时区数据库，支持世界各地的时区和夏令时调整。需要使用 `pip install pytz` 安装：

```python
from datetime import datetime
import pytz

# 获取所有可用时区
all_timezones = pytz.all_timezones
print(f"时区数量: {len(all_timezones)}")
print(f"部分时区: {all_timezones[:5]}")

# 使用特定时区
tz_shanghai = pytz.timezone('Asia/Shanghai')
tz_nyc = pytz.timezone('America/New_York')

# 创建带时区的 datetime
dt_shanghai = datetime.now(tz_shanghai)
print(f"上海时间: {dt_shanghai}")

# 转换时区
dt_nyc = dt_shanghai.astimezone(tz_nyc)
print(f"纽约时间: {dt_nyc}")

# 本地化 naive datetime
naive = datetime.now()
dt_shanghai = tz_shanghai.localize(naive)
print(f"本地化后的上海时间: {dt_shanghai}")

# 使用 UTC
utc = pytz.UTC
dt_utc = datetime.now(utc)
print(f"UTC 时间: {dt_utc}")
```

### Python 3.9+ zoneinfo 模块

从 Python 3.9 开始，标准库增加了 `zoneinfo` 模块，提供与 `pytz` 类似的功能：

```python
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# 创建带时区的 datetime
dt_shanghai = datetime.now(ZoneInfo("Asia/Shanghai"))
print(f"上海时间: {dt_shanghai}")

# 转换时区
dt_nyc = dt_shanghai.astimezone(ZoneInfo("America/New_York"))
print(f"纽约时间: {dt_nyc}")

# 查看可用时区
from zoneinfo import available_timezones
zones = list(available_timezones())
print(f"时区数量: {len(zones)}")
print(f"部分时区: {zones[:5]}")
```

## ISO 格式和时间戳

### ISO 格式

ISO 8601 是一种标准化的日期和时间表示法：

```python
from datetime import datetime

# 获取当前时间的 ISO 格式
now = datetime.now()
iso_format = now.isoformat()
print(f"ISO 格式: {iso_format}")  # 例如: 2023-07-14T15:30:45.123456

# 带时区的 ISO 格式
now_utc = datetime.now(timezone.utc)
iso_format_tz = now_utc.isoformat()
print(f"带时区的 ISO 格式: {iso_format_tz}")  # 例如: 2023-07-14T15:30:45.123456+00:00

# 从 ISO 格式解析
dt = datetime.fromisoformat(iso_format_tz)
print(dt)
```

### 时间戳

Unix 时间戳表示从 1970 年 1 月 1 日 00:00:00 UTC 开始经过的秒数：

```python
from datetime import datetime
import time

# 获取当前时间戳
timestamp = time.time()
print(f"当前时间戳: {timestamp}")  # 例如: 1689342645.123456

# 从时间戳创建 datetime
dt = datetime.fromtimestamp(timestamp)
print(f"从时间戳创建的本地时间: {dt}")

# 创建 UTC 时间
dt_utc = datetime.fromtimestamp(timestamp, timezone.utc)
print(f"从时间戳创建的 UTC 时间: {dt_utc}")

# 将 datetime 转换为时间戳
timestamp = dt.timestamp()
print(f"从 datetime 获取的时间戳: {timestamp}")
```

## time 模块

除了 `datetime` 模块，Python 还提供了 `time` 模块，用于获取当前时间、执行时间测量和时间转换：

```python
import time

# 获取当前时间戳（秒）
timestamp = time.time()
print(f"当前时间戳: {timestamp}")

# 获取可读的时间字符串
time_string = time.ctime(timestamp)
print(f"可读时间: {time_string}")  # 例如: Fri Jul 14 15:30:45 2023

# 获取 struct_time 对象
time_struct = time.localtime(timestamp)
print(f"本地时间结构: {time_struct}")

# 从 struct_time 创建格式化的时间字符串
formatted = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
print(f"格式化时间: {formatted}")

# 将时间字符串解析为 struct_time
time_struct = time.strptime("2023-07-14 15:30:45", "%Y-%m-%d %H:%M:%S")
print(f"解析后的时间结构: {time_struct}")

# 休眠（暂停执行）
print("开始休眠 1 秒...")
time.sleep(1)
print("休眠结束")

# 测量代码执行时间
start = time.time()
# 执行一些操作...
time.sleep(0.5)  # 模拟操作
end = time.time()
print(f"操作耗时: {end - start:.6f} 秒")

# 更精确的计时（推荐用于性能测量）
import time
start = time.perf_counter()
# 执行一些操作...
time.sleep(0.5)  # 模拟操作
end = time.perf_counter()
print(f"操作精确耗时: {end - start:.9f} 秒")
```

## calendar 模块

`calendar` 模块提供了与日历相关的功能：

```python
import calendar

# 检查是否为闰年
is_leap = calendar.isleap(2024)
print(f"2024 是闰年: {is_leap}")  # True

# 获取指定月份的日历
cal = calendar.month(2023, 7)
print(f"2023年7月的日历:\n{cal}")

# 获取指定年份的日历
cal = calendar.calendar(2023)
print(f"2023年的日历:\n{cal}")

# 计算某月的天数
days = calendar.monthrange(2023, 7)[1]
print(f"2023年7月有 {days} 天")

# 计算某天是星期几（0 是星期一，6 是星期日）
weekday = calendar.weekday(2023, 7, 14)
print(f"2023年7月14日是星期 {weekday + 1}")

# 设置每周的第一天（0 是星期一，6 是星期日）
calendar.setfirstweekday(calendar.SUNDAY)
cal = calendar.month(2023, 7)
print(f"以星期日为第一天的日历:\n{cal}")
```

## 实际应用示例

### 计算日期差异

```python
from datetime import datetime, timedelta

def calculate_age(birth_date):
    """计算年龄"""
    today = datetime.now().date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# 计算年龄
birth_date = datetime(1990, 5, 15).date()
age = calculate_age(birth_date)
print(f"年龄: {age} 岁")

# 计算两个日期之间的工作日
def business_days_between(start_date, end_date):
    """计算两个日期之间的工作日数量（不包括周末）"""
    if start_date > end_date:
        return 0
    
    days = 0
    current = start_date
    while current <= end_date:
        # 0 代表星期一，6 代表星期日
        if current.weekday() < 5:  # 0-4 是工作日
            days += 1
        current += timedelta(days=1)
    return days

start = datetime(2023, 7, 1).date()
end = datetime(2023, 7, 31).date()
days = business_days_between(start, end)
print(f"工作日数量: {days} 天")
```

### 创建日期范围

```python
from datetime import datetime, timedelta

def date_range(start_date, end_date, step=1):
    """生成日期范围"""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=step)

# 生成一个月的日期
start = datetime(2023, 7, 1).date()
end = datetime(2023, 7, 31).date()

print("七月的所有日期:")
for date in date_range(start, end):
    print(date.strftime("%Y-%m-%d"))

# 每周一的日期
print("\n七月的每个星期一:")
for date in date_range(start, end):
    if date.weekday() == 0:  # 星期一
        print(date.strftime("%Y-%m-%d"))
```

### 日期时间的算术运算

```python
from datetime import datetime, timedelta

now = datetime.now()
print(f"当前时间: {now}")

# 一天后
one_day_later = now + timedelta(days=1)
print(f"一天后: {one_day_later}")

# 一周后
one_week_later = now + timedelta(weeks=1)
print(f"一周后: {one_week_later}")

# 一小时前
one_hour_ago = now - timedelta(hours=1)
print(f"一小时前: {one_hour_ago}")

# 增加一个月（注意：日历月份的长度不同）
def add_month(date_obj, months):
    """向日期添加指定的月数"""
    month = date_obj.month - 1 + months
    year = date_obj.year + month // 12
    month = month % 12 + 1
    day = min(date_obj.day, calendar.monthrange(year, month)[1])
    return date_obj.replace(year=year, month=month, day=day)

one_month_later = add_month(now, 1)
print(f"一个月后: {one_month_later}")
```

### 定期任务调度

```python
import time
import datetime
import threading

def scheduled_task(interval_seconds, task_func, *args, **kwargs):
    """以固定间隔执行任务"""
    next_time = time.time() + interval_seconds
    while True:
        time.sleep(max(0, next_time - time.time()))
        try:
            task_func(*args, **kwargs)
        except Exception as e:
            print(f"任务执行出错: {e}")
        next_time += interval_seconds

def print_time():
    """打印当前时间"""
    print(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 每 2 秒执行一次任务
thread = threading.Thread(target=scheduled_task, args=(2, print_time))
thread.daemon = True  # 让线程随主程序退出而退出

print("开始定期任务...")
thread.start()

# 主程序继续运行一段时间后退出
time.sleep(10)
print("主程序退出")
```

## 日期时间的性能优化

处理大量日期时间数据时，可以采用一些优化技巧：

```python
import datetime
import time

# 1. 使用 datetime.timestamp() 和 datetime.fromtimestamp() 进行快速转换
def timestamp_optimization():
    start = time.perf_counter()
    
    now = datetime.datetime.now()
    timestamps = []
    for _ in range(1000000):
        timestamps.append(now.timestamp())
    
    end = time.perf_counter()
    print(f"生成 100 万个时间戳: {end - start:.6f} 秒")
    
    start = time.perf_counter()
    
    dates = []
    for ts in timestamps[:1000]:  # 仅转换前 1000 个以节省时间
        dates.append(datetime.datetime.fromtimestamp(ts))
    
    end = time.perf_counter()
    print(f"从 1000 个时间戳创建日期: {end - start:.6f} 秒")

# 2. 日期解析优化：避免在循环中重复解析相同格式的日期
def parsing_optimization():
    date_strings = ["2023-07-14"] * 10000
    
    # 慢：每次循环创建新的解析器
    start = time.perf_counter()
    dates = []
    for date_str in date_strings:
        dates.append(datetime.datetime.strptime(date_str, "%Y-%m-%d"))
    end = time.perf_counter()
    print(f"常规解析 10000 个日期: {end - start:.6f} 秒")
    
    # 更快：使用预编译的正则表达式（需要更多代码）
    import re
    pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
    
    start = time.perf_counter()
    dates = []
    for date_str in date_strings:
        match = pattern.match(date_str)
        if match:
            year, month, day = map(int, match.groups())
            dates.append(datetime.datetime(year, month, day))
    end = time.perf_counter()
    print(f"使用正则表达式解析 10000 个日期: {end - start:.6f} 秒")

# 执行优化测试
timestamp_optimization()
parsing_optimization()
```

## 最佳实践

1. **使用 ISO 8601 格式**：存储和交换日期时，尽可能使用 ISO 8601 格式 (`YYYY-MM-DDTHH:MM:SS±HH:MM`)，这是一种国际标准。

2. **始终处理时区**：在处理跨时区的日期时间时，始终确保正确处理时区信息，避免时区相关的错误。

3. **使用 `with` 语句处理时区变换**：使用 `pytz` 时，注意使用 `localize()` 而不是直接将 `tzinfo` 添加到 naive datetime。

4. **使用 `timedelta` 进行日期运算**：对于日期的加减，使用 `timedelta` 对象，而不是手动计算。

5. **小心处理闰年和闰秒**：计算日期差异时，要考虑闰年和可能的闰秒。

6. **格式化时使用 `strftime()` 和 `strptime()`**：对于日期的格式化和解析，使用 `strftime()` 和 `strptime()`，而不是手动拼接字符串。

## 下一步

现在您已经掌握了 Python 中日期和时间处理的基础知识，接下来可以探索 [Python 数据结构](/intermediate/data-structures)，学习如何处理各种高级数据结构以便更有效地组织和处理程序中的数据。 