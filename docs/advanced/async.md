# Python 异步编程

异步编程是一种特殊的并发编程范式，它使用非阻塞 I/O 和事件循环来实现高并发。Python 3.4 引入了 `asyncio` 模块，3.5 引入了 `async/await` 语法，使异步编程变得更加直观和强大。本章将深入探讨 Python 异步编程的概念、语法和最佳实践。

## 异步编程核心概念

### 同步 vs 异步

- **同步编程**：代码按顺序执行，每个操作完成后才进行下一个操作。
- **异步编程**：允许在等待某个操作完成时继续执行其他操作，提高整体效率。

### 阻塞 vs 非阻塞

- **阻塞操作**：调用时会阻止程序继续执行，直到操作完成。
- **非阻塞操作**：调用后立即返回，允许程序继续执行其他操作。

### 事件循环

事件循环是异步编程的核心机制，负责：
- 监控异步操作
- 分派任务执行
- 处理回调函数

```python
import asyncio

# 获取事件循环
loop = asyncio.get_event_loop()

# Python 3.7+ 简化方法: asyncio.run()
```

### 协程 (Coroutines)

协程是异步编程的基本构建块，可以在执行中暂停和恢复：

```python
# Python 3.5+ 定义协程的方式
async def my_coroutine():
    print("协程开始")
    await asyncio.sleep(1)  # 非阻塞等待
    print("协程结束")
    return "完成"

# 执行协程
async def main():
    result = await my_coroutine()
    print(f"结果: {result}")

# Python 3.7+
asyncio.run(main())
```

### 任务 (Tasks)

任务是对协程的包装，让协程可以在事件循环中被调度执行：

```python
async def main():
    # 创建任务
    task = asyncio.create_task(my_coroutine())
    
    # 等待任务完成
    result = await task
    print(f"任务结果: {result}")

asyncio.run(main())
```

## 深入 async/await 语法

### async def - 定义协程函数

```python
# 定义协程函数
async def fetch_data():
    print("开始获取数据")
    await asyncio.sleep(2)  # 模拟 I/O 操作
    print("数据获取完成")
    return {"data": "这是一些数据"}
```

### await - 等待协程完成

`await` 表达式暂停当前协程的执行，直到被等待的对象完成：

```python
async def process_data():
    # 等待 fetch_data 协程完成
    data = await fetch_data()
    
    # 数据处理
    result = data["data"].upper()
    return result
```

### 可等待对象 (Awaitables)

在 `await` 表达式中可以使用以下类型的对象：

1. **协程**：由 `async def` 函数创建的对象
2. **任务**：由 `asyncio.create_task()` 创建的对象
3. **Future**：低级别可等待对象

```python
async def example():
    # 等待协程
    result1 = await fetch_data()
    
    # 等待任务
    task = asyncio.create_task(process_data())
    result2 = await task
    
    # 等待 Future
    future = asyncio.Future()
    asyncio.create_task(set_future_result(future))
    result3 = await future
    
    return result1, result2, result3

async def set_future_result(future):
    await asyncio.sleep(1)
    future.set_result("Future 结果")
```

## asyncio 模块详解

### 运行协程

```python
# Python 3.7+ 推荐方式
asyncio.run(main())

# 旧版本
loop = asyncio.get_event_loop()
result = loop.run_until_complete(main())
loop.close()
```

### 创建和管理任务

```python
async def main():
    # 创建任务
    task1 = asyncio.create_task(my_coroutine("任务1"))
    task2 = asyncio.create_task(my_coroutine("任务2"))
    
    # 获取所有任务
    all_tasks = asyncio.all_tasks()
    print(f"正在运行的任务数: {len(all_tasks)}")
    
    # 等待特定任务完成
    await task1
    await task2
    
    # 或者等待所有任务完成
    await asyncio.gather(task1, task2)
```

### 并发执行协程

`asyncio.gather()` 并发运行多个协程：

```python
async def main():
    # 并发执行多个协程
    results = await asyncio.gather(
        fetch_data("URL1"),
        fetch_data("URL2"),
        fetch_data("URL3")
    )
    
    # 处理结果
    for i, result in enumerate(results):
        print(f"结果 {i+1}: {result}")
```

### 超时控制

```python
async def main():
    try:
        # 设置超时
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
        print(f"操作结果: {result}")
    except asyncio.TimeoutError:
        print("操作超时")
```

### Shield

`asyncio.shield()` 保护任务不被取消：

```python
async def main():
    task = asyncio.create_task(long_running_task())
    
    # 保护任务不被取消
    try:
        shielded = asyncio.shield(task)
        await asyncio.wait_for(shielded, timeout=1)
    except asyncio.TimeoutError:
        print("Shield 超时，但任务继续运行")
        result = await task
        print(f"任务最终完成: {result}")
```

### 等待多个协程

`asyncio.wait()` 提供比 `gather()` 更灵活的等待机制：

```python
async def main():
    tasks = [
        asyncio.create_task(my_coroutine("Task1")),
        asyncio.create_task(my_coroutine("Task2")),
        asyncio.create_task(my_coroutine("Task3"))
    ]
    
    # 等待所有任务完成
    done, pending = await asyncio.wait(tasks)
    
    # 等待第一个任务完成
    done, pending = await asyncio.wait(
        tasks, 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    # 带超时的等待
    done, pending = await asyncio.wait(
        tasks,
        timeout=2.0
    )
    
    # 取消剩余任务
    for task in pending:
        task.cancel()
```

## 异步迭代和异步上下文管理器

### 异步迭代器

通过实现 `__aiter__` 和 `__anext__` 方法创建异步迭代器：

```python
class AsyncIterator:
    def __init__(self, limit):
        self.limit = limit
        self.counter = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.counter < self.limit:
            self.counter += 1
            await asyncio.sleep(0.1)  # 模拟异步操作
            return self.counter
        else:
            raise StopAsyncIteration

async def main():
    # 使用异步迭代器
    async for i in AsyncIterator(5):
        print(f"异步迭代值: {i}")
```

### 异步生成器

使用 `async def` 函数和 `yield` 创建异步生成器：

```python
async def async_generator(limit):
    for i in range(limit):
        await asyncio.sleep(0.1)  # 模拟异步操作
        yield i

async def main():
    # 使用异步生成器
    async for i in async_generator(5):
        print(f"生成的值: {i}")
```

### 异步上下文管理器

通过实现 `__aenter__` 和 `__aexit__` 方法创建异步上下文管理器：

```python
class AsyncResource:
    async def __aenter__(self):
        print("获取异步资源")
        await asyncio.sleep(0.1)  # 模拟异步获取资源
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("释放异步资源")
        await asyncio.sleep(0.1)  # 模拟异步释放资源
        return False  # 不吞没异常

async def main():
    # 使用异步上下文管理器
    async with AsyncResource() as resource:
        print("正在使用异步资源")
        await asyncio.sleep(0.5)
```

## 实用异步库

### aiohttp - 异步 HTTP 客户端/服务器

```python
import aiohttp
import asyncio

async def fetch(url):
    """异步获取URL内容"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        "https://www.python.org",
        "https://www.google.com",
        "https://github.com"
    ]
    
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for url, html in zip(urls, results):
        print(f"{url}: {len(html)} 字符")

asyncio.run(main())
```

### aiofiles - 异步文件操作

```python
import aiofiles
import asyncio

async def read_write_file():
    # 异步写入文件
    async with aiofiles.open('test_file.txt', 'w') as f:
        await f.write('Hello, 异步文件操作!\n')
        await f.write('这是第二行内容。')
    
    # 异步读取文件
    async with aiofiles.open('test_file.txt', 'r') as f:
        content = await f.read()
        print(f"文件内容: {content}")

asyncio.run(read_write_file())
```

### aiomysql/asyncpg - 异步数据库操作

```python
import asyncio
import asyncpg

async def test_db():
    # 连接到 PostgreSQL 数据库
    conn = await asyncpg.connect(
        user='user',
        password='password',
        database='database',
        host='localhost'
    )
    
    # 创建表
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            name TEXT,
            age INTEGER
        )
    ''')
    
    # 插入数据
    await conn.execute(
        'INSERT INTO users(name, age) VALUES($1, $2)',
        '张三',
        30
    )
    
    # 查询数据
    rows = await conn.fetch('SELECT * FROM users')
    for row in rows:
        print(f"用户: {row['name']}, 年龄: {row['age']}")
    
    # 关闭连接
    await conn.close()

# asyncio.run(test_db())  # 取消注释以运行
```

## 异步编程模式与最佳实践

### 异步编程的常见模式

#### 生产者/消费者模式

```python
import asyncio
import random

async def producer(queue):
    """生产者：生成数据并放入队列"""
    for i in range(5):
        item = random.randint(1, 100)
        await queue.put(item)
        print(f"生产: {item}")
        await asyncio.sleep(0.5)
    
    # 发送结束信号
    await queue.put(None)

async def consumer(queue):
    """消费者：从队列获取数据并处理"""
    while True:
        item = await queue.get()
        if item is None:  # 结束信号
            break
        
        print(f"消费: {item}")
        await asyncio.sleep(1)  # 模拟处理时间
        queue.task_done()

async def main():
    # 创建队列
    queue = asyncio.Queue()
    
    # 并发运行生产者和消费者
    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue))
    
    # 等待生产者完成
    await producer_task
    
    # 等待消费者完成
    await consumer_task

asyncio.run(main())
```

#### 并发限制

```python
import asyncio
import aiohttp

async def fetch_with_semaphore(semaphore, url):
    """带并发限制的网页获取"""
    async with semaphore:
        print(f"获取: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

async def main():
    # 限制最多同时运行 5 个请求
    semaphore = asyncio.Semaphore(5)
    
    # 生成多个URL
    urls = [f"https://example.com/{i}" for i in range(20)]
    
    # 创建任务
    tasks = [fetch_with_semaphore(semaphore, url) for url in urls]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    print(f"完成了 {len(results)} 个请求")

# asyncio.run(main())  # 取消注释以运行
```

#### 定时任务

```python
import asyncio
import time

async def periodic_task():
    """每隔一段时间执行的定时任务"""
    while True:
        print(f"定时任务执行: {time.strftime('%H:%M:%S')}")
        await asyncio.sleep(2)  # 每 2 秒执行一次

async def main():
    # 创建定时任务
    task = asyncio.create_task(periodic_task())
    
    # 让定时任务运行 10 秒
    await asyncio.sleep(10)
    
    # 取消定时任务
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("定时任务已取消")

asyncio.run(main())
```

### 错误处理和异常管理

```python
async def might_fail(name, should_fail=False):
    """可能会失败的协程"""
    print(f"{name} 开始")
    await asyncio.sleep(0.5)
    
    if should_fail:
        raise ValueError(f"{name} 失败")
    
    return f"{name} 成功"

async def main():
    # 处理单个协程的异常
    try:
        result = await might_fail("任务1", should_fail=True)
        print(result)
    except ValueError as e:
        print(f"捕获到异常: {e}")
    
    # 处理并发协程的异常
    tasks = [
        might_fail("任务2"),
        might_fail("任务3", should_fail=True),
        might_fail("任务4")
    ]
    
    # 方法1: gather 带 return_exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i+2} 出错: {result}")
        else:
            print(f"任务 {i+2} 结果: {result}")
    
    # 方法2: 独立捕获每个任务的异常
    for i, task_func in enumerate(tasks):
        try:
            result = await task_func
            print(f"独立任务 {i} 结果: {result}")
        except Exception as e:
            print(f"独立任务 {i} 错误: {e}")

asyncio.run(main())
```

### 调试技巧

```python
import asyncio
import logging

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 启用 asyncio 调试
# 也可以设置环境变量: PYTHONASYNCIODEBUG=1
asyncio.get_event_loop().set_debug(True)

async def debug_task():
    """演示调试特性的任务"""
    logging.debug("任务开始")
    await asyncio.sleep(0.1)
    
    # 故意等待太久
    await asyncio.sleep(0.5)
    
    logging.debug("任务结束")

async def main():
    # 创建任务
    task = asyncio.create_task(debug_task())
    
    # 检查任务的调用栈
    await asyncio.sleep(0.2)
    task_stack = task.get_stack()
    print(f"任务调用栈帧数: {len(task_stack)}")
    
    # 等待任务完成
    await task

asyncio.run(main())
```

## 实际应用示例

### 异步网络爬虫

```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import time

async def fetch_url(session, url):
    """异步获取URL内容"""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
            return None
    except Exception as e:
        print(f"获取 {url} 时出错: {e}")
        return None

async def parse_links(html, base_url):
    """解析HTML中的链接"""
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            links.append(href)
        elif href.startswith('/'):
            links.append(f"{base_url}{href}")
    
    return links[:5]  # 限制链接数量

async def crawl(start_url, max_depth=2):
    """异步爬取网页"""
    visited = set()
    
    async with aiohttp.ClientSession() as session:
        async def _crawl(url, depth):
            if depth > max_depth or url in visited:
                return
            
            visited.add(url)
            print(f"爬取 ({depth}): {url}")
            
            html = await fetch_url(session, url)
            links = await parse_links(html, url)
            
            # 递归爬取链接
            tasks = []
            for link in links:
                if link not in visited:
                    task = asyncio.create_task(_crawl(link, depth + 1))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
        
        await _crawl(start_url, 1)
    
    return visited

async def main():
    start_time = time.time()
    
    # 从Python官网开始爬取
    start_url = "https://python.org"
    visited = await crawl(start_url, max_depth=2)
    
    elapsed = time.time() - start_time
    print(f"\n爬取完成! 访问了 {len(visited)} 个URL")
    print(f"耗时: {elapsed:.2f} 秒")

# 运行爬虫
# asyncio.run(main())  # 取消注释以运行
```

### 异步 API 服务器

使用 `aiohttp.web` 实现的简单异步 API 服务器：

```python
from aiohttp import web
import asyncio
import json

# 模拟数据库
USERS = {}

# 异步处理函数
async def handle_get_users(request):
    """获取所有用户"""
    await asyncio.sleep(0.1)  # 模拟数据库查询
    return web.json_response(USERS)

async def handle_get_user(request):
    """获取单个用户"""
    user_id = request.match_info.get('id')
    await asyncio.sleep(0.1)  # 模拟数据库查询
    
    if user_id in USERS:
        return web.json_response(USERS[user_id])
    return web.json_response({"error": "User not found"}, status=404)

async def handle_create_user(request):
    """创建用户"""
    data = await request.json()
    
    if 'id' not in data or 'name' not in data:
        return web.json_response(
            {"error": "id and name are required"},
            status=400
        )
    
    user_id = str(data['id'])
    USERS[user_id] = data
    
    await asyncio.sleep(0.1)  # 模拟数据库写入
    return web.json_response(data, status=201)

# 创建应用
app = web.Application()
app.add_routes([
    web.get('/users', handle_get_users),
    web.get('/users/{id}', handle_get_user),
    web.post('/users', handle_create_user),
])

if __name__ == '__main__':
    web.run_app(app, port=8080)
```

## 异步编程的优缺点

### 优点

1. **高 I/O 并发**：能够同时处理成千上万的连接
2. **资源占用少**：相比线程和进程，协程占用的资源更少
3. **代码复杂度降低**：避免回调地狱，使代码更线性和可读

### 缺点

1. **CPU 密集型任务不适合**：异步最适合 I/O 密集型任务
2. **学习曲线**：需要理解事件循环和协程的工作方式
3. **生态系统转换**：许多库需要异步版本才能在 asyncio 中使用
4. **调试复杂**：异步代码的调试相对更复杂

## Python 3.11+ 异步特性

Python 3.11 和 3.13 引入了一些新的异步编程特性：

### 任务组 (Python 3.11+)

```python
import asyncio

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(asyncio.sleep(1))
        task2 = tg.create_task(asyncio.sleep(2))
        print("所有任务已创建")
    
    # TaskGroup 退出时，所有任务都已完成
    print("所有任务已完成")
    
    # 如果任何任务引发异常，异常会在这里传播

asyncio.run(main())
```

### asyncio.timeout (Python 3.11+)

```python
import asyncio

async def main():
    try:
        # 使用新的 asyncio.timeout 上下文管理器
        async with asyncio.timeout(1.0):
            await asyncio.sleep(2.0)  # 这将超时
            print("不会执行到这里")
    except TimeoutError:
        print("操作超时")

asyncio.run(main())
```

## 下一步

掌握了异步编程的基础后，您可以进一步探索 Python 的高级特性，例如元编程。请查看 [Python 元编程](/advanced/metaprogramming) 学习更多内容。 