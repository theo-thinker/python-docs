# Python 并发编程

并发编程允许程序同时或看似同时地执行多项任务，从而充分利用现代计算机的多核处理能力。Python 提供了多种处理并发的方式，本章将介绍 Python 中的主要并发编程模型。

## 并发与并行

在开始之前，我们需要明确两个概念：

- **并发(Concurrency)**: 指程序的不同部分可以"同时"(不一定是真正同时)执行，互不干扰。
- **并行(Parallelism)**: 指程序的不同部分在物理上真正同时执行(如利用多核 CPU)。

Python 中的并发模型主要有：

1. **多线程(Threading)**: 共享内存的并发执行
2. **多进程(Multiprocessing)**: 分离内存的并发执行
3. **异步 IO(Asyncio)**: 单线程协程，事件驱动的并发

## Python 的全局解释器锁 (GIL)

理解 Python 并发编程的关键是了解全局解释器锁(Global Interpreter Lock, GIL)：

- GIL 是 CPython 解释器的一个互斥锁，确保同一时刻只有一个线程执行 Python 字节码
- 这意味着 Python 的多线程不能在多核上实现真正的并行计算
- GIL 对 I/O 密集型任务影响不大，但对 CPU 密集型任务影响显著
- 多进程可以绕过 GIL 限制，但有更高的资源开销

## 多线程编程

Python 的 `threading` 模块提供了多线程支持。多线程适合 I/O 密集型任务(如网络请求、文件操作)。

### 基本线程用法

```python
import threading
import time

def worker(name):
    """线程工作函数"""
    print(f"线程 {name} 开始工作")
    time.sleep(2)  # 模拟 I/O 操作
    print(f"线程 {name} 工作完成")

# 创建两个线程
t1 = threading.Thread(target=worker, args=("T1",))
t2 = threading.Thread(target=worker, args=("T2",))

# 启动线程
t1.start()
t2.start()

# 等待线程完成
t1.join()
t2.join()

print("所有线程已完成")
```

### 线程安全与同步

由于线程共享内存，需要特别注意共享数据的安全访问：

```python
import threading

# 共享资源
counter = 0
lock = threading.Lock()  # 互斥锁

def increment_counter(count):
    global counter
    for _ in range(count):
        # 使用锁保护共享资源
        with lock:
            counter += 1

# 创建两个线程，都修改 counter
t1 = threading.Thread(target=increment_counter, args=(10000,))
t2 = threading.Thread(target=increment_counter, args=(10000,))

t1.start()
t2.start()
t1.join()
t2.join()

print(f"最终计数: {counter}")  # 应该是 20000
```

### 更多线程同步原语

除了互斥锁，Python 还提供了其他同步原语：

```python
from threading import RLock, Semaphore, Event, Condition

# 可重入锁(RLock)：同一线程可多次获取
rlock = RLock()

# 信号量(Semaphore)：限制访问资源的线程数量
semaphore = Semaphore(2)  # 最多允许两个线程同时访问

# 事件(Event)：线程间发送信号
event = Event()

# 条件变量(Condition)：复杂的线程间通信
condition = Condition()
```

### 线程池

使用 `concurrent.futures` 模块的 `ThreadPoolExecutor` 可以方便地创建线程池：

```python
from concurrent.futures import ThreadPoolExecutor
import requests

# 线程池中的工作函数
def fetch_url(url):
    """获取URL内容"""
    response = requests.get(url)
    return f"{url}: {len(response.text)} 字符"

urls = [
    "https://www.python.org",
    "https://www.google.com",
    "https://www.github.com"
]

# 使用线程池处理多个URL请求
with ThreadPoolExecutor(max_workers=3) as executor:
    # 方法1：map
    for result in executor.map(fetch_url, urls):
        print(result)
    
    # 方法2：submit 和 as_completed
    from concurrent.futures import as_completed
    future_to_url = {executor.submit(fetch_url, url): url for url in urls}
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
            print(data)
        except Exception as e:
            print(f"{url} 生成错误: {e}")
```

## 多进程编程

Python 的 `multiprocessing` 模块提供了类似于 `threading` 模块的 API，但基于进程而非线程。多进程适合 CPU 密集型任务，因为它可以绕过 GIL 限制。

### 基本进程用法

```python
import multiprocessing
import time

def worker(name):
    """进程工作函数"""
    print(f"进程 {name} 开始工作")
    time.sleep(2)  # 模拟耗时操作
    print(f"进程 {name} 工作完成")

if __name__ == "__main__":
    # 在 Windows 平台上，multiprocessing 需要在 __main__ 块中使用
    
    # 创建两个进程
    p1 = multiprocessing.Process(target=worker, args=("P1",))
    p2 = multiprocessing.Process(target=worker, args=("P2",))
    
    # 启动进程
    p1.start()
    p2.start()
    
    # 等待进程完成
    p1.join()
    p2.join()
    
    print("所有进程已完成")
```

### 进程间通信

由于进程不共享内存，进程间需要使用特殊的通信机制：

```python
from multiprocessing import Process, Queue, Pipe

def sender(conn_or_queue, items):
    """发送数据到另一个进程"""
    for item in items:
        print(f"发送: {item}")
        conn_or_queue.put(item)
    conn_or_queue.put(None)  # 发送终止信号

def receiver(conn_or_queue):
    """从另一个进程接收数据"""
    while True:
        item = conn_or_queue.get()
        if item is None:  # 接收到终止信号
            break
        print(f"接收: {item}")

if __name__ == "__main__":
    # 使用队列通信
    q = Queue()
    items = ["消息1", "消息2", "消息3"]
    
    p1 = Process(target=sender, args=(q, items))
    p2 = Process(target=receiver, args=(q,))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    # 使用管道通信
    parent_conn, child_conn = Pipe()
    
    p3 = Process(target=sender, args=(parent_conn, items))
    p4 = Process(target=receiver, args=(child_conn,))
    
    p3.start()
    p4.start()
    p3.join()
    p4.join()
```

### 进程池

类似于线程池，`concurrent.futures` 模块也提供了 `ProcessPoolExecutor`：

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive_task(n):
    """CPU密集型任务：计算n的阶乘"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return f"{n}! = {result}"

if __name__ == "__main__":
    numbers = [10000, 20000, 30000, 40000]
    
    # 使用进程池处理CPU密集型任务
    with ProcessPoolExecutor(max_workers=4) as executor:
        for num, result in zip(numbers, executor.map(cpu_intensive_task, numbers)):
            print(f"{num} 的结果长度: {len(result)}")
```

### multiprocessing 模块其他功能

```python
from multiprocessing import Pool, Manager, Value, Array

if __name__ == "__main__":
    # 进程池
    with Pool(processes=4) as pool:
        results = pool.map(cpu_intensive_task, numbers)
    
    # 共享内存
    shared_value = Value('i', 0)  # 共享整数
    shared_array = Array('i', [1, 2, 3, 4])  # 共享数组
    
    # 进程间共享对象
    manager = Manager()
    shared_dict = manager.dict()
    shared_list = manager.list()
```

## 异步 IO 编程

Python 3.4 引入了 `asyncio` 模块，提供了基于协程的异步 IO 框架。这种方式特别适合高 IO 密集型应用。

### 基本协程用法

```python
import asyncio

async def hello_world():
    """一个简单的协程"""
    print("Hello")
    await asyncio.sleep(1)  # 非阻塞等待1秒
    print("World")
    return "完成"

# Python 3.7+
async def main():
    result = await hello_world()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())  # Python 3.7+ 推荐方式
```

### 并发执行多个协程

```python
import asyncio
import time

async def say_after(delay, what):
    """等待指定时间后打印消息"""
    await asyncio.sleep(delay)
    print(what)
    return what

async def main():
    start = time.time()
    
    # 串行执行
    print("\n串行执行:")
    serial_start = time.time()
    await say_after(1, "hello")
    await say_after(2, "world")
    print(f"串行耗时: {time.time() - serial_start:.2f}秒")  # 约3秒
    
    # 并发执行
    print("\n并发执行:")
    concurrent_start = time.time()
    task1 = asyncio.create_task(say_after(1, "hello"))
    task2 = asyncio.create_task(say_after(2, "world"))
    
    # 等待两个任务完成
    results = await asyncio.gather(task1, task2)
    print(f"结果: {results}")
    print(f"并发耗时: {time.time() - concurrent_start:.2f}秒")  # 约2秒
    
    print(f"\n总耗时: {time.time() - start:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())
```

### 异步 IO 实际应用：网络请求

使用 `aiohttp` 库进行异步 HTTP 请求：

```python
import asyncio
import aiohttp
import time

async def fetch(session, url):
    """异步获取URL内容"""
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://www.python.org",
        "https://www.google.com",
        "https://www.github.com"
    ]
    
    start = time.time()
    
    # 创建异步HTTP会话
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        for url, result in zip(urls, results):
            print(f"{url}: {len(result)} 字符")
    
    print(f"总耗时: {time.time() - start:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())
```

### asyncio 其他重要功能

```python
import asyncio

# 超时处理
async def with_timeout():
    try:
        await asyncio.wait_for(asyncio.sleep(10), timeout=2)
    except asyncio.TimeoutError:
        print("任务超时")

# 取消任务
async def cancel_task():
    task = asyncio.create_task(asyncio.sleep(10))
    await asyncio.sleep(2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("任务被取消")

# 等待第一个完成的任务
async def wait_for_first():
    task1 = asyncio.create_task(asyncio.sleep(2))
    task2 = asyncio.create_task(asyncio.sleep(1))
    done, pending = await asyncio.wait(
        [task1, task2],
        return_when=asyncio.FIRST_COMPLETED
    )
    print(f"完成的任务数: {len(done)}")
    print(f"等待的任务数: {len(pending)}")
    
    # 取消未完成的任务
    for task in pending:
        task.cancel()
```

## 使用并发的最佳实践

### 选择合适的并发模型

- **多线程**: 适合 I/O 密集型任务，如网络请求、文件操作
- **多进程**: 适合 CPU 密集型任务，如复杂计算
- **异步 IO**: 适合高并发 I/O 场景，如网络服务器

### 线程与进程的安全使用

1. 避免在线程间共享可变数据，如需共享请使用锁或其他同步机制
2. 进程间通信应使用队列或管道，避免复杂的共享状态
3. 线程池和进程池大小通常设为 CPU 核心数的 2-4 倍
4. 避免在线程中使用 `os.fork()`
5. 确保多线程/多进程中资源的正确释放

### 异步 IO 的有效使用

1. 不要在协程中使用阻塞操作，总是使用 `await`
2. 如果需要进行 CPU 密集型操作，考虑在单独的线程或进程中运行
3. 使用 `asyncio.gather()` 并发执行多个协程
4. 捕获并处理超时和取消
5. 记住协程不是线程或进程，它们在单线程中执行

## 混合使用多种并发模型

在实际应用中，可能需要混合使用多种并发模型：

```python
import asyncio
import concurrent.futures
import threading
import multiprocessing

def cpu_bound(n):
    """CPU密集型计算"""
    return sum(i * i for i in range(n))

async def main():
    # 使用线程池执行I/O密集型任务
    with concurrent.futures.ThreadPoolExecutor() as thread_pool:
        loop = asyncio.get_running_loop()
        
        # 在线程池中运行异步任务
        await loop.run_in_executor(
            thread_pool,
            lambda: time.sleep(1)  # 模拟I/O操作
        )
    
    # 使用进程池执行CPU密集型任务
    with concurrent.futures.ProcessPoolExecutor() as process_pool:
        loop = asyncio.get_running_loop()
        
        # 在进程池中运行CPU密集型任务
        results = await asyncio.gather(
            loop.run_in_executor(process_pool, cpu_bound, 10000000),
            loop.run_in_executor(process_pool, cpu_bound, 20000000),
            loop.run_in_executor(process_pool, cpu_bound, 30000000),
        )
        
        print(f"CPU密集型任务结果: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 并发编程调试技巧

并发程序的调试通常比顺序程序更复杂：

1. 使用 `threading.current_thread()` 和 `multiprocessing.current_process()` 识别当前线程/进程
2. 在并发代码中添加详细的日志
3. 使用 `asyncio.Task.get_stack()` 查看协程的调用栈
4. 设置 `PYTHONASYNCIODESUG=1` 环境变量启用 asyncio 调试
5. 考虑减少并发性进行问题定位

## 示例：构建一个简单的并发爬虫

以下是一个使用不同并发模型的网站爬虫示例：

```python
import requests
import time
import concurrent.futures
import asyncio
import aiohttp
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# 共同的爬取函数
def fetch_url(url):
    try:
        response = requests.get(url, timeout=5)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# 顺序爬取
def sequential_crawler(urls):
    results = []
    start = time.time()
    for url in urls:
        results.append(fetch_url(url))
    elapsed = time.time() - start
    return results, elapsed

# 线程池爬取
def threaded_crawler(urls):
    results = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_url, urls))
    elapsed = time.time() - start
    return results, elapsed

# 进程池爬取
def process_crawler(urls):
    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_url, urls))
    elapsed = time.time() - start
    return results, elapsed

# 异步爬取
async def fetch_url_async(session, url):
    try:
        async with session.get(url, timeout=5) as response:
            return await response.text()
    except Exception as e:
        return f"Error: {e}"

async def async_crawler(urls):
    results = []
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start
    return results, elapsed

# 主函数
def main():
    # 准备URL列表
    urls = [
        "https://www.python.org",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org"
    ]
    
    # 测试不同的爬虫方法
    print("顺序爬取:")
    _, seq_time = sequential_crawler(urls)
    print(f"耗时: {seq_time:.2f}秒\n")
    
    print("线程池爬取:")
    _, thread_time = threaded_crawler(urls)
    print(f"耗时: {thread_time:.2f}秒\n")
    
    print("进程池爬取:")
    _, process_time = process_crawler(urls)
    print(f"耗时: {process_time:.2f}秒\n")
    
    print("异步爬取:")
    loop = asyncio.get_event_loop()
    _, async_time = loop.run_until_complete(async_crawler(urls))
    print(f"耗时: {async_time:.2f}秒\n")
    
    # 比较
    print("性能比较:")
    print(f"顺序/线程比: {seq_time/thread_time:.2f}x")
    print(f"顺序/进程比: {seq_time/process_time:.2f}x")
    print(f"顺序/异步比: {seq_time/async_time:.2f}x")

if __name__ == "__main__":
    main()
```

## 总结

1. **多线程**:
   - 优点: 轻量级，共享内存，适合 I/O 密集型任务
   - 缺点: 受 GIL 限制，不适合 CPU 密集型任务，共享状态可能导致复杂性

2. **多进程**:
   - 优点: 可充分利用多核 CPU，适合 CPU 密集型任务
   - 缺点: 资源开销大，进程间通信复杂

3. **异步 IO**:
   - 优点: 轻量级，高效处理大量 I/O 任务
   - 缺点: 需要特殊语法，阻塞操作会影响整个事件循环

选择合适的并发模型取决于任务的性质和系统的特点。在实际应用中，可能需要结合使用多种并发模型以获得最佳性能。

## 下一步

继续深入学习 Python 并发编程的更多内容，请参考 [Python 异步编程](/advanced/async)。 