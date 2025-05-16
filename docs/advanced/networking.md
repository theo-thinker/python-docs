# Python 网络编程

Python 提供了强大的网络编程支持，从低级套接字操作到高级协议实现，使得创建网络应用和服务成为可能。本章将介绍 Python 中网络编程的基础知识和常用技术。

## 网络编程基础

### 计算机网络基本概念

在深入 Python 网络编程之前，让我们先回顾一些基本概念：

- **IP 地址**：标识网络上的设备
- **端口**：用于区分同一设备上的不同网络服务
- **协议**：设备之间通信的规则（如 TCP、UDP、HTTP、FTP 等）
- **客户端/服务器模式**：网络应用的基本架构模式
- **套接字（Socket）**：网络编程的基本 API

### Python 中的网络模块

Python 提供了多个用于网络编程的模块：

- **socket**：低级网络接口，支持 TCP/IP 和 UDP 等协议
- **http.client** 和 **http.server**：HTTP 协议客户端和服务器实现
- **urllib** 和 **urllib.request**：用于处理 URL 和网络请求
- **ftplib**：FTP 协议客户端
- **smtplib** 和 **email**：用于发送电子邮件
- **poplib** 和 **imaplib**：用于接收电子邮件
- **socketserver**：简化网络服务器开发的框架
- **asyncio**：异步 I/O、事件循环和协程，适用于高并发网络应用

## 套接字编程

套接字（Socket）是网络编程的基础，它提供了一种应用程序与网络通信的方法。

### 创建 TCP 套接字

```python
import socket

# 创建一个 TCP 套接字
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
```

参数说明：
- `socket.AF_INET`：IPv4 地址族
- `socket.SOCK_STREAM`：TCP 协议（面向连接）

### TCP 服务器示例

```python
import socket

# 创建服务器套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 设置选项：允许地址重用
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 绑定地址和端口
server_address = ('localhost', 8888)
server_socket.bind(server_address)

# 开始监听，参数5表示连接队列的最大长度
server_socket.listen(5)
print(f"服务器启动，监听 {server_address}")

try:
    while True:
        # 接受客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"客户端 {client_address} 已连接")
        
        try:
            # 接收数据
            data = client_socket.recv(1024)
            if data:
                # 发送响应
                message = f"已收到 {len(data)} 字节的数据"
                client_socket.sendall(message.encode('utf-8'))
            else:
                print(f"客户端 {client_address} 未发送数据")
        finally:
            # 关闭客户端连接
            client_socket.close()
            print(f"客户端 {client_address} 连接已关闭")
            
except KeyboardInterrupt:
    print("服务器关闭")
finally:
    # 关闭服务器套接字
    server_socket.close()
```

### TCP 客户端示例

```python
import socket

# 创建客户端套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 服务器地址
server_address = ('localhost', 8888)

try:
    # 连接服务器
    client_socket.connect(server_address)
    print(f"已连接到服务器 {server_address}")
    
    # 发送数据
    message = "你好，服务器！"
    client_socket.sendall(message.encode('utf-8'))
    
    # 接收响应
    data = client_socket.recv(1024)
    print(f"收到服务器响应：{data.decode('utf-8')}")
    
finally:
    # 关闭套接字
    client_socket.close()
    print("连接已关闭")
```

### UDP 通信

UDP（用户数据报协议）是一种无连接的传输协议，适用于对实时性要求较高，但对可靠性要求较低的场景。

#### UDP 服务器示例

```python
import socket

# 创建 UDP 套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定地址和端口
server_address = ('localhost', 9999)
server_socket.bind(server_address)

print(f"UDP 服务器启动，监听 {server_address}")

try:
    while True:
        # 接收数据和客户端地址
        data, client_address = server_socket.recvfrom(1024)
        print(f"从 {client_address} 收到：{data.decode('utf-8')}")
        
        # 发送响应
        message = f"已收到 {len(data)} 字节的数据"
        server_socket.sendto(message.encode('utf-8'), client_address)
        
except KeyboardInterrupt:
    print("服务器关闭")
finally:
    server_socket.close()
```

#### UDP 客户端示例

```python
import socket

# 创建 UDP 套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 服务器地址
server_address = ('localhost', 9999)

try:
    # 发送数据
    message = "你好，UDP 服务器！"
    client_socket.sendto(message.encode('utf-8'), server_address)
    
    # 接收响应
    data, server = client_socket.recvfrom(1024)
    print(f"收到来自 {server} 的响应：{data.decode('utf-8')}")
    
finally:
    client_socket.close()
```

### 套接字选项和超时

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 设置套接字选项
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 设置超时（秒）
s.settimeout(10.0)

try:
    s.connect(('example.com', 80))
    # 在超时内完成操作
except socket.timeout:
    print("连接超时")
finally:
    s.close()
```

### 非阻塞套接字

```python
import socket
import select

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 8888))
server_socket.listen(5)

# 设置为非阻塞模式
server_socket.setblocking(False)

# 待监听的套接字列表
inputs = [server_socket]

try:
    while inputs:
        # select 监听套接字事件，timeout 为 1 秒
        readable, writable, exceptional = select.select(inputs, [], inputs, 1)
        
        for s in readable:
            if s is server_socket:
                # 服务器套接字有事件意味着新的客户端连接
                client_socket, client_address = s.accept()
                print(f"客户端 {client_address} 已连接")
                client_socket.setblocking(False)
                inputs.append(client_socket)
            else:
                # 客户端套接字有事件意味着有数据可读或客户端关闭连接
                try:
                    data = s.recv(1024)
                    if data:
                        print(f"收到数据：{data.decode('utf-8')}")
                        s.sendall("已收到数据".encode('utf-8'))
                    else:
                        # 客户端关闭连接
                        print("客户端关闭连接")
                        inputs.remove(s)
                        s.close()
                except ConnectionError:
                    print("连接错误")
                    inputs.remove(s)
                    s.close()
        
        for s in exceptional:
            print(f"套接字异常")
            inputs.remove(s)
            s.close()
            
except KeyboardInterrupt:
    print("服务器关闭")
finally:
    for s in inputs:
        s.close()
```

## HTTP 客户端

Python 提供了多种方式来发送 HTTP 请求。

### urllib 模块

```python
import urllib.request
import urllib.parse

# 简单的 GET 请求
with urllib.request.urlopen('https://www.python.org') as response:
    html = response.read()
    print(f"网页大小：{len(html)} 字节")
    print(f"状态码：{response.status}")

# 带参数的 GET 请求
params = {'q': 'python', 'page': '1'}
url = f"https://www.example.com/search?{urllib.parse.urlencode(params)}"
with urllib.request.urlopen(url) as response:
    data = response.read()

# POST 请求
data = urllib.parse.urlencode({'user': 'username', 'password': 'pass123'}).encode('utf-8')
req = urllib.request.Request('https://www.example.com/login', data=data, method='POST')
with urllib.request.urlopen(req) as response:
    result = response.read()

# 添加请求头
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Content-Type': 'application/x-www-form-urlencoded'
}
req = urllib.request.Request('https://www.example.com', headers=headers)
with urllib.request.urlopen(req) as response:
    data = response.read()
```

### requests 库

`requests` 是一个第三方库，提供了更简洁、更强大的 HTTP 客户端功能。安装：`pip install requests`。

```python
import requests

# 发送 GET 请求
response = requests.get('https://www.python.org')
print(f"状态码: {response.status_code}")
print(f"内容类型: {response.headers['content-type']}")
print(f"编码: {response.encoding}")
print(f"内容长度: {len(response.text)} 字符")

# 带参数的 GET 请求
params = {'q': 'python', 'page': '1'}
response = requests.get('https://www.example.com/search', params=params)
print(f"URL: {response.url}")

# POST 请求 - 表单数据
data = {'username': 'user', 'password': 'pass123'}
response = requests.post('https://www.example.com/login', data=data)

# POST 请求 - JSON 数据
json_data = {'name': 'John', 'age': 30}
response = requests.post('https://api.example.com/users', json=json_data)
print(response.json())  # 解析 JSON 响应

# 自定义请求头
headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
response = requests.get('https://api.example.com/data', headers=headers)

# 会话（保持 cookies）
session = requests.Session()
session.get('https://www.example.com/login')  # 获取 cookies
response = session.post('https://www.example.com/login', data={'user': 'name', 'pass': '123'})
# 后续请求会保持会话状态
session.get('https://www.example.com/profile')

# 超时设置
try:
    response = requests.get('https://www.example.com', timeout=3)
except requests.Timeout:
    print("请求超时")

# 文件上传
files = {'file': open('document.txt', 'rb')}
response = requests.post('https://www.example.com/upload', files=files)

# 使用代理
proxies = {
    'http': 'http://10.10.10.10:8000',
    'https': 'http://10.10.10.10:8000',
}
response = requests.get('https://www.example.com', proxies=proxies)

# 响应内容处理
response = requests.get('https://api.github.com/events')
data = response.json()  # JSON 解析
response = requests.get('https://www.python.org')
text = response.text  # 文本内容
binary = response.content  # 二进制内容
```

### aiohttp 库（异步 HTTP 客户端）

`aiohttp` 是一个支持异步 HTTP 请求的库。安装：`pip install aiohttp`。

```python
import aiohttp
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        'https://www.python.org',
        'https://www.github.com',
        'https://www.stackoverflow.com'
    ]
    
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for url, result in zip(urls, results):
        print(f"{url}: 内容长度 {len(result)} 字符")

# 运行异步主函数
asyncio.run(main())
```

## HTTP 服务器

Python 提供了多种方式来创建 HTTP 服务器。

### 使用 http.server 模块

```python
import http.server
import socketserver

# 定义请求处理器
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Hello, World!</h1></body></html>")
        elif self.path == '/api':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"message": "Hello from API"}')
        else:
            # 默认处理（提供静态文件）
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/submit':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = f'{{"received": "{post_data.decode("utf-8")}"}}'
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

# 创建服务器
PORT = 8000
handler = MyHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"服务器运行在端口 {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("服务器已关闭")
```

### 使用 Flask 框架

Flask 是一个轻量级的 Web 应用框架。安装：`pip install flask`。

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Hello, World!</h1>'

@app.route('/api/data')
def get_data():
    # 获取查询参数
    name = request.args.get('name', 'Guest')
    return jsonify({'message': f'Hello, {name}!'})

@app.route('/api/users', methods=['POST'])
def create_user():
    # 获取 JSON 数据
    user_data = request.json
    # 处理数据（这里只是返回）
    return jsonify({'user': user_data, 'status': 'created'}), 201

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    # 处理路径参数
    return jsonify({'user_id': user_id, 'name': f'User {user_id}'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
```

## 其他网络协议

### FTP 客户端

```python
import ftplib

# 连接到 FTP 服务器
ftp = ftplib.FTP('ftp.example.com')
ftp.login(user='username', passwd='password')

# 显示当前目录内容
ftp.dir()

# 切换目录
ftp.cwd('/pub')

# 下载文件
with open('downloaded_file.txt', 'wb') as f:
    ftp.retrbinary('RETR README.txt', f.write)

# 上传文件
with open('local_file.txt', 'rb') as f:
    ftp.storbinary('STOR remote_file.txt', f)

# 关闭连接
ftp.quit()
```

### SMTP（发送邮件）

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 创建邮件
msg = MIMEMultipart()
msg['From'] = 'sender@example.com'
msg['To'] = 'recipient@example.com'
msg['Subject'] = '测试邮件'

# 添加正文
body = "这是一封测试邮件的正文。"
msg.attach(MIMEText(body, 'plain'))

# 连接到 SMTP 服务器
server = smtplib.SMTP('smtp.example.com', 587)
server.starttls()  # 启用 TLS 加密
server.login('username', 'password')

# 发送邮件
server.send_message(msg)

# 关闭连接
server.quit()
```

### POP3/IMAP（接收邮件）

```python
# POP3 示例
import poplib
from email import parser

# 连接到 POP3 服务器
pop_server = poplib.POP3_SSL('pop.example.com')
pop_server.user('username')
pop_server.pass_('password')

# 获取邮件数量和大小
num_messages = len(pop_server.list()[1])
print(f"邮箱中有 {num_messages} 封邮件")

# 获取最新邮件
response, lines, octets = pop_server.retr(num_messages)
message_content = b'\n'.join(lines).decode('utf-8')
message = parser.Parser().parsestr(message_content)

print(f"From: {message['From']}")
print(f"Subject: {message['Subject']}")

# 关闭连接
pop_server.quit()

# IMAP 示例
import imaplib
import email

# 连接到 IMAP 服务器
imap_server = imaplib.IMAP4_SSL('imap.example.com')
imap_server.login('username', 'password')

# 选择邮箱
imap_server.select('INBOX')

# 搜索邮件
status, data = imap_server.search(None, 'ALL')
mail_ids = data[0].split()

# 获取最新邮件
latest_email_id = mail_ids[-1]
status, data = imap_server.fetch(latest_email_id, '(RFC822)')
raw_email = data[0][1]
email_message = email.message_from_bytes(raw_email)

print(f"From: {email_message['From']}")
print(f"Subject: {email_message['Subject']}")

# 关闭连接
imap_server.logout()
```

## 网络安全考虑

### 加密通信（SSL/TLS）

```python
import socket
import ssl

# 创建上下文
context = ssl.create_default_context()

# 作为客户端（验证服务器证书）
with socket.create_connection(('www.python.org', 443)) as sock:
    with context.wrap_socket(sock, server_hostname='www.python.org') as ssock:
        print(f"使用的 SSL 版本: {ssock.version()}")
        print(f"加密套件: {ssock.cipher()}")
        cert = ssock.getpeercert()
        print(f"服务器证书: {cert}")

# 作为服务器（提供证书）
server_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
server_context.load_cert_chain(certfile='server.crt', keyfile='server.key')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(('localhost', 8443))
    sock.listen(5)
    
    with server_context.wrap_socket(sock, server_side=True) as ssock:
        conn, addr = ssock.accept()
        # 处理安全连接...
```

### 防止常见攻击

```python
# 防止 SQL 注入
import sqlite3

# 不安全的方式（容易受到 SQL 注入攻击）
def unsafe_query(user_input):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{user_input}'"  # 危险！
    cursor.execute(query)
    return cursor.fetchall()

# 安全的方式（使用参数化查询）
def safe_query(user_input):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = ?"
    cursor.execute(query, (user_input,))  # 安全
    return cursor.fetchall()

# 防止 XSS 攻击（在 Web 应用中）
import html

def render_user_content(content):
    # 转义用户输入内容，防止 XSS
    safe_content = html.escape(content)
    return f"<div>{safe_content}</div>"
```

## Python 3.13 网络编程新特性

Python 3.13 在网络编程方面引入了一些改进：

### asyncio 改进

```python
import asyncio

# 以下是 Python 3.13 中 asyncio 模块的一些改进示例

async def main():
    # 任务组（Python 3.11+，在 3.13 中有性能改进）
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(asyncio.sleep(1))
        task2 = tg.create_task(asyncio.sleep(2))
        task3 = tg.create_task(asyncio.sleep(3))
    
    # 结果在这里可用，任何任务失败都会正确传播异常
    
    # 超时处理改进
    try:
        result = await asyncio.wait_for(long_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("操作超时")

async def long_operation():
    await asyncio.sleep(10)
    return "完成"

asyncio.run(main())
```

### SSL/TLS 增强

```python
import ssl

# 创建具有更严格安全设置的上下文
context = ssl.create_default_context()

# Python 3.13 改进了对最新 TLS 版本和加密套件的支持
context.minimum_version = ssl.TLSVersion.TLSv1_3  # 仅支持 TLS 1.3

# 检查 Python 支持的 SSL/TLS 协议版本
print("支持的 SSL/TLS 版本:")
for version in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1_1', 'TLSv1_2', 'TLSv1_3']:
    try:
        protocol = getattr(ssl.TLSVersion, version)
        print(f"  {version}: 支持")
    except AttributeError:
        print(f"  {version}: 不支持")
```

## 最佳实践和性能优化

### 异步 I/O 提高并发性能

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    start_time = time.time()
    async with session.get(url) as response:
        text = await response.text()
        elapsed = time.time() - start_time
        print(f"{url} - 获取了 {len(text)} 字节，耗时 {elapsed:.2f} 秒")
        return len(text)

async def main():
    urls = [
        'https://www.python.org',
        'https://www.github.com',
        'https://www.stackoverflow.com',
        'https://www.wikipedia.org',
        'https://www.reddit.com'
    ] * 2  # 重复多次以显示性能差异
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    total_bytes = sum(results)
    total_time = time.time() - start_time
    print(f"总计: {len(urls)} 个 URL, {total_bytes} 字节, 耗时 {total_time:.2f} 秒")

# 运行异步代码
asyncio.run(main())

# 对比顺序下载
def sequential_fetch():
    import requests
    
    urls = [
        'https://www.python.org',
        'https://www.github.com',
        'https://www.stackoverflow.com',
        'https://www.wikipedia.org',
        'https://www.reddit.com'
    ] * 2
    
    start_time = time.time()
    total_bytes = 0
    
    for url in urls:
        start = time.time()
        response = requests.get(url)
        size = len(response.text)
        elapsed = time.time() - start
        total_bytes += size
        print(f"{url} - 获取了 {size} 字节，耗时 {elapsed:.2f} 秒")
    
    total_time = time.time() - start_time
    print(f"总计: {len(urls)} 个 URL, {total_bytes} 字节, 耗时 {total_time:.2f} 秒")

# 运行顺序代码
# sequential_fetch()  # 取消注释以运行顺序下载测试
```

### 连接池和长连接

```python
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 创建会话对象
session = requests.Session()

# 配置重试策略
retries = Retry(
    total=5,  # 最大重试次数
    backoff_factor=0.5,  # 重试间隔增长因子
    status_forcelist=[500, 502, 503, 504]  # 需要重试的 HTTP 状态码
)

# 配置连接池和重试策略
adapter = HTTPAdapter(
    max_retries=retries,
    pool_connections=10,  # 连接池中连接的数量
    pool_maxsize=10       # 连接池中最大的连接数
)

# 挂载适配器
session.mount('http://', adapter)
session.mount('https://', adapter)

# 使用会话发送请求
response = session.get('https://www.example.com')
```

### 网络编程常见错误和解决方案

```python
import socket
import requests
import time

# 处理连接错误
def handle_connection_errors(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            return response
        except requests.ConnectionError:
            print(f"连接错误，尝试 {attempt + 1}/{max_retries}")
            time.sleep(2 ** attempt)  # 指数退避
        except requests.Timeout:
            print(f"请求超时，尝试 {attempt + 1}/{max_retries}")
            time.sleep(2 ** attempt)
    
    raise Exception(f"无法连接到 {url}，已达到最大重试次数")

# 优雅地关闭套接字
def graceful_socket_shutdown(sock):
    try:
        # 关闭写入端（发送 FIN）
        sock.shutdown(socket.SHUT_WR)
        
        # 读取剩余数据
        while True:
            data = sock.recv(1024)
            if not data:
                break
        
        # 完全关闭套接字
        sock.close()
    except OSError as e:
        print(f"关闭套接字时出错: {e}")
        sock.close()
```

## 下一步

现在您已经了解了 Python 的网络编程基础，接下来可以探索 [Python 与数据库](/advanced/databases) 的交互，学习如何将网络应用与数据库结合起来。 