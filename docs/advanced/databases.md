# Python 与数据库

Python提供了多种方式与各类数据库进行交互，从简单的文件型数据库到企业级关系型数据库和NoSQL数据库。本章将介绍Python连接和操作各种数据库的方法。

## 数据库基础

### 数据库类型

在Python中，我们可以操作多种类型的数据库：

1. **关系型数据库**：使用SQL语言，如MySQL、PostgreSQL、SQLite、Oracle、SQL Server等
2. **NoSQL数据库**：如MongoDB、Redis、Cassandra等
3. **文件型数据库**：如SQLite、Berkeley DB等
4. **内存数据库**：如Redis（可同时作为内存和持久化数据库）

### Python数据库访问层次

Python操作数据库通常有以下几个层次：

1. **数据库驱动**：低级API，如sqlite3、pymysql、psycopg2等
2. **数据库工具包**：中级API，如SQLAlchemy Core、PyMongo等
3. **ORM**：高级API，如SQLAlchemy ORM、Django ORM、Peewee等

## SQLite

SQLite是一个轻量级的文件型数据库，Python标准库中自带sqlite3模块，无需额外安装。

### 基本操作

```python
import sqlite3

# 连接到数据库文件（如果不存在会创建）
conn = sqlite3.connect('example.db')

# 创建一个游标对象
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER
)
''')

# 插入数据
cursor.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
               ("张三", "zhangsan@example.com", 30))

# 插入多条数据
users = [
    ("李四", "lisi@example.com", 25),
    ("王五", "wangwu@example.com", 35)
]
cursor.executemany("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", users)

# 提交事务
conn.commit()

# 查询数据
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row)

# 关闭连接
conn.close()
```

### 使用上下文管理器

```python
import sqlite3

# 使用 with 语句自动处理连接的打开和关闭
with sqlite3.connect('example.db') as conn:
    cursor = conn.cursor()
    
    # 更新数据
    cursor.execute("UPDATE users SET age = ? WHERE name = ?", (31, "张三"))
    
    # 删除数据
    cursor.execute("DELETE FROM users WHERE name = ?", ("王五",))
    
    # 自动提交事务（在 with 块结束时）
```

### 结果以字典形式返回

```python
import sqlite3

# 连接数据库并返回字典结果
conn = sqlite3.connect('example.db')
conn.row_factory = sqlite3.Row

cursor = conn.cursor()
cursor.execute("SELECT * FROM users")

for row in cursor:
    # 可以通过字段名访问
    print(f"ID: {row['id']}, 姓名: {row['name']}, 邮箱: {row['email']}")

conn.close()
```

### SQLite事务控制

```python
import sqlite3

conn = sqlite3.connect('example.db')

try:
    # 开始一个显式事务
    conn.execute("BEGIN TRANSACTION")
    
    # 执行多个操作
    conn.execute("UPDATE users SET age = age + 1 WHERE name = '张三'")
    conn.execute("INSERT INTO users (name, email, age) VALUES ('赵六', 'zhaoliu@example.com', 28)")
    
    # 提交事务
    conn.commit()
    
except Exception as e:
    # 出现异常时回滚事务
    conn.rollback()
    print(f"错误: {e}")
    
finally:
    # 关闭连接
    conn.close()
```

## MySQL

MySQL是一个广泛使用的关系型数据库管理系统。Python可以通过多种驱动连接MySQL。

### 安装MySQL驱动

```bash
pip install pymysql  # 纯Python实现
# 或者
pip install mysql-connector-python  # Oracle官方驱动
```

### 使用PyMySQL

```python
import pymysql

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    user='username',
    password='password',
    database='testdb',
    charset='utf8mb4'
)

try:
    with conn.cursor() as cursor:
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            department VARCHAR(100),
            salary DECIMAL(10, 2)
        )
        ''')
        
        # 插入数据
        cursor.execute('''
        INSERT INTO employees (name, department, salary)
        VALUES (%s, %s, %s)
        ''', ('张三', '研发部', 10000.00))
        
        # 插入多条数据
        employees = [
            ('李四', '市场部', 9000.00),
            ('王五', '财务部', 11000.00)
        ]
        cursor.executemany('''
        INSERT INTO employees (name, department, salary)
        VALUES (%s, %s, %s)
        ''', employees)
        
        # 提交事务
        conn.commit()
        
        # 查询数据
        cursor.execute("SELECT * FROM employees")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
            
except Exception as e:
    # 出错时回滚
    conn.rollback()
    print(f"错误: {e}")
    
finally:
    # 关闭连接
    conn.close()
```

### 使用mysql-connector

```python
import mysql.connector

# 连接到MySQL数据库
conn = mysql.connector.connect(
    host='localhost',
    user='username',
    password='password',
    database='testdb'
)

cursor = conn.cursor(dictionary=True)  # 使用字典游标

# 查询数据
cursor.execute("SELECT * FROM employees WHERE department = %s", ("研发部",))
employees = cursor.fetchall()

for emp in employees:
    print(f"ID: {emp['id']}, 姓名: {emp['name']}, 薪资: {emp['salary']}")

# 关闭连接
cursor.close()
conn.close()
```

## PostgreSQL

PostgreSQL是一个功能强大的开源关系型数据库系统。Python使用psycopg2作为首选驱动。

### 安装PostgreSQL驱动

```bash
pip install psycopg2  # 需要编译器和PostgreSQL开发库
# 或者
pip install psycopg2-binary  # 预编译版本，无需额外依赖
```

### 基本操作

```python
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# 连接到PostgreSQL数据库
conn = psycopg2.connect(
    host='localhost',
    user='username',
    password='password',
    dbname='testdb'
)

try:
    # 创建一个游标对象
    with conn.cursor() as cursor:
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            price DECIMAL(10, 2),
            category VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 插入数据
        cursor.execute('''
        INSERT INTO products (name, price, category)
        VALUES (%s, %s, %s)
        ''', ('笔记本电脑', 6999.99, '电子产品'))
        
        # 使用SQL构造器（防止SQL注入）
        table_name = 'products'
        column_name = 'category'
        value = '办公用品'
        
        query = sql.SQL("SELECT * FROM {} WHERE {} = %s").format(
            sql.Identifier(table_name),
            sql.Identifier(column_name)
        )
        
        cursor.execute(query, [value])
        result = cursor.fetchall()
        print(result)
        
        # 提交事务
        conn.commit()
        
    # 使用字典游标
    with conn.cursor(cursor_factory=RealDictCursor) as dict_cursor:
        dict_cursor.execute("SELECT * FROM products")
        products = dict_cursor.fetchall()
        for product in products:
            print(f"名称: {product['name']}, 价格: {product['price']}")
            
except Exception as e:
    conn.rollback()
    print(f"错误: {e}")
    
finally:
    conn.close()
```

### PostgreSQL特有功能

```python
import psycopg2
from psycopg2.extras import Json

conn = psycopg2.connect("dbname=testdb user=username")
cursor = conn.cursor()

# 创建带JSON字段的表
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    items JSONB,
    status VARCHAR(20)
)
''')

# 插入JSON数据
order_items = [
    {"product_id": 1, "quantity": 2, "price": 199.99},
    {"product_id": 3, "quantity": 1, "price": 299.99}
]

cursor.execute('''
INSERT INTO orders (customer_id, items, status)
VALUES (%s, %s, %s)
''', (42, Json(order_items), 'pending'))

# 查询JSON数据
cursor.execute('''
SELECT * FROM orders
WHERE items @> '[{"product_id": 1}]'
''')

for order in cursor.fetchall():
    print(order)

conn.commit()
cursor.close()
conn.close()
```

## SQLAlchemy

SQLAlchemy是Python中最流行的ORM（对象关系映射）库，它提供了两种使用方式：Core（SQL表达式语言）和ORM（对象关系映射）。

### 安装SQLAlchemy

```bash
pip install sqlalchemy
```

### SQLAlchemy Core

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, select

# 创建引擎
engine = create_engine('sqlite:///example.db', echo=True)

# 定义元数据
metadata = MetaData()

# 定义表
users = Table('users', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(50), nullable=False),
    Column('email', String(100), unique=True),
    Column('age', Integer)
)

# 创建表
metadata.create_all(engine)

# 插入数据
with engine.connect() as conn:
    # 开始事务
    trans = conn.begin()
    try:
        # 插入一行
        conn.execute(users.insert().values(name='张三', email='zhangsan@example.com', age=30))
        
        # 插入多行
        conn.execute(users.insert(), [
            {'name': '李四', 'email': 'lisi@example.com', 'age': 25},
            {'name': '王五', 'email': 'wangwu@example.com', 'age': 35}
        ])
        
        # 提交事务
        trans.commit()
    except:
        # 回滚事务
        trans.rollback()
        raise

# 查询数据
with engine.connect() as conn:
    # 构建查询
    query = select(users).where(users.c.age > 25)
    
    # 执行查询
    result = conn.execute(query)
    
    # 处理结果
    for row in result:
        print(f"ID: {row.id}, 姓名: {row.name}, 年龄: {row.age}")
```

### SQLAlchemy ORM

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# 创建引擎
engine = create_engine('sqlite:///ormexample.db', echo=True)

# 创建基类
Base = declarative_base()

# 定义模型类
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)
    age = Column(Integer)
    
    posts = relationship("Post", back_populates="author")
    
    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}', age={self.age})>"

class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    content = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    author = relationship("User", back_populates="posts")
    
    def __repr__(self):
        return f"<Post(title='{self.title}')>"

# 创建表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

try:
    # 添加用户
    new_user = User(name='赵六', email='zhaoliu@example.com', age=28)
    session.add(new_user)
    
    # 添加多个用户
    session.add_all([
        User(name='孙七', email='sunqi@example.com', age=22),
        User(name='周八', email='zhouba@example.com', age=40)
    ])
    
    # 提交会话
    session.commit()
    
    # 添加文章
    user = session.query(User).filter_by(name='赵六').first()
    post = Post(title='我的第一篇文章', content='这是内容...', author=user)
    session.add(post)
    session.commit()
    
    # 查询
    for user in session.query(User).filter(User.age > 25).order_by(User.name):
        print(user)
        print(f"文章数: {len(user.posts)}")
        
    # 更新
    user = session.query(User).filter_by(name='赵六').first()
    user.age = 29
    session.commit()
    
    # 删除
    post_to_delete = session.query(Post).first()
    session.delete(post_to_delete)
    session.commit()
    
except:
    # 出错时回滚
    session.rollback()
    raise
    
finally:
    # 关闭会话
    session.close()
```

## MongoDB

MongoDB是一个流行的NoSQL文档数据库。Python使用PyMongo与MongoDB交互。

### 安装PyMongo

```bash
pip install pymongo
```

### 基本操作

```python
from pymongo import MongoClient
from bson.objectid import ObjectId

# 连接到MongoDB
client = MongoClient('mongodb://localhost:27017/')

# 获取数据库
db = client['testdb']

# 获取集合
collection = db['users']

# 插入文档
user = {
    "name": "张三",
    "email": "zhangsan@example.com",
    "age": 30,
    "tags": ["开发", "Python"],
    "address": {
        "city": "北京",
        "district": "海淀"
    }
}

# 插入一个文档
result = collection.insert_one(user)
print(f"插入的ID: {result.inserted_id}")

# 插入多个文档
users = [
    {"name": "李四", "email": "lisi@example.com", "age": 25},
    {"name": "王五", "email": "wangwu@example.com", "age": 35}
]
result = collection.insert_many(users)
print(f"插入的ID列表: {result.inserted_ids}")

# 查询文档
print("\n查找一个文档:")
user = collection.find_one({"name": "张三"})
print(user)

print("\n查找多个文档:")
for user in collection.find({"age": {"$gt": 25}}):
    print(user)

# 更新文档
result = collection.update_one(
    {"name": "张三"},
    {"$set": {"age": 31, "tags": ["开发", "Python", "MongoDB"]}}
)
print(f"更新匹配的文档数: {result.matched_count}")
print(f"更新修改的文档数: {result.modified_count}")

# 使用filter和update更新多个文档
result = collection.update_many(
    {"age": {"$lt": 30}},
    {"$inc": {"age": 1}}
)
print(f"更新修改的文档数: {result.modified_count}")

# 删除文档
result = collection.delete_one({"name": "王五"})
print(f"删除的文档数: {result.deleted_count}")

# 通过ID删除
result = collection.delete_one({"_id": ObjectId(result.inserted_ids[0])})

# 删除多个文档
result = collection.delete_many({"age": {"$gt": 30}})
print(f"删除的文档数: {result.deleted_count}")

# 聚合操作
pipeline = [
    {"$match": {"age": {"$gt": 20}}},
    {"$group": {"_id": "$address.city", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
for result in collection.aggregate(pipeline):
    print(result)

# 创建索引
collection.create_index("email", unique=True)
collection.create_index([("name", 1), ("age", -1)])

# 关闭连接
client.close()
```

## Redis

Redis是一个内存数据结构存储系统，可用作数据库、缓存和消息队列。Python使用redis-py与Redis交互。

### 安装redis-py

```bash
pip install redis
```

### 基本操作

```python
import redis

# 连接到Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# 字符串操作
r.set('name', '张三')
r.set('counter', 1)

name = r.get('name')
print(f"名字: {name.decode('utf-8')}")

# 自增
r.incr('counter')
r.incrby('counter', 5)
counter = r.get('counter')
print(f"计数器: {int(counter)}")

# 设置过期时间
r.setex('session_token', 3600, 'ABC123')  # 1小时后过期

# 哈希操作
r.hset('user:1000', 'name', '李四')
r.hset('user:1000', 'email', 'lisi@example.com')
r.hset('user:1000', 'age', 25)

# 一次设置多个字段
r.hmset('user:1001', {
    'name': '王五',
    'email': 'wangwu@example.com',
    'age': 35
})

# 获取单个字段
email = r.hget('user:1000', 'email')
print(f"邮箱: {email.decode('utf-8')}")

# 获取所有字段
user = r.hgetall('user:1000')
for key, value in user.items():
    print(f"{key.decode('utf-8')}: {value.decode('utf-8')}")

# 列表操作
r.lpush('tasks', 'task1')
r.lpush('tasks', 'task2')
r.rpush('tasks', 'task3')

# 获取列表元素
task = r.lpop('tasks')
print(f"下一个任务: {task.decode('utf-8')}")

# 获取列表范围
tasks = r.lrange('tasks', 0, -1)
for task in tasks:
    print(f"任务: {task.decode('utf-8')}")

# 集合操作
r.sadd('tags', 'python', 'redis', 'database')
r.sadd('languages', 'python', 'java', 'go')

# 检查成员
is_member = r.sismember('tags', 'python')
print(f"Python是否在标签集合中: {is_member}")

# 集合交集
common = r.sinter('tags', 'languages')
print("公共标签:", [tag.decode('utf-8') for tag in common])

# 有序集合
r.zadd('scores', {'张三': 89, '李四': 95, '王五': 78})

# 按分数获取成员
top_students = r.zrevrange('scores', 0, 1, withscores=True)
for student, score in top_students:
    print(f"{student.decode('utf-8')}: {score}")

# 发布订阅
pubsub = r.pubsub()
pubsub.subscribe('channel1')

# 在另一个线程或进程中发布消息
# r.publish('channel1', 'Hello Redis!')

# 接收消息
for message in pubsub.listen():
    if message['type'] == 'message':
        print(f"收到消息: {message['data'].decode('utf-8')}")
        break

# 清理资源
pubsub.unsubscribe()
```

## 数据库连接池与连接管理

### SQLite连接池

SQLite本身不支持连接池，但我们可以使用连接缓存模拟：

```python
import sqlite3
import threading

class SQLiteConnectionPool:
    def __init__(self, database, max_connections=5):
        self.database = database
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        
    def get_connection(self):
        with self.lock:
            if self.connections:
                return self.connections.pop()
            else:
                return sqlite3.connect(self.database)
    
    def release_connection(self, connection):
        with self.lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(connection)
            else:
                connection.close()
    
    def close_all(self):
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections = []

# 使用连接池
pool = SQLiteConnectionPool('example.db')

def do_work():
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        result = cursor.fetchall()
        return result
    finally:
        pool.release_connection(conn)

# 并发使用
results = []
threads = []
for i in range(10):
    t = threading.Thread(target=lambda: results.append(do_work()))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"获取到 {len(results)} 个结果")

# 关闭连接池
pool.close_all()
```

### PostgreSQL连接池

```python
import psycopg2
from psycopg2 import pool

# 创建线程安全的连接池
connection_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host='localhost',
    user='username',
    password='password',
    dbname='testdb'
)

# 获取连接
conn = connection_pool.getconn()

try:
    # 使用连接
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM products")
        for record in cursor:
            print(record)
            
finally:
    # 归还连接到池中
    connection_pool.putconn(conn)

# 应用程序结束时关闭池
connection_pool.closeall()
```

## 数据库迁移

数据库迁移是管理数据库模式（schema）变更的过程。Python有多个库支持数据库迁移。

### 使用Alembic（与SQLAlchemy配合）

```bash
pip install alembic
```

#### 初始化Alembic

```bash
# 创建迁移环境
alembic init migrations
```

#### 配置alembic.ini

```ini
# 编辑 alembic.ini 文件
sqlalchemy.url = sqlite:///example.db
```

#### 创建迁移脚本

```bash
# 创建迁移脚本
alembic revision -m "创建用户表"
```

#### 编辑迁移脚本

```python
# 编辑 migrations/versions/xxxx_创建用户表.py
"""创建用户表

Revision ID: xxxx
Revises: 
Create Date: 2023-07-18 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = 'xxxx'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('email', sa.String(100), unique=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )


def downgrade():
    op.drop_table('users')
```

#### 应用迁移

```bash
# 升级到最新版本
alembic upgrade head

# 回滚到上一个版本
alembic downgrade -1

# 生成新的迁移脚本（自动检测模型变化）
alembic revision --autogenerate -m "添加用户角色"
```

## 数据库安全最佳实践

### 防止SQL注入

```python
import sqlite3

# 不安全的方式 - 容易受到SQL注入攻击
def unsafe_search(username):
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    # 危险！直接拼接用户输入
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchall()

# 安全的方式 - 使用参数化查询
def safe_search(username):
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    # 安全：使用参数化查询
    query = "SELECT * FROM users WHERE name = ?"
    cursor.execute(query, (username,))
    return cursor.fetchall()

# 测试
print(safe_search("张三"))
# 不要使用：unsafe_search("张三' OR '1'='1")
```

### 凭证保护

```python
import os
from dotenv import load_dotenv
import psycopg2

# 加载环境变量
load_dotenv()

# 从环境变量获取敏感信息
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

# 使用环境变量连接数据库
conn = psycopg2.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    dbname=db_name
)
```

### 连接加密

```python
import pymysql

# 使用SSL连接MySQL
conn = pymysql.connect(
    host='localhost',
    user='username',
    password='password',
    db='testdb',
    ssl={
        'ca': '/path/to/ca-cert.pem',
        'cert': '/path/to/client-cert.pem',
        'key': '/path/to/client-key.pem'
    }
)
```

## Python 3.13 数据库相关新特性

Python 3.13 在数据库处理方面引入了一些改进：

### sqlite3模块增强

```python
import sqlite3

# Python 3.13 中对 sqlite3 模块的改进示例

# 创建数据库连接
conn = sqlite3.connect('example.db')

# 启用外键支持（更简化的方法）
conn.execute("PRAGMA foreign_keys = ON")

# 改进的类型转换
conn.execute("CREATE TABLE IF NOT EXISTS data (value JSON)")
conn.execute("INSERT INTO data VALUES (?)", ('{"name": "张三", "age": 30}',))

# 查询 JSON 数据
cursor = conn.execute("SELECT json_extract(value, '$.name') FROM data")
print(cursor.fetchone()[0])  # 输出: 张三

# 使用更多的 SQLite 功能
conn.create_function("reverse", 1, lambda s: s[::-1])
cursor = conn.execute("SELECT reverse(name) FROM users")
for row in cursor:
    print(row[0])  # 反转后的名字

conn.close()
```

## 数据库性能优化

### 查询优化

```python
import time
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# 创建测试表和索引
cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, data TEXT)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_data ON test (data)")

# 插入测试数据
for i in range(10000):
    cursor.execute("INSERT INTO test (data) VALUES (?)", (f"data_{i}",))
conn.commit()

# 未优化查询
def unoptimized_query():
    start = time.time()
    cursor.execute("SELECT * FROM test WHERE data LIKE 'data_9%'")
    results = cursor.fetchall()
    end = time.time()
    print(f"未优化查询: {len(results)} 行, 耗时 {end - start:.6f} 秒")

# 优化查询 - 使用索引
def optimized_query():
    start = time.time()
    cursor.execute("SELECT id FROM test WHERE data LIKE 'data_9%'")
    results = cursor.fetchall()
    end = time.time()
    print(f"优化查询: {len(results)} 行, 耗时 {end - start:.6f} 秒")

# 批量操作
def batch_insert():
    start = time.time()
    data = [(f"batch_{i}",) for i in range(10000)]
    cursor.executemany("INSERT INTO test (data) VALUES (?)", data)
    conn.commit()
    end = time.time()
    print(f"批量插入: 10000 行, 耗时 {end - start:.6f} 秒")

# 测试性能
unoptimized_query()
optimized_query()
batch_insert()

conn.close()
```

### 连接池和批处理

```python
import time
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor

# 创建带连接池的引擎
engine = create_engine(
    'sqlite:///example.db',
    pool_size=20,  # 连接池大小
    max_overflow=10,  # 允许的最大溢出连接数
    pool_timeout=30,  # 等待连接的超时时间
    pool_recycle=1800  # 连接回收时间(秒)
)

# 批量操作
def batch_execute(batch_size=1000):
    data = [(f"item_{i}",) for i in range(10000)]
    start = time.time()
    
    with engine.begin() as conn:  # 自动处理事务
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            conn.execute(
                text("INSERT INTO test (data) VALUES (:data)"),
                [{"data": item[0]} for item in batch]
            )
    
    end = time.time()
    print(f"批量执行: {len(data)} 行, 批次大小 {batch_size}, 耗时 {end - start:.6f} 秒")

# 并发查询
def concurrent_queries(num_threads=5):
    queries = ["SELECT COUNT(*) FROM test WHERE data LIKE 'data_%'",
              "SELECT COUNT(*) FROM test WHERE data LIKE 'batch_%'",
              "SELECT COUNT(*) FROM test WHERE id < 5000",
              "SELECT COUNT(*) FROM test WHERE id > 5000"]
    
    start = time.time()
    
    def execute_query(query):
        with engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            return result
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(execute_query, queries * 5))  # 执行20个查询
    
    end = time.time()
    print(f"并发查询: {len(queries) * 5} 查询, {num_threads} 线程, 耗时 {end - start:.6f} 秒")

# 测试性能
batch_execute(500)
batch_execute(1000)
concurrent_queries(5)
concurrent_queries(10)
```

## 下一步

现在您已经了解了 Python 与各种数据库的交互方式，接下来可以探索 [Python 网络编程](/advanced/networking)，学习如何创建网络应用以及如何将网络应用与数据库结合起来构建完整的系统。或者，您可以探索 [实践项目](/projects/web-app) 章节，应用所学知识构建实际的 Python 应用程序。 