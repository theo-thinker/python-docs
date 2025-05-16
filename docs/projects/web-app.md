# Web 应用开发

本章将介绍如何使用 Python 开发 Web 应用，我们将从基础开始，逐步构建一个功能完整的 Web 应用程序。

## Web 应用基础

### Web 应用架构

现代 Web 应用通常采用以下架构：

1. **前端**：用户界面，通常使用 HTML、CSS 和 JavaScript 构建
2. **后端**：处理业务逻辑，通常使用 Python、Java、Node.js 等语言构建
3. **数据库**：存储和管理数据，如 MySQL、PostgreSQL、MongoDB 等
4. **API**：前后端通信接口，通常使用 REST 或 GraphQL

Python 在 Web 开发中主要用于后端开发，但也可以用于全栈开发。

### Python Web 框架

Python 有多种 Web 框架，最流行的包括：

- **Flask**：轻量级框架，适合小型应用和 API
- **Django**：全功能框架，包含 ORM、管理后台等，适合大型应用
- **FastAPI**：现代高性能框架，专注于 API 开发，支持异步
- **Pyramid**：灵活的中型框架，可扩展性强
- **Bottle**：超轻量级单文件框架，适合简单应用和学习

本章将主要使用 Flask 和 FastAPI 进行演示。

## 使用 Flask 开发 Web 应用

[Flask](https://flask.palletsprojects.com/) 是一个轻量级的 Python Web 框架，它提供了构建 Web 应用的基本工具，但也可以通过扩展来增加功能。

### 安装 Flask

```bash
pip install flask
```

### 创建第一个 Flask 应用

```python
# app.py
from flask import Flask

# 创建 Flask 应用
app = Flask(__name__)

# 定义路由
@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About page'

# 动态路由
@app.route('/user/<username>')
def user_profile(username):
    return f'User profile: {username}'

# 启动应用
if __name__ == '__main__':
    app.run(debug=True)
```

运行应用：

```bash
python app.py
```

访问 http://127.0.0.1:5000/ 即可看到 "Hello, World!"。

### 使用模板

Flask 使用 Jinja2 作为模板引擎，可以将 HTML 与 Python 代码结合起来。

```python
# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title='Home')

@app.route('/users')
def users():
    users_list = [
        {'name': '张三', 'age': 25},
        {'name': '李四', 'age': 30},
        {'name': '王五', 'age': 35}
    ]
    return render_template('users.html', users=users_list)

if __name__ == '__main__':
    app.run(debug=True)
```

创建模板文件：

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>欢迎来到我的网站</h1>
    <p>这是使用 Flask 构建的网站。</p>
</body>
</html>
```

```html
<!-- templates/users.html -->
<!DOCTYPE html>
<html>
<head>
    <title>用户列表</title>
</head>
<body>
    <h1>用户列表</h1>
    <ul>
        {% for user in users %}
            <li>{{ user.name }} - {{ user.age }} 岁</li>
        {% endfor %}
    </ul>
</body>
</html>
```

### 处理表单

Flask 可以方便地处理 HTML 表单：

```python
# app.py
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # 用于 flash 消息

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 简单的验证逻辑
        if username == 'admin' and password == 'password':
            flash('登录成功！')
            return redirect(url_for('home'))
        else:
            flash('用户名或密码错误！')
    
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
```

创建登录表单模板：

```html
<!-- templates/login.html -->
<!DOCTYPE html>
<html>
<head>
    <title>登录</title>
</head>
<body>
    <h1>登录</h1>
    
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            <ul class="flashes">
                {% for message in messages %}
                    <li>{{ message }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    {% endwith %}
    
    <form method="post">
        <div>
            <label for="username">用户名：</label>
            <input type="text" id="username" name="username" required>
        </div>
        <div>
            <label for="password">密码：</label>
            <input type="password" id="password" name="password" required>
        </div>
        <button type="submit">登录</button>
    </form>
</body>
</html>
```

### 使用数据库

Flask 自身不提供数据库支持，但可以很容易地集成 SQLAlchemy 或其他数据库库。

```bash
pip install flask-sqlalchemy
```

```python
# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'some_secret_key'

db = SQLAlchemy(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"Post('{self.title}', '{self.created_at}')"

# 创建数据库表
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('home.html', posts=posts)

@app.route('/post/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        
        post = Post(title=title, content=content)
        db.session.add(post)
        db.session.commit()
        
        flash('文章已发布！')
        return redirect(url_for('home'))
    
    return render_template('create_post.html')

@app.route('/post/<int:post_id>')
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', post=post)

if __name__ == '__main__':
    app.run(debug=True)
```

创建相应的模板文件：

```html
<!-- templates/home.html -->
<!DOCTYPE html>
<html>
<head>
    <title>博客首页</title>
</head>
<body>
    <h1>博客文章</h1>
    
    <a href="{{ url_for('new_post') }}">发布新文章</a>
    
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            <ul class="flashes">
                {% for message in messages %}
                    <li>{{ message }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    {% endwith %}
    
    {% if posts %}
        {% for post in posts %}
            <article>
                <h2><a href="{{ url_for('post', post_id=post.id) }}">{{ post.title }}</a></h2>
                <p>发布时间: {{ post.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p>{{ post.content[:100] }}{% if post.content|length > 100 %}...{% endif %}</p>
            </article>
        {% endfor %}
    {% else %}
        <p>暂无文章</p>
    {% endif %}
</body>
</html>
```

```html
<!-- templates/create_post.html -->
<!DOCTYPE html>
<html>
<head>
    <title>发布新文章</title>
</head>
<body>
    <h1>发布新文章</h1>
    
    <form method="post">
        <div>
            <label for="title">标题:</label>
            <input type="text" id="title" name="title" required>
        </div>
        <div>
            <label for="content">内容:</label>
            <textarea id="content" name="content" rows="10" required></textarea>
        </div>
        <button type="submit">发布</button>
    </form>
    
    <a href="{{ url_for('home') }}">返回首页</a>
</body>
</html>
```

```html
<!-- templates/post.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ post.title }}</title>
</head>
<body>
    <article>
        <h1>{{ post.title }}</h1>
        <p>发布时间: {{ post.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <div>
            {{ post.content }}
        </div>
    </article>
    
    <a href="{{ url_for('home') }}">返回首页</a>
</body>
</html>
```

## 使用 FastAPI 开发 API

[FastAPI](https://fastapi.tiangolo.com/) 是一个现代、高性能的 Python Web 框架，专注于构建 API，自动生成 API 文档，并支持异步处理。

### 安装 FastAPI

```bash
pip install fastapi uvicorn
```

### 创建第一个 FastAPI 应用

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# 创建 FastAPI 应用
app = FastAPI(title="任务管理 API")

# 定义数据模型
class Task(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    completed: bool = False

# 模拟数据库
tasks_db = []
task_id_counter = 1

# 定义 API 路由
@app.get("/")
def read_root():
    return {"message": "欢迎使用任务管理 API"}

@app.get("/tasks", response_model=List[Task])
def get_tasks():
    return tasks_db

@app.post("/tasks", response_model=Task, status_code=201)
def create_task(task: Task):
    global task_id_counter
    task.id = task_id_counter
    task_id_counter += 1
    tasks_db.append(task)
    return task

@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: int):
    for task in tasks_db:
        if task.id == task_id:
            return task
    return {"error": "Task not found"}

@app.put("/tasks/{task_id}", response_model=Task)
def update_task(task_id: int, task: Task):
    for i, t in enumerate(tasks_db):
        if t.id == task_id:
            task.id = task_id
            tasks_db[i] = task
            return task
    return {"error": "Task not found"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    for i, task in enumerate(tasks_db):
        if task.id == task_id:
            del tasks_db[i]
            return {"message": "Task deleted"}
    return {"error": "Task not found"}
```

运行应用：

```bash
uvicorn main:app --reload
```

访问 http://127.0.0.1:8000/docs 可以看到自动生成的 API 文档。

### 使用 SQLAlchemy 操作数据库

```bash
pip install sqlalchemy databases
```

```python
# main.py
import databases
import sqlalchemy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 数据库配置
DATABASE_URL = "sqlite:///./tasks.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# 定义数据表
tasks = sqlalchemy.Table(
    "tasks",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("title", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("description", sqlalchemy.String),
    sqlalchemy.Column("completed", sqlalchemy.Boolean, default=False),
)

# 创建数据库引擎和表
engine = sqlalchemy.create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
metadata.create_all(engine)

# 定义数据模型
class TaskIn(BaseModel):
    title: str
    description: Optional[str] = None
    completed: bool = False

class Task(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False

# 创建 FastAPI 应用
app = FastAPI(title="任务管理 API")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/tasks", response_model=Task, status_code=201)
async def create_task(task: TaskIn):
    query = tasks.insert().values(
        title=task.title,
        description=task.description,
        completed=task.completed
    )
    last_record_id = await database.execute(query)
    return {**task.dict(), "id": last_record_id}

@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    query = tasks.select()
    return await database.fetch_all(query)

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: int):
    query = tasks.select().where(tasks.c.id == task_id)
    task = await database.fetch_one(query)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, task: TaskIn):
    query = tasks.select().where(tasks.c.id == task_id)
    db_task = await database.fetch_one(query)
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    update_query = tasks.update().where(tasks.c.id == task_id).values(
        title=task.title,
        description=task.description,
        completed=task.completed
    )
    await database.execute(update_query)
    
    return {**task.dict(), "id": task_id}

@app.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: int):
    query = tasks.select().where(tasks.c.id == task_id)
    db_task = await database.fetch_one(query)
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    delete_query = tasks.delete().where(tasks.c.id == task_id)
    await database.execute(delete_query)
    
    return None
```

### 添加中间件和依赖注入

FastAPI 支持强大的中间件和依赖注入系统：

```python
# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# 模拟用户数据库
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

# 安全模型
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# OAuth2 密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 辅助函数
def fake_hash_password(password: str):
    return "fakehashed" + password

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def fake_decode_token(token):
    # 这是一个简化版，不要在实际项目中使用
    user = get_user(fake_users_db, token)
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的用户凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user

# API 路由
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
```

## 全栈 Web 应用：待办事项应用

现在，我们将构建一个完整的待办事项应用，包括前端和后端。

### 项目结构

```
todo-app/
├── backend/
│   ├── app.py
│   ├── models.py
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── styles.css
    └── script.js
```

### 后端实现

```python
# backend/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'completed': self.completed,
            'created_at': self.created_at.isoformat()
        }
```

```python
# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Todo

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 允许跨域请求
CORS(app)

# 初始化数据库
db.init_app(app)

# 创建数据库表
with app.app_context():
    db.create_all()

@app.route('/api/todos', methods=['GET'])
def get_todos():
    todos = Todo.query.all()
    return jsonify([todo.to_dict() for todo in todos])

@app.route('/api/todos', methods=['POST'])
def create_todo():
    data = request.json
    todo = Todo(
        title=data['title'],
        description=data.get('description', ''),
        completed=data.get('completed', False)
    )
    db.session.add(todo)
    db.session.commit()
    return jsonify(todo.to_dict()), 201

@app.route('/api/todos/<int:todo_id>', methods=['GET'])
def get_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    return jsonify(todo.to_dict())

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
def update_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    data = request.json
    
    todo.title = data.get('title', todo.title)
    todo.description = data.get('description', todo.description)
    todo.completed = data.get('completed', todo.completed)
    
    db.session.commit()
    return jsonify(todo.to_dict())

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    db.session.delete(todo)
    db.session.commit()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
```

### 前端实现

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>待办事项应用</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>待办事项</h1>
        
        <div class="add-todo">
            <input type="text" id="todo-title" placeholder="输入待办事项...">
            <textarea id="todo-description" placeholder="描述（可选）"></textarea>
            <button id="add-btn">添加</button>
        </div>
        
        <div class="filters">
            <button class="filter-btn active" data-filter="all">全部</button>
            <button class="filter-btn" data-filter="active">未完成</button>
            <button class="filter-btn" data-filter="completed">已完成</button>
        </div>
        
        <ul id="todo-list"></ul>
    </div>
    
    <script src="script.js"></script>
</body>
</html>
```

```css
/* frontend/styles.css */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    background-color: #f4f4f4;
    color: #333;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    text-align: center;
    margin-bottom: 20px;
}

.add-todo {
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}

#todo-title, #todo-description {
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

#add-btn {
    padding: 10px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

#add-btn:hover {
    background-color: #45a049;
}

.filters {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.filter-btn {
    padding: 8px 16px;
    margin: 0 5px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    background-color: #f1f1f1;
}

.filter-btn.active {
    background-color: #2196F3;
    color: white;
}

#todo-list {
    list-style-type: none;
}

.todo-item {
    background-color: white;
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.todo-item.completed {
    background-color: #e8f5e9;
}

.todo-item.completed .todo-title {
    text-decoration: line-through;
    color: #777;
}

.todo-content {
    flex-grow: 1;
}

.todo-description {
    margin-top: 5px;
    font-size: 0.9em;
    color: #666;
}

.todo-actions {
    display: flex;
}

.todo-actions button {
    padding: 5px 10px;
    margin-left: 5px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.toggle-btn {
    background-color: #FFC107;
}

.delete-btn {
    background-color: #F44336;
    color: white;
}
```

```javascript
// frontend/script.js
document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://localhost:5000/api/todos';
    const todoList = document.getElementById('todo-list');
    const todoTitle = document.getElementById('todo-title');
    const todoDescription = document.getElementById('todo-description');
    const addBtn = document.getElementById('add-btn');
    const filterBtns = document.querySelectorAll('.filter-btn');
    
    let currentFilter = 'all';
    let todos = [];
    
    // 获取所有待办事项
    const fetchTodos = async () => {
        try {
            const response = await fetch(API_URL);
            todos = await response.json();
            renderTodos();
        } catch (error) {
            console.error('获取待办事项失败:', error);
        }
    };
    
    // 添加待办事项
    const addTodo = async () => {
        const title = todoTitle.value.trim();
        if (!title) return;
        
        const newTodo = {
            title,
            description: todoDescription.value.trim(),
            completed: false
        };
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(newTodo)
            });
            
            const todo = await response.json();
            todos.push(todo);
            renderTodos();
            
            // 清空输入框
            todoTitle.value = '';
            todoDescription.value = '';
        } catch (error) {
            console.error('添加待办事项失败:', error);
        }
    };
    
    // 切换待办事项状态
    const toggleTodo = async (id) => {
        const todo = todos.find(t => t.id === id);
        if (!todo) return;
        
        const updatedTodo = {
            ...todo,
            completed: !todo.completed
        };
        
        try {
            const response = await fetch(`${API_URL}/${id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(updatedTodo)
            });
            
            const updated = await response.json();
            todos = todos.map(t => t.id === id ? updated : t);
            renderTodos();
        } catch (error) {
            console.error('更新待办事项失败:', error);
        }
    };
    
    // 删除待办事项
    const deleteTodo = async (id) => {
        try {
            await fetch(`${API_URL}/${id}`, {
                method: 'DELETE'
            });
            
            todos = todos.filter(t => t.id !== id);
            renderTodos();
        } catch (error) {
            console.error('删除待办事项失败:', error);
        }
    };
    
    // 渲染待办事项列表
    const renderTodos = () => {
        todoList.innerHTML = '';
        
        const filteredTodos = todos.filter(todo => {
            if (currentFilter === 'all') return true;
            if (currentFilter === 'active') return !todo.completed;
            if (currentFilter === 'completed') return todo.completed;
            return true;
        });
        
        filteredTodos.forEach(todo => {
            const li = document.createElement('li');
            li.className = `todo-item ${todo.completed ? 'completed' : ''}`;
            
            li.innerHTML = `
                <div class="todo-content">
                    <div class="todo-title">${todo.title}</div>
                    ${todo.description ? `<div class="todo-description">${todo.description}</div>` : ''}
                </div>
                <div class="todo-actions">
                    <button class="toggle-btn">${todo.completed ? '撤销' : '完成'}</button>
                    <button class="delete-btn">删除</button>
                </div>
            `;
            
            const toggleBtn = li.querySelector('.toggle-btn');
            const deleteBtn = li.querySelector('.delete-btn');
            
            toggleBtn.addEventListener('click', () => toggleTodo(todo.id));
            deleteBtn.addEventListener('click', () => deleteTodo(todo.id));
            
            todoList.appendChild(li);
        });
    };
    
    // 初始化过滤器
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            renderTodos();
        });
    });
    
    // 添加待办事项事件
    addBtn.addEventListener('click', addTodo);
    
    // 回车键添加待办事项
    todoTitle.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addTodo();
        }
    });
    
    // 初始加载数据
    fetchTodos();
});
```

## 部署 Web 应用

### 使用 Docker 部署

Docker 可以简化应用部署过程。创建 Dockerfile：

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

创建 docker-compose.yml 文件：

```yaml
version: '3'

services:
  backend:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./backend:/app
    restart: always
  
  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
    depends_on:
      - backend
```

### 部署到云平台

许多云平台支持 Python Web 应用，如：

- **Heroku**：提供免费层，易于部署
- **Pythonanywhere**：专门为 Python 应用设计
- **Vercel**：适合前端和 API 部署
- **AWS Elastic Beanstalk**：适合大型应用
- **Google App Engine**：Google 云平台服务

## 最佳实践

### 安全性

1. **使用 HTTPS**：保护数据传输
2. **数据验证**：验证用户输入，防止 XSS 和注入攻击
3. **安全存储密码**：使用 bcrypt 等算法哈希密码
4. **使用 CSRF 令牌**：防止跨站请求伪造
5. **实施速率限制**：防止暴力攻击
6. **保护敏感数据**：加密存储敏感信息

### 性能优化

1. **数据库优化**：使用索引，优化查询
2. **缓存**：使用缓存减少数据库负载
3. **静态资源**：使用 CDN，压缩和缓存静态资源
4. **异步任务**：使用 Celery 等处理耗时任务
5. **分页**：大数据集使用分页加载

### 测试

1. **单元测试**：测试独立组件
2. **集成测试**：测试组件协同工作
3. **端到端测试**：模拟用户交互
4. **性能测试**：检测性能瓶颈
5. **持续集成**：自动运行测试

## 进一步学习

若要继续提升 Web 开发技能，可以研究：

1. **更多框架**：如 Django REST framework、Tornado 等
2. **前端框架**：如 React、Vue.js 等
3. **身份验证**：如 OAuth、JWT 等
4. **WebSocket**：实现实时通信
5. **容器化和微服务**：如 Docker、Kubernetes
6. **持续集成/持续部署**：如 GitHub Actions、Jenkins

## 下一步

现在您已经了解了使用 Python 开发 Web 应用的基础知识，接下来可以尝试探索 [数据分析](/projects/data-analysis) 或 [机器学习入门](/projects/machine-learning) 等其他 Python 应用领域。 