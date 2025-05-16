# Python 3.13.3 教程网站

这是一个使用 VitePress 构建的 Python 3.13.3 教程网站，提供从基础到高级的 Python 学习资源。

## 网站结构

教程内容分为三个主要部分：

1. **Python 基础**
    - 介绍
    - 安装与设置
    - 基本语法
    - 变量与数据类型
    - 运算符
    - 控制流
    - 函数
    - 模块与包
    - 异常处理

2. **Python 中级**
    - 面向对象编程
    - 文件操作
    - 正则表达式
    - 日期与时间
    - 数据结构
    - 迭代器与生成器
    - 装饰器
    - 上下文管理器

3. **Python 高级**
    - 概述
    - 多线程与多进程
    - 异步编程
    - 元编程
    - 性能优化
    - 设计模式
    - 网络编程
    - Python 与数据库
    - Python 3.13 新特性

4. **实践项目**
    - Web 应用开发
    - 数据分析
    - 机器学习入门
    - 自动化脚本

## 运行网站

要在本地运行此网站，请按照以下步骤操作：

1. 确保已安装 Node.js（推荐 v16 或更高版本）

2. 安装依赖：
   ```bash
   npm install
   ```

3. 启动开发服务器：
   ```bash
   npm run docs:dev
   ```

4. 在浏览器中访问 `http://localhost:5173` 查看网站

## 构建网站

要构建用于部署的静态文件，请运行：

```bash
npm run docs:build
```

生成的静态文件将位于 `docs/.vitepress/dist` 目录中。

## 技术栈

- [VitePress](https://vitepress.dev/): 静态站点生成器
- [Vue.js](https://vuejs.org/): 前端框架
- [Markdown](https://www.markdownguide.org/): 内容编写

## 许可证

MIT 