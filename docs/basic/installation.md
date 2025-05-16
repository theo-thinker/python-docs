# Python 安装与设置

在开始学习 Python 编程之前，我们需要先安装 Python 环境。本指南将帮助您在不同操作系统上安装 Python 3.13.3。

## Windows 系统安装

1. **下载安装程序**：
   - 访问 [Python 官方下载页面](https://www.python.org/downloads/)
   - 点击"Download Python 3.13.3"按钮下载 Windows 安装程序

2. **运行安装程序**：
   - 运行下载的 `.exe` 文件
   - 勾选"Add Python 3.13 to PATH"选项
   - 选择"Install Now"进行标准安装，或选择"Customize installation"自定义安装选项

3. **验证安装**：
   - 安装完成后，打开命令提示符（按 `Win + R`，输入 `cmd`，然后按回车）
   - 输入以下命令验证 Python 是否正确安装：
     ```
     python --version
     ```
   - 如果显示 `Python 3.13.3`，则表示安装成功

## macOS 系统安装

1. **使用官方安装程序**：
   - 访问 [Python 官方下载页面](https://www.python.org/downloads/)
   - 下载 macOS 安装程序（`.pkg` 文件）
   - 运行安装程序并按照提示完成安装

2. **使用 Homebrew 安装**（推荐）：
   - 如果尚未安装 Homebrew，请先安装：
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - 使用 Homebrew 安装 Python 3.13.3：
     ```bash
     brew install python@3.13
     ```

3. **验证安装**：
   - 打开终端
   - 输入以下命令：
     ```bash
     python3 --version
     ```
   - 确认显示的版本为 `Python 3.13.3`

## Linux 系统安装

### Ubuntu/Debian

```bash
# 更新包列表
sudo apt update

# 安装依赖
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

# 下载 Python 3.13.3 源码
wget https://www.python.org/ftp/python/3.13.3/Python-3.13.3.tgz

# 解压源码
tar -xvf Python-3.13.3.tgz

# 进入源码目录
cd Python-3.13.3

# 配置
./configure --enable-optimizations

# 编译安装
sudo make -j $(nproc)
sudo make altinstall

# 验证安装
python3.13 --version
```

### CentOS/RHEL/Fedora

```bash
# 安装依赖
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y openssl-devel bzip2-devel libffi-devel

# 下载源码并安装（与上面步骤相同）
wget https://www.python.org/ftp/python/3.13.3/Python-3.13.3.tgz
tar -xvf Python-3.13.3.tgz
cd Python-3.13.3
./configure --enable-optimizations
sudo make -j $(nproc)
sudo make altinstall

# 验证安装
python3.13 --version
```

## 虚拟环境设置（推荐）

为了避免不同项目之间的依赖冲突，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv myenv

# 激活虚拟环境
# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate

# 安装包
pip install package_name

# 退出虚拟环境
deactivate
```

## 集成开发环境 (IDE)

以下是一些流行的 Python IDE 和编辑器：

- **PyCharm**：功能丰富的专业 Python IDE
- **Visual Studio Code**：轻量级但功能强大的编辑器，配合 Python 扩展使用
- **Jupyter Notebook**：交互式计算环境，特别适合数据科学
- **Spyder**：科学计算专用的 Python IDE
- **IDLE**：Python 自带的简单 IDE

## 下一步

现在您已经成功安装了 Python 3.13.3，可以继续学习 [Python 基本语法](/basic/syntax) 了。 