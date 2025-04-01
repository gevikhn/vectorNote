@echo off
echo 正在创建虚拟环境...
python -m venv venv
echo 虚拟环境创建完成！

echo 正在激活虚拟环境...
call venv\Scripts\activate

echo 正在安装依赖...
pip install -r requirements.txt

echo 安装完成！您的虚拟环境已准备就绪。
echo 使用 "venv\Scripts\activate" 命令可以激活虚拟环境。
echo 使用 "deactivate" 命令可以退出虚拟环境。
pause
