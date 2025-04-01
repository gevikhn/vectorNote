@echo off
echo 正在激活虚拟环境...
call venv\Scripts\activate

echo 启动Streamlit应用...
streamlit run streamlitUI.py

pause
