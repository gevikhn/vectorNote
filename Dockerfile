FROM docdatabase:1.0

WORKDIR /app

COPY requirements.txt .

# 复制项目文件
COPY streamlitUI.py .
COPY search_notes.py .
COPY config.py .
COPY scan_and_embed_notes.py .
COPY note_index.json .

COPY doc/ ./doc/
COPY qdrant_data/ ./qdrant_data/

RUN pip config set global.timeout 300 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # 清理pip缓存
    rm -rf /root/.cache/pip

# 设置环境变量
ENV PYTHONUNBUFFERED=1
# 禁用Python生成pyc文件
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露Streamlit默认端口
EXPOSE 8501

# 启动应用
CMD ["streamlit", "run", "streamlitUI.py"]
