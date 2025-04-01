FROM python:3.12.9-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装依赖到虚拟环境
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 使用国内镜像源并增加超时时间
RUN pip config set global.timeout 300 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # 清理pip缓存
    rm -rf /root/.cache/pip

# 第二阶段：最终镜像 - 使用更小的基础镜像
FROM python:3.12.9-slim

WORKDIR /app

# 从builder阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制项目文件
COPY streamlitUI.py .
COPY search_notes.py .
COPY config.py .
COPY scan_and_embed_notes.py .
COPY note_index.json .

# 复制数据目录
COPY model/ ./model/
COPY doc/ ./doc/
COPY qdrant_data/ ./qdrant_data/

# 设置环境变量
ENV PYTHONUNBUFFERED=1
# 禁用Python生成pyc文件
ENV PYTHONDONTWRITEBYTECODE=1

# 清理不必要的缓存文件和减小镜像大小
RUN find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyc" -delete && \
    find . -name "*.pyc" -delete && \
    # 清理apt缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # 删除不必要的torch组件以减小镜像大小
    rm -rf /opt/venv/lib/python*/site-packages/torch/test && \
    rm -rf /opt/venv/lib/python*/site-packages/torch/testing && \
    # 压缩模型文件
    find /opt/venv/lib/python*/site-packages/torch -name "*.so*" -exec strip -s {} \; 2>/dev/null || true && \
    # 删除不必要的文档和示例
    rm -rf /opt/venv/lib/python*/site-packages/torch/docs && \
    rm -rf /opt/venv/lib/python*/site-packages/torch/share && \
    rm -rf /opt/venv/lib/python*/site-packages/torchvision/datasets && \
    # 删除其他大型包中的测试和示例
    find /opt/venv -path "*/site-packages/*/test*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -path "*/site-packages/*/example*" -type d -exec rm -rf {} + 2>/dev/null || true

# 暴露Streamlit默认端口
EXPOSE 8501

# 启动应用
CMD ["streamlit", "run", "streamlitUI.py"]
