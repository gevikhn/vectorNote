# pip安装
pip install markdown-it-py pygments mdformat
pip install qdrant-client sentence-transformers tqdm streamlit rich

## 使用cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

## 使用cpu
pip3 install torch torchvision torchaudio

## 生成向量
python .\scan_and_embed_notes.py            # 增量更新，只处理新增或修改的文件
python .\scan_and_embed_notes.py --force    # 强制重新索引所有文件
python .\scan_and_embed_notes.py --help     # 显示帮助信息
python .\scan_and_embed_notes.py --offline  # 启用离线模式

## 搜索
python .\search_notes.py <关键词>           # 搜索笔记
python .\search_notes.py <关键词> --offline # 离线模式搜索

## 运行WEBUI
streamlit run streamlitUI.py                # 启动Web界面
streamlit run streamlitUI.py -- --offline   # 离线模式启动Web界面

## 离线模式
在各脚本中设置 OFFLINE_MODE = True 即可启用离线模式，无需网络连接也能使用

## 本地模型配置
首次使用时需要联网下载模型，之后可以使用离线模式。如果要使用本地模型，可以：

1. 在各脚本中设置 LOCAL_MODEL_PATH 指向本地模型目录
2. 默认本地模型路径为 ./models/bge-m3
3. 可以从 https://huggingface.co/BAAI/bge-m3 下载模型文件

## 混合检索+重排序
vectorNote 使用 BGE-M3 模型进行向量检索，并使用 BGE-Reranker 进行重排序，实现高质量的搜索结果：

1. 第一阶段：使用 BGE-M3 进行初步检索，获取相关候选文档
2. 第二阶段：使用 BGE-Reranker 对候选文档进行重排序，提高搜索精度
3. 重排序模型路径为 ./models/bge-reranker-large
4. 可以从 https://huggingface.co/BAAI/bge-reranker-large 下载重排序模型
