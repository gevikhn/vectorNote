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

## 搜索
python .\search_notes.py <关键词>

## 运行WEBUI
streamlit run streamlitUI.py
