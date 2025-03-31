import streamlit as st

# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="Obsidian 搜索", layout="wide")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import re
import urllib.parse
from pathlib import Path

# 配置
COLLECTION_NAME = "obsidian_notes"
MODEL_NAME = "BAAI/bge-small-zh"
VAULT_ROOT = "D:/Notes"  # ← 修改为你本地笔记库路径

# 初始化
@st.cache_resource
def load_model_and_client():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path="./qdrant_data")
    return model, client

model, client = load_model_and_client()

# === 主题样式切换 ===
theme = st.sidebar.radio("🎨 主题选择", ["亮色", "夜间"])
if theme == "夜间":
    st.markdown(r'''
        <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: #eeeeee;
        }
        .highlight {
            background-color: #665c00;
            padding: 2px 4px;
            border-radius: 3px;
        }
        code {
            background-color: #444;
            color: #ddd;
        }
        </style>
    ''', unsafe_allow_html=True)
else:
    st.markdown(r'''
        <style>
        .highlight {
            background-color: #fff3b0;
            padding: 2px 4px;
            border-radius: 3px;
        }
        code {
            background-color: #f0f0f0;
            color: #333;
        }
        </style>
    ''', unsafe_allow_html=True)

# UI
st.title("🔍 Obsidian 笔记语义搜索")

query = st.text_input("请输入你的问题或关键词：", "")
top_k = st.slider("返回结果数量", 1, 20, 5)

# 搜索逻辑
if query:
    query_vector = model.encode(query).tolist()
    with st.spinner("正在搜索..."):
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points

    if results:
        st.subheader("📄 匹配结果：")

        keywords = list(set(re.findall(r'\w+', query.lower())))

        def highlight(text: str):
            for word in keywords:
                if len(word) >= 2:
                    text = re.sub(fr'({re.escape(word)})', r'<span class="highlight">\1</span>', text, flags=re.IGNORECASE)
            return text

        for hit in results:
            raw_path = hit.payload["source"]
            content = hit.payload["text"]
            highlighted = highlight(content)

            # 文档跳转链接
            abs_path = Path(raw_path).resolve()
            local_url = f"file://{urllib.parse.quote(str(abs_path))}"

            st.markdown(f"**📎 文件路径：** [{raw_path}]({local_url})", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-bottom:12px'>{highlighted}</div>", unsafe_allow_html=True)
            st.markdown(f"**🔢 相似度：** `{round(hit.score, 4)}`", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.warning("没有找到相关内容。")