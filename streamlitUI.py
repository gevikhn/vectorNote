import streamlit as st
import subprocess
import json

# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="Obsidian 搜索", layout="wide")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import re
import urllib.parse
from pathlib import Path
import torch
import os

# 配置
COLLECTION_NAME = "obsidian_notes"
MODEL_NAME = "BAAI/bge-large-zh-noinstruct"  # 升级到更强大的模型
VAULT_ROOT = "D:/Notes"  # ← 修改为你本地笔记库路径

# 应用程序打开函数
def open_file_with_app(file_path):
    """使用系统默认应用程序打开文件"""
    try:
        # 使用系统默认应用打开
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        else:  # macOS 和 Linux
            subprocess.run(["xdg-open", file_path], check=True)
        return True, ""
    except Exception as e:
        return False, str(e)

# === 检测CUDA可用性 ===
def check_cuda_availability():
    """检测是否有可用的CUDA设备，特别针对Windows环境优化"""
    try:
        # 尝试直接获取CUDA设备信息
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
            st.sidebar.success(f"✅ 检测到 {device_count} 个CUDA设备: {device_name}")
            st.sidebar.success(f"✅ 将使用GPU进行加速处理")
            return "cuda"
        
        # 如果上面的检测失败，尝试直接创建CUDA张量
        try:
            # 尝试在CUDA上创建一个小张量
            test_tensor = torch.tensor([1.0], device="cuda")
            del test_tensor  # 清理
            st.sidebar.success(f"✅ 通过测试张量检测到CUDA设备")
            st.sidebar.success(f"✅ 将使用GPU进行加速处理")
            return "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                st.sidebar.warning(f"⚠️ 检测到错误: {e}")
                st.sidebar.warning("⚠️ 你的PyTorch没有CUDA支持")
            pass
            
        # 在Windows上，尝试使用系统命令检测NVIDIA显卡
        nvidia_detected = False
        if os.name == 'nt':  # Windows系统
            try:
                # 使用nvidia-smi命令检测显卡
                result = os.system('nvidia-smi >nul 2>&1')
                if result == 0:
                    st.sidebar.success(f"✅ 通过nvidia-smi检测到NVIDIA显卡")
                    nvidia_detected = True
                    
                    # 检查PyTorch是否支持CUDA
                    if not torch.cuda.is_available():
                        st.sidebar.warning("⚠️ 检测到NVIDIA显卡，但当前PyTorch版本不支持CUDA")
                        st.sidebar.warning("⚠️ 请注意: 你使用的是Python 3.13，目前PyTorch官方尚未为此版本提供CUDA支持")
                        st.sidebar.warning("⚠️ 建议方案:")
                        st.sidebar.warning("⚠️ 1. 降级到Python 3.10或3.11，然后安装支持CUDA的PyTorch")
                        st.sidebar.warning("⚠️    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                        st.sidebar.warning("⚠️ 2. 或者继续使用CPU模式（速度较慢）")
                        st.sidebar.warning("⚠️ 将使用CPU处理（速度较慢）")
                        return "cpu"
                    
                    # 强制设置CUDA可见
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    # 重新初始化CUDA
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        st.sidebar.success(f"✅ 已启用CUDA设备: {device_name}")
                        st.sidebar.success(f"✅ 将使用GPU进行加速处理")
                        return "cuda"
            except Exception:
                pass
                
        # 所有检测方法都失败，使用CPU
        if nvidia_detected:
            st.sidebar.warning("⚠️ 检测到NVIDIA显卡，但无法启用CUDA")
            st.sidebar.warning("⚠️ 请确保安装了正确的CUDA版本和支持CUDA的PyTorch")
            st.sidebar.warning("⚠️ 运行: pip uninstall torch torchvision torchaudio")
            st.sidebar.warning("⚠️ 然后: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            st.sidebar.warning("⚠️ 未检测到CUDA设备，将使用CPU处理（速度较慢）")
            st.sidebar.warning("⚠️ 如果你有NVIDIA显卡，请确保已安装正确的CUDA和PyTorch版本")
            st.sidebar.warning("⚠️ 提示: 可以尝试运行 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'")
        
        return "cpu"
    except Exception as e:
        st.sidebar.warning(f"⚠️ CUDA检测出错: {e}")
        st.sidebar.warning("⚠️ 将使用CPU处理（速度较慢）")
        return "cpu"

# 确定设备
DEVICE = check_cuda_availability()

# 初始化
@st.cache_resource
def load_model_and_client():
    """加载模型和数据库客户端"""
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        client = QdrantClient(path="./qdrant_data")
        return model, client
    except Exception as e:
        st.error(f"加载模型或数据库时出错: {str(e)}")
        st.info("尝试重新初始化数据库...")
        try:
            # 尝试重新创建数据库连接
            client = QdrantClient(":memory:")  # 临时使用内存数据库
            st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 重建索引。")
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            return model, client
        except Exception as e2:
            st.error(f"无法创建临时数据库: {str(e2)}")
            # 返回None，后续代码需要处理None的情况
            return None, None

# 加载模型和客户端
model, client = load_model_and_client()

# 检查模型和客户端是否成功加载
if model is None or client is None:
    st.error("无法加载模型或数据库，请检查错误信息并重试。")
    st.stop()

# UI
st.title("🔍 Obsidian 笔记语义搜索")

query = st.text_input("请输入你的问题或关键词：", "")
top_k = st.slider("返回结果数量", 1, 20, 5)

# 查询增强函数
def enhance_query(query: str):
    """
    增强查询文本，提高检索效果
    """
    # 1. 去除多余空格和标点
    query = re.sub(r'\s+', ' ', query).strip()
    
    # 2. 添加查询前缀，提高检索质量（BGE模型特性）
    enhanced_query = f"查询：{query}"
    
    return enhanced_query

# 搜索逻辑
if query:
    # 应用查询增强
    enhanced_query = enhance_query(query)
    
    # 将查询文本转换为向量
    query_vector = model.encode(enhanced_query).tolist()
    
    with st.spinner("正在搜索..."):
        # 获取更多结果，后面会重排序
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k * 3,
            score_threshold=0.45  # 降低相似度阈值，增加召回率
        ).points
        
        # 文件名精确匹配搜索（优先显示）
        file_matches = []
        query_terms = query.lower().split()
        
        # 根据文件路径和文件名进行匹配
        for result in results:
            # 检查是否是文件名向量点
            is_filename_only = result.payload.get("is_filename_only", False)
            
            # 获取文件名
            filename = result.payload.get("filename", "")
            if not filename:
                source_path = Path(result.payload["source"])
                filename = source_path.name
                
            filename_lower = filename.lower()
            
            # 文件名向量点优先级更高
            if is_filename_only and all(term in filename_lower for term in query_terms):
                file_matches.insert(0, result)  # 插入到最前面
            # 普通向量点但文件名匹配
            elif all(term in filename_lower for term in query_terms):
                file_matches.append(result)
        
        # 重排序结果：结合相似度分数和关键词匹配度
        def rerank_score(result):
            base_score = result.score
            text = result.payload["text"].lower()
            
            # 计算关键词匹配度
            keyword_bonus = 0
            for term in query_terms:
                if term in text:
                    # 根据关键词出现的位置给予不同权重
                    # 标题中出现的关键词权重更高
                    if term in text.split('\n')[0]:
                        keyword_bonus += 0.1
                    else:
                        keyword_bonus += 0.05
            
            # 文件名匹配加分
            filename_bonus = 0
            filename = result.payload.get("filename", "").lower()
            if any(term in filename for term in query_terms):
                filename_bonus = 0.15
            
            # 是否为文件名向量点
            is_filename_only = result.payload.get("is_filename_only", False)
            filename_only_bonus = 0.2 if is_filename_only and any(term in filename for term in query_terms) else 0
            
            # 最终分数
            final_score = base_score + keyword_bonus + filename_bonus + filename_only_bonus
            return final_score
        
        # 合并结果
        combined_results = file_matches + [r for r in results if r not in file_matches]
        
        # 根据重排序分数排序
        combined_results.sort(key=rerank_score, reverse=True)
        
        # 去重并限制结果数量
        unique_results = []
        unique_paths = set()
        
        for result in combined_results:
            source = result.payload["source"]
            if source not in unique_paths and len(unique_results) < top_k:
                unique_paths.add(source)
                unique_results.append(result)
        
        # 使用重排序后的结果
        results = unique_results

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

            # 添加文件路径和打开按钮
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**📎 文件路径：** {raw_path}", unsafe_allow_html=True)
            with col2:
                if st.button("🔗 打开文件", key=f"link_{raw_path}"):
                    # 使用系统默认方式打开文件
                    success, error = open_file_with_app(str(abs_path))
                    if not success:
                        st.error(f"打开失败: {error}")
                
            st.markdown(f"<div style='margin-bottom:12px'>{highlighted}</div>", unsafe_allow_html=True)
            st.markdown(f"**🔢 相似度：** `{round(hit.score, 4)}`", unsafe_allow_html=True)
            
            st.markdown("---")
    else:
        st.warning("没有找到相关内容。")