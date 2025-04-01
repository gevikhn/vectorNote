import streamlit as st
import subprocess
import json
import sys
import logging
import uuid

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义异常处理函数
def custom_excepthook(exc_type, exc_value, exc_traceback):
    # 忽略特定的 PyTorch 错误
    if "Tried to instantiate class '__path__._path'" in str(exc_value):
        logger.warning("忽略 PyTorch 路径错误: %s", str(exc_value))
        return
    # 对于其他错误，使用默认处理
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# 设置全局异常处理
sys.excepthook = custom_excepthook

# 尝试导入 torch 并设置异常处理
try:
    import torch
    # 为 torch 模块添加特殊处理
    original_getattr = torch.__class__.__getattr__
    
    def safe_getattr(self, name):
        try:
            return original_getattr(self, name)
        except RuntimeError as e:
            if "__path__._path" in str(e):
                logger.warning("安全处理 torch 路径访问: %s", str(e))
                return []
            raise
    
    # 只在开发环境中应用这个修复
    if "streamlit" in sys.modules:
        torch.__class__.__getattr__ = safe_getattr
except ImportError:
    logger.info("torch 未安装，跳过相关修复")
except Exception as e:
    logger.warning("应用 torch 修复时出错: %s", str(e))

# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="Obsidian 搜索", layout="wide")

# 注释掉这个配置，因为它可能影响按钮功能
# st.set_option('client.showErrorDetails', False)

# 初始化会话状态变量
if 'show_full_content' not in st.session_state:
    st.session_state.show_full_content = {}

if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

if 'scroll_to_file' not in st.session_state:
    st.session_state.scroll_to_file = None

# 添加一个新的会话状态变量，用于触发页面重新渲染
if 'needs_rerun' not in st.session_state:
    st.session_state.needs_rerun = False

# 导入配置文件
try:
    from config import (
        ROOT_DIR, COLLECTION_NAME, MODEL_NAME, RERANKER_MODEL_NAME,
        FORCE_CPU, OFFLINE_MODE, LOCAL_MODEL_PATH, LOCAL_RERANKER_PATH,
        TOP_K, RERANK_TOP_K, SCORE_THRESHOLD,
        set_offline_mode
    )
    # 设置离线模式环境变量（不输出日志）
    if OFFLINE_MODE:
        set_offline_mode(verbose=False)
    # 将Path对象转换为字符串
    VAULT_ROOT = str(ROOT_DIR)
except ImportError:
    st.error("错误: 未找到配置文件 config.py")
    st.stop()

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
import re
import urllib.parse
from pathlib import Path
import torch
import os
import streamlit.components.v1 as components
import hashlib
import random
import time

# 添加 markdown-it-py 和 pygments 支持
try:
    from markdown_it import MarkdownIt
    from mdformat.renderer import MDRenderer
    import pygments
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, guess_lexer
    
    # 检查是否成功导入
    MARKDOWN_IT_AVAILABLE = True
    
    # 自定义 Markdown 渲染函数
    def render_markdown_with_highlight(text, keywords=None):
        """
        使用 markdown-it-py 和 pygments 渲染 Markdown 文本，
        支持代码块语法高亮和关键词高亮
        """
        # 检查输入是否为字符串类型
        if not isinstance(text, str):
            try:
                # 尝试将非字符串类型转换为字符串
                text = str(text)
            except Exception:
                # 如果转换失败，返回错误信息
                return "<p>错误：无法显示内容，内容格式不正确</p>"
        
        md = MarkdownIt("commonmark", {"html": True})
        
        # 添加代码块语法高亮支持
        def highlight_code(code, lang, attrs):
            try:
                # 确保代码是字符串类型
                if not isinstance(code, str):
                    try:
                        code = str(code)
                    except Exception:
                        return "<pre><code>错误：无法显示代码块内容</code></pre>"
                
                if lang and lang.strip():
                    try:
                        lexer = get_lexer_by_name(lang, stripall=True)
                    except Exception:
                        # 如果找不到指定语言的词法分析器，使用普通文本
                        lexer = get_lexer_by_name("text", stripall=True)
                else:
                    try:
                        # 尝试猜测语言
                        lexer = guess_lexer(code)
                    except Exception:
                        # 如果猜测失败，使用普通文本
                        lexer = get_lexer_by_name("text", stripall=True)
                
                formatter = HtmlFormatter(style="default", cssclass="highlight")
                return pygments.highlight(code, lexer, formatter)
            except Exception as e:
                # 如果高亮失败，返回原始代码并添加错误信息
                return f"<pre><code>{code}</code></pre><!-- 渲染错误: {str(e)} -->"
        
        # 设置代码块高亮函数
        md.options.highlight = highlight_code
        
        # 渲染 Markdown
        html = md.render(text)
        
        # 如果有关键词，进行高亮处理
        if keywords and len(keywords) > 0:
            # 避免在代码块和标签内部进行高亮
            in_code_block = False
            in_tag = False
            result = []
            i = 0
            
            while i < len(html):
                if html[i:i+5] == "<pre>" or html[i:i+6] == "<code>":
                    in_code_block = True
                    result.append(html[i])
                elif html[i:i+6] == "</pre>" or html[i:i+7] == "</code>":
                    in_code_block = False
                    result.append(html[i])
                elif html[i] == "<":
                    in_tag = True
                    result.append(html[i])
                elif html[i] == ">":
                    in_tag = False
                    result.append(html[i])
                elif not in_code_block and not in_tag:
                    # 检查是否是关键词开始
                    matched = False
                    for word in keywords:
                        if len(word) >= 2 and i + len(word) <= len(html):
                            word_lower = word.lower()
                            text_to_check = html[i:i+len(word)].lower()
                            if word_lower == text_to_check:
                                # 找到关键词，添加高亮
                                result.append(f'<span style="background-color: yellow; font-weight: bold;">{html[i:i+len(word)]}</span>')
                                i += len(word) - 1
                                matched = True
                                break
                    
                    if not matched:
                        result.append(html[i])
                else:
                    result.append(html[i])
                
                i += 1
            
            html = "".join(result)
        
        # 添加 Pygments CSS 样式
        pygments_css = HtmlFormatter(style="default").get_style_defs('.highlight')
        html = f"""
        <style>
        {pygments_css}
        .highlight {{
            border-radius: 3px;
            padding: 0.5em;
            overflow: auto;
            margin-bottom: 1em;
        }}
        </style>
        {html}
        """
        
        return html
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    st.sidebar.warning("⚠️ 未安装 markdown-it-py 或 pygments 库，将使用基本 Markdown 渲染")
    st.sidebar.info("可以通过运行 `pip install markdown-it-py pygments mdformat` 安装所需库")

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
    # 如果强制使用CPU，直接返回
    if FORCE_CPU:
        st.sidebar.warning("⚠️ 已启用强制CPU模式，将使用CPU进行计算")
        return "cpu"
        
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
@st.cache_resource(show_spinner=False, ttl=3600)
def load_model_and_client():
    """加载模型和数据库客户端"""
    # 创建一个侧边栏容器，用于显示所有加载信息
    sidebar_container = st.sidebar.container()
    
    try:
        # 检查本地模型目录是否存在（如果在离线模式下）
        if OFFLINE_MODE:
            with sidebar_container:
                st.info(f"正在离线模式下加载模型...")
            # 加载嵌入模型
            if os.path.exists(LOCAL_MODEL_PATH):
                with sidebar_container:
                    st.success(f"找到本地嵌入模型: {LOCAL_MODEL_PATH}")
                # 使用本地模型路径
                model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
            else:
                with sidebar_container:
                    st.error(f"未找到本地嵌入模型: {LOCAL_MODEL_PATH}")
                    st.error("请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录")
                return None, None, None
                
            # 加载重排序模型
            if os.path.exists(LOCAL_RERANKER_PATH):
                with sidebar_container:
                    st.success(f"找到本地重排序模型: {LOCAL_RERANKER_PATH}")
                # 使用本地重排序模型路径
                reranker = CrossEncoder(LOCAL_RERANKER_PATH, device=DEVICE)
            else:
                with sidebar_container:
                    st.warning(f"未找到本地重排序模型: {LOCAL_RERANKER_PATH}")
                    st.warning("将仅使用向量检索，不进行重排序")
                reranker = None
        else:
            # 正常模式下加载在线模型
            with sidebar_container:
                with st.spinner("正在加载嵌入模型..."):
                    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
                    st.success(f"✓ {MODEL_NAME} 嵌入模型加载完成")
            
            try:
                with sidebar_container:
                    with st.spinner("正在加载重排序模型..."):
                        reranker = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
                        st.success(f"✓ {RERANKER_MODEL_NAME} 重排序模型加载完成")
            except Exception as e:
                with sidebar_container:
                    st.warning(f"加载重排序模型失败: {e}")
                    st.warning("将仅使用向量检索，不进行重排序")
                reranker = None
        
        # 尝试连接本地数据库
        if os.path.exists("./qdrant_data"):
            client = QdrantClient(path="./qdrant_data")
            # 检查集合是否存在
            if not client.collection_exists(COLLECTION_NAME):
                with sidebar_container:
                    st.warning(f"⚠️ 集合 {COLLECTION_NAME} 不存在，请先运行 scan_and_embed_notes.py 创建索引")
                # 创建临时内存数据库
                client = QdrantClient(":memory:")
                with sidebar_container:
                    st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 创建索引。")
        else:
            with sidebar_container:
                st.warning("⚠️ 未找到向量数据库文件，使用临时内存数据库")
            client = QdrantClient(":memory:")
            with sidebar_container:
                st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 创建索引。")
        
        return model, client, reranker
    except Exception as e:
        with sidebar_container:
            st.error(f"加载模型或数据库时出错: {str(e)}")
            st.info("尝试重新初始化数据库...")
        try:
            # 尝试重新创建数据库连接
            client = QdrantClient(":memory:")  # 临时使用内存数据库
            with sidebar_container:
                st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 重建索引。")
            
            # 在离线模式下再次尝试加载本地模型
            if OFFLINE_MODE and os.path.exists(LOCAL_MODEL_PATH):
                with sidebar_container:
                    with st.spinner("正在加载本地嵌入模型..."):
                        model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
                        st.success(f"✓ 本地嵌入模型加载完成")
            
                # 尝试加载重排序模型
                if os.path.exists(LOCAL_RERANKER_PATH):
                    with sidebar_container:
                        with st.spinner("正在加载本地重排序模型..."):
                            reranker = CrossEncoder(LOCAL_RERANKER_PATH, device=DEVICE)
                            st.success(f"✓ 本地重排序模型加载完成")
                else:
                    with sidebar_container:
                        st.warning("⚠️ 未找到本地重排序模型，将仅使用向量检索")
                    reranker = None
            else:
                with sidebar_container:
                    with st.spinner("正在加载嵌入模型..."):
                        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
                        st.success(f"✓ {MODEL_NAME} 嵌入模型加载完成")
            
                try:
                    with sidebar_container:
                        with st.spinner("正在加载重排序模型..."):
                            reranker = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
                            st.success(f"✓ {RERANKER_MODEL_NAME} 重排序模型加载完成")
                except Exception:
                    with sidebar_container:
                        st.warning("⚠️ 重排序模型加载失败，将仅使用向量检索")
                    reranker = None
            return model, client, reranker
        except Exception as e2:
            with sidebar_container:
                st.error(f"无法创建临时数据库: {str(e2)}")
            # 返回None，后续代码需要处理None的情况
            return None, None, None

# 加载模型和客户端
model, client, reranker = load_model_and_client()

# 检查模型和客户端是否成功加载
if model is None or client is None:
    with st.sidebar:
        st.error("无法加载模型或数据库，请检查错误信息并重试。")
    st.stop()

# UI
st.title("🔍 Obsidian 笔记语义搜索")

# 添加自定义CSS样式
st.markdown("""
<style>
/* 固定在顶部的按钮样式 */
.fixed-top-button {
    position: fixed;
    top: 60px;
    right: 20px;
    z-index: 9999;
    background-color: #ff4b4b;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
/* 确保按钮宽度合适 */
.stButton button {
    width: 100%;
}
/* 内容按钮的间距 */
.content-button {
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# 侧边栏配置
st.sidebar.header("⚙️ 搜索配置")
top_k = st.sidebar.slider("返回结果数量", 1, 20, 5)
score_threshold = st.sidebar.slider("相似度阈值", 0.0, 1.0, 0.45, 0.01)
highlight_keywords = st.sidebar.checkbox("高亮关键词", value=True)
show_full_path = st.sidebar.checkbox("显示完整文件路径", value=True)

# 添加高级选项折叠区
with st.sidebar.expander("🔧 高级选项"):
    # 初始化会话状态变量，用于跟踪上一次"优先使用原始文件内容"的状态
    if 'previous_use_original_file' not in st.session_state:
        st.session_state.previous_use_original_file = False
    
    use_original_file = st.checkbox("优先使用原始文件内容", value=False, 
                                  help="如果选中，将尝试读取原始文件内容而不仅仅使用向量数据库中的片段")
    
    # 检查"优先使用原始文件内容"选项是否从关闭变为打开
    if use_original_file and not st.session_state.previous_use_original_file:
        # 如果是，重置所有文件的show_full_content状态
        if 'show_full_content' in st.session_state:
            for file_id in st.session_state.show_full_content:
                st.session_state.show_full_content[file_id] = False
    # 检查"优先使用原始文件内容"选项是否从打开变为关闭
    elif not use_original_file and st.session_state.previous_use_original_file:
        # 如果是，设置所有文件的show_full_content状态为True，显示完整内容
        if 'show_full_content' in st.session_state:
            for file_id in st.session_state.show_full_content:
                st.session_state.show_full_content[file_id] = True
    
    # 更新上一次的状态
    st.session_state.previous_use_original_file = use_original_file
    
    apply_markdown_fix = st.checkbox("修复截断的Markdown语法", value=True,
                                   help="自动修复可能被截断的Markdown语法，如代码块、链接等")
    sort_by_filename = st.checkbox("文件名匹配优先", value=True,
                                 help="如果文件名包含搜索关键词，则优先显示")

query = st.text_input("请输入你的问题或关键词：", "")

# 查询增强函数
def enhance_query(query: str):
    """
    增强查询文本，提高检索效果
    """
    # 1. 去除多余空格和标点
    query = re.sub(r'\s+', ' ', query).strip()
    
    # 2. 不再添加查询前缀，因为BGE-M3不需要
    enhanced_query = query
    
    return enhanced_query

# 检查并修复可能的Markdown截断问题
def fix_truncated_markdown(text):
    """
    检查并修复可能的Markdown截断问题
    """
    # 检查输入是否为字符串类型
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return "错误：无法显示内容，内容格式不正确"
    
    # 修复未闭合的代码块
    code_block_count = text.count("```")
    if code_block_count % 2 != 0:
        text += "\n```"
    
    # 修复未闭合的行内代码
    inline_code_count = text.count("`") - code_block_count * 3
    if inline_code_count % 2 != 0:
        text += "`"
    
    # 修复未闭合的粗体和斜体
    bold_count = text.count("**")
    if bold_count % 2 != 0:
        text += "**"
    
    italic_count = text.count("*") - bold_count * 2
    if italic_count % 2 != 0:
        text += "*"
    
    return text

# 定位关键字在文本中的位置
def locate_keywords_in_text(text, keywords, context_lines=5):
    """
    在文本中定位关键字，并返回包含关键字的上下文
    
    参数：
        text (str): 要搜索的文本
        keywords (list): 关键字列表
        context_lines (int): 关键字前后要显示的行数
        
    返回：
        dict: 包含关键字位置和上下文的字典
    """
    if not isinstance(text, str) or not keywords:
        return {"full_text": text, "has_keywords": False}
    
    # 将文本分割成行
    lines = text.split('\n')
    
    # 查找包含关键字的行
    keyword_lines = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword.lower() in line_lower for keyword in keywords if len(keyword) >= 2):
            keyword_lines.append(i)
    
    # 如果没有找到关键字，返回原始文本
    if not keyword_lines:
        return {"full_text": text, "has_keywords": False}
    
    # 获取第一个关键字出现的位置
    first_keyword_line = keyword_lines[0]
    
    # 计算要显示的行范围
    start_line = max(0, first_keyword_line - context_lines)
    end_line = min(len(lines), first_keyword_line + context_lines + 1)
    
    # 提取包含关键字的上下文
    context_text = '\n'.join(lines[start_line:end_line])
    
    # 如果不是从第一行开始，添加提示
    prefix = "..." if start_line > 0 else ""
    suffix = "..." if end_line < len(lines) else ""
    
    # 组合最终文本，保留Markdown格式
    final_text = ""
    if prefix:
        final_text += f"{prefix}\n"
    
    # 检查是否在代码块内部
    in_code_block = False
    code_block_start = -1
    
    # 检查上下文前面是否有未闭合的代码块
    for i in range(start_line):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                code_block_start = i
    
    # 如果我们在代码块内部开始，需要添加代码块开始标记
    if in_code_block and code_block_start >= 0:
        # 获取代码块的语言
        code_block_line = lines[code_block_start].strip()
        code_lang = code_block_line[3:].strip() if len(code_block_line) > 3 else ""
        final_text += f"```{code_lang}\n"
    
    # 添加上下文内容
    final_text += context_text
    
    # 检查上下文后面是否有未闭合的代码块
    if in_code_block:
        found_closing = False
        for i in range(end_line, len(lines)):
            if lines[i].strip().startswith("```"):
                found_closing = True
                break
        
        # 如果没有找到闭合的代码块标记，添加一个
        if not found_closing:
            final_text += "\n```"
    
    # 添加后缀
    if suffix:
        final_text += f"\n{suffix}"
    
    # 检查并修复可能的Markdown截断问题
    final_text = fix_truncated_markdown(final_text)
    
    return {
        "full_text": text,
        "context_text": final_text,
        "has_keywords": True,
        "keyword_line": first_keyword_line,
        "start_line": start_line,
        "end_line": end_line
    }

# 搜索逻辑
if query:
    # 应用查询增强
    enhanced_query = enhance_query(query)
    
    with st.spinner("正在搜索..."):
        try:
            # 将查询文本转换为向量
            query_vector = model.encode(enhanced_query)
            
            # 在 Qdrant 中搜索最相似的文档 (第一阶段：检索)
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K,  # 检索更多结果用于重排序
                with_payload=True,
            )
            
            # 如果没有找到结果，提示用户
            if not search_result or not hasattr(search_result, 'points') or not search_result.points:
                st.warning("没有找到相关内容。")
                st.stop()
            
            # 获取实际的点列表
            search_points = search_result.points
            
            # 准备重排序 (第二阶段：重排序)
            if reranker is not None:
                with st.spinner("正在重排序结果..."):
                    # 确保搜索结果是可迭代的对象
                    if not hasattr(search_points, '__iter__'):
                        st.error(f"搜索结果类型错误: {type(search_points)}")
                        st.stop()
                    
                    # 安全地提取文本内容，处理不同类型的结果
                    passages = []
                    file_matches = []  # 初始化文件名匹配列表
                    for point in search_points:
                        try:
                            # 从 ScoredPoint 对象中提取 payload 和文本
                            payload = point.payload
                            
                            # 提取文本内容用于重排序
                            if isinstance(payload, dict):
                                text = payload.get("text", "")
                            elif isinstance(payload, list):
                                # 如果payload是列表，尝试获取第一个元素并确保是字符串
                                text = str(payload[0]) if payload else ""
                            else:
                                # 其他类型，转换为字符串
                                text = str(payload)
                            
                            # 确保text是字符串类型
                            if not isinstance(text, str):
                                text = str(text)
                            
                            # 限制文本长度，防止模型处理过长文本
                            MAX_TEXT_LENGTH = 512  # 根据模型的最大输入长度调整
                            if len(text) > MAX_TEXT_LENGTH:
                                text = text[:MAX_TEXT_LENGTH]
                            
                            passages.append(text)
                            
                            # 检查是否是文件名向量点
                            is_filename_only = payload.get("is_filename_only", False)
                            
                            # 获取文件名
                            filename = payload.get("filename", "")
                            if not filename:
                                source_path = Path(payload.get("source", payload.get("file_path", "")))
                                filename = source_path.name
                                
                            filename_lower = filename.lower()
                            
                            # 文件名向量点优先级更高
                            if is_filename_only and all(term in filename_lower for term in query.lower().split()):
                                file_matches.insert(0, point)  # 插入到最前面
                            # 普通向量点但文件名匹配
                            elif all(term in filename_lower for term in query.lower().split()):
                                file_matches.append(point)
                        except Exception as e:
                            with st.sidebar:
                                st.warning(f"提取文本时出错: {str(e)}")
                            passages.append("")  # 添加空字符串作为占位符
                    
                    # 创建查询-文档对，用于重排序
                    query_passage_pairs = []
                    for passage in passages:
                        # 确保查询和文本都是字符串类型
                        query_str = str(query)
                        passage_str = str(passage)
                        query_passage_pairs.append([query_str, passage_str])
                    
                    try:
                        # 检查是否有足够的文本进行重排序
                        if not query_passage_pairs:
                            with st.sidebar:
                                st.warning("没有足够的文本进行重排序，将使用原始搜索结果")
                        else:
                            # 使用重排序模型计算相关性分数
                            rerank_scores = reranker.predict(query_passage_pairs)
                            
                            # 将重排序分数与检索结果合并
                            for i, point in enumerate(search_points):
                                if i < len(rerank_scores):  # 确保索引在有效范围内
                                    try:
                                        # 更新分数
                                        point.score = float(rerank_scores[i])
                                    except Exception as e:
                                        with st.sidebar:
                                            st.warning(f"更新分数时出错: {str(e)}")
                    
                    except Exception as e:
                        with st.sidebar:
                            st.warning(f"重排序过程中出错: {str(e)}")
                        # 如果重排序失败，继续使用原始搜索结果
            
            # 过滤掉低于阈值的结果
            results = []
            for point in search_points:
                try:
                    # 降低阈值，临时设置为0.01
                    if point.score > 0.01:  # 原来是 SCORE_THRESHOLD (0.45)
                        results.append(point)
                except Exception as e:
                    with st.sidebar:
                        st.warning(f"过滤结果时出错: {str(e)}")
            
            # 文件名精确匹配搜索（优先显示）
            file_matches = []
            query_terms = query.lower().split()
            
            # 根据文件路径和文件名进行匹配
            for point in results:
                try:
                    # 获取payload
                    payload = point.payload
                    
                    # 检查是否是文件名向量点
                    is_filename_only = payload.get("is_filename_only", False)
                    
                    # 获取文件名
                    filename = payload.get("filename", "")
                    if not filename:
                        source_path = Path(payload.get("source", payload.get("file_path", "")))
                        filename = source_path.name
                        
                    filename_lower = filename.lower()
                    
                    # 文件名向量点优先级更高
                    if is_filename_only and all(term in filename_lower for term in query_terms):
                        file_matches.insert(0, point)  # 插入到最前面
                    # 普通向量点但文件名匹配
                    elif all(term in filename_lower for term in query_terms):
                        file_matches.append(point)
                except Exception as e:
                    with st.sidebar:
                        st.warning(f"处理文件名匹配时出错: {str(e)}")
            
            # 合并结果
            combined_results = file_matches + [r for r in results if r not in file_matches]
            
            # 使用重排序后的结果
            results = combined_results
        except Exception as e:
            with st.sidebar:
                st.error(f"搜索过程中出错: {str(e)}")
            st.stop()
    
    if results:
        st.subheader("📄 匹配结果：")

        keywords = list(set(re.findall(r'\w+', query.lower())))

        # 为每个文件生成唯一ID
        # 移除这里的全局文件ID生成，我们将为每个文件单独生成ID
        # file_id = str(uuid.uuid4())
        # if file_id not in st.session_state.show_full_content:
        #     st.session_state.show_full_content[file_id] = False
        
        # 如果有正在查看完整内容的文件，显示固定在顶部的按钮
        if use_original_file and st.session_state.current_file_id and st.session_state.show_full_content.get(st.session_state.current_file_id, False):
            with st.sidebar:
                st.markdown("### 文档控制")
                
                # 定义收起按钮的回调函数
                def on_collapse_click():
                    file_id = st.session_state.current_file_id
                    st.session_state.show_full_content[file_id] = False
                    st.session_state.scroll_to_file = file_id
                    # 设置需要重新渲染的标志
                    st.session_state.needs_rerun = True
                
                # 使用带回调的按钮
                st.button("📎 收起完整内容", key=f"collapse_button_{int(time.time())}_{random.randint(10000, 99999)}", 
                         on_click=on_collapse_click, type="primary")
        
        for i, hit in enumerate(results):
            try:
                # 检查是否为元组类型（可能是旧版本的结果格式）
                if isinstance(hit, tuple):
                    # 如果是元组，假设第一个元素是分数，第二个元素是payload
                    score, payload = hit
                    raw_path = payload.get("source", payload.get("file_path", ""))
                    content = payload.get("text", "")
                else:
                    # 正常处理对象类型
                    raw_path = hit.payload["source"]
                    content = hit.payload["text"]
            except Exception as e:
                st.error(f"处理搜索结果时出错: {str(e)}")
                st.error(f"结果类型: {type(hit)}")
                st.error(f"结果内容: {str(hit)[:500]}")
                continue
            
            # 文档路径信息
            abs_path = Path(raw_path).resolve()
            
            # 添加文件路径和打开按钮（移到顶部）
            col1, col2 = st.columns([4, 1])
            with col1:
                if show_full_path:
                    st.markdown(f"**📎 文件路径：** {raw_path}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**📎 文件名：** {abs_path.name}", unsafe_allow_html=True)
            with col2:
                # 使用索引和文件路径组合作为唯一key
                button_key = f"link_{i}_{abs_path.name.replace('.', '_')}"
                if st.button("🔗 打开文件", key=button_key):
                    # 使用系统默认方式打开文件
                    success, error = open_file_with_app(str(abs_path))
                    if not success:
                        st.error(f"打开失败: {error}")
            
            # 显示相似度
            st.markdown(f"**🔢 相似度：** `{round(hit.score, 4)}`", unsafe_allow_html=True)
            
            # 尝试读取原始文件以获取更完整的内容
            if use_original_file:
                try:
                    if os.path.exists(raw_path):
                        with open(raw_path, 'r', encoding='utf-8') as f:
                            full_content = f.read()
                        
                        # 直接显示全部内容
                        content = full_content
                        content_has_keywords = True
                        full_file_content = full_content
                except Exception as e:
                    st.warning(f"读取原始文件时出错: {str(e)}，将使用向量数据库中的内容片段")
                    # 即使读取出错，也标记为有关键字，以便显示查看完整内容按钮
                    content_has_keywords = True
                    # 使用向量数据库中的内容作为完整内容
                    full_file_content = content
            else:
                content_has_keywords = False
                full_file_content = None
            
            # 检查并修复可能的Markdown截断问题
            if apply_markdown_fix:
                content = fix_truncated_markdown(content)
            
            # 确保内容是字符串类型
            if not isinstance(content, str):
                try:
                    content = str(content)
                except Exception:
                    content = "错误：无法显示内容，内容格式不正确"
            
            # 定位关键字并获取上下文
            keyword_info = locate_keywords_in_text(content, keywords, context_lines=10)
            
            # 为当前文件创建唯一ID - 使用一致的方法
            file_id = hashlib.md5(str(point.id).encode()).hexdigest()
            
            # 确保文件ID在会话状态中存在
            if file_id not in st.session_state.show_full_content:
                st.session_state.show_full_content[file_id] = False
            
            # 创建锚点，用于滚动定位
            st.markdown(f'<div id="file_{file_id}"></div>', unsafe_allow_html=True)
            
            # 如果需要滚动到此文件，添加JavaScript滚动代码
            if st.session_state.scroll_to_file == file_id:
                st.markdown(
                    f"""
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {{
                            const element = document.getElementById('file_{file_id}');
                            if (element) {{
                                element.scrollIntoView({{behavior: 'smooth'}});
                            }}
                        }});
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                # 重置滚动状态，防止重复滚动
                st.session_state.scroll_to_file = None
            
            # 使用Streamlit的expander组件显示内容
            with st.expander("📝 笔记内容", expanded=True):
                # 如果找到了关键字，先显示包含关键字的上下文
                if keyword_info["has_keywords"] and not st.session_state.show_full_content[file_id]:
                    st.markdown("**🔍 关键字匹配位置:**", unsafe_allow_html=True)
                    
                    # 使用自定义Markdown渲染函数或Streamlit的markdown组件
                    if MARKDOWN_IT_AVAILABLE:
                        # 如果可用，使用自定义的Markdown渲染函数
                        rendered_html = render_markdown_with_highlight(keyword_info["context_text"], keywords)
                        content_height = max(300, len(keyword_info["context_text"].split('\n')) * 20)
                        styled_html = f"""
                        <style>
                        .markdown-content {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                            line-height: 1.6;
                            padding: 10px;
                            overflow-y: auto;
                            max-height: 100%;
                            border-radius: 5px;
                        }}
                        .markdown-content pre {{
                            background-color: #f5f5f5;
                            padding: 10px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                        </style>
                        <div class="markdown-content">
                        {rendered_html}
                        </div>
                        """
                        st.components.v1.html(styled_html, height=content_height, scrolling=True)
                    else:
                        # 否则使用Streamlit的markdown组件
                        st.markdown(keyword_info["context_text"], unsafe_allow_html=True)
                    
                    # 只在"优先使用原始文件内容"模式下显示"查看完整内容"按钮
                    if use_original_file:
                        # 定义按钮点击回调函数
                        def on_view_full_click():
                            st.session_state.show_full_content[file_id] = True
                            st.session_state.current_file_id = file_id
                            # 设置需要重新渲染的标志
                            st.session_state.needs_rerun = True
                        
                        # 使用更复杂的唯一键，包含时间戳、随机数和索引
                        timestamp = int(time.time())
                        random_suffix = random.randint(100000, 999999)
                        unique_button_key = f"view_full_{file_id}_{i}_{timestamp}_{random_suffix}"
                        
                        # 使用普通按钮，但添加on_click回调
                        st.button("查看完整内容", key=unique_button_key, on_click=on_view_full_click)
                
                # 显示完整内容
                elif st.session_state.show_full_content.get(file_id, False):
                    # 记录当前文件ID
                    st.session_state.current_file_id = file_id  # 确保记录当前文件ID
                    st.markdown("**📄 完整内容:**", unsafe_allow_html=True)
                    
                    # 确定要显示的内容
                    display_content = full_file_content if full_file_content is not None else keyword_info["full_text"]
                    
                    # 使用自定义Markdown渲染函数或Streamlit的markdown组件
                    if MARKDOWN_IT_AVAILABLE:
                        rendered_html = render_markdown_with_highlight(display_content, keywords)
                        content_height = max(500, len(display_content.split('\n')) * 20)
                        styled_html = f"""
                        <style>
                        .markdown-content {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                            line-height: 1.6;
                            padding: 10px;
                            overflow-y: auto;
                            max-height: 100%;
                            border-radius: 5px;
                        }}
                        .markdown-content pre {{
                            background-color: #f5f5f5;
                            padding: 10px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                        </style>
                        <div class="markdown-content">
                        {rendered_html}
                        </div>
                        """
                        st.components.v1.html(styled_html, height=content_height, scrolling=True)
                    else:
                        st.markdown(display_content, unsafe_allow_html=True)
                    
                    # 移除这里的按钮代码，避免在每个搜索结果中都显示按钮
                    # if st.session_state.current_file_id and st.session_state.show_full_content.get(st.session_state.current_file_id, False):
                    #     fixed_button_html = f"""
                    #     <button 
                    #         onclick="window.location.href='?collapse={file_id}'" 
                    #         class="fixed-top-button"
                    #     >
                    #         📎 收起完整内容
                    #     </button>
                    #     """
                    #     st.markdown(fixed_button_html, unsafe_allow_html=True)
                    #     
                    #     # 检查URL参数，处理收起操作
                    #     query_params = st.experimental_get_query_params()
                    #     if "collapse" in query_params:
                    #         collapse_id = query_params["collapse"][0]
                    #         if collapse_id in st.session_state.show_full_content:
                    #             st.session_state.show_full_content[collapse_id] = False
                    #             st.session_state.scroll_to_file = collapse_id
                    #             # 清除URL参数
                    #             st.experimental_set_query_params()
                    #             st.rerun()
                
                # 如果原始文件内容被截断但没有找到关键字，提供查看完整内容的选项
                elif content_has_keywords and full_file_content is not None and not st.session_state.show_full_content.get(file_id, False):
                    # 使用自定义Markdown渲染函数或Streamlit的markdown组件
                    if MARKDOWN_IT_AVAILABLE:
                        rendered_html = render_markdown_with_highlight(content, keywords)
                        content_height = max(300, len(content.split('\n')) * 20)
                        styled_html = f"""
                        <style>
                        .markdown-content {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                            line-height: 1.6;
                            padding: 10px;
                            overflow-y: auto;
                            max-height: 100%;
                            border-radius: 5px;
                        }}
                        .markdown-content pre {{
                            background-color: #f5f5f5;
                            padding: 10px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                        </style>
                        <div class="markdown-content">
                        {rendered_html}
                        </div>
                        """
                        st.components.v1.html(styled_html, height=content_height, scrolling=True)
                    else:
                        st.markdown(content, unsafe_allow_html=True)
                    
                    # 只在"优先使用原始文件内容"模式下显示"查看完整内容"按钮
                    if use_original_file:
                        # 定义按钮点击回调函数
                        def on_view_full_click():
                            st.session_state.show_full_content[file_id] = True
                            st.session_state.current_file_id = file_id
                            # 设置需要重新渲染的标志
                            st.session_state.needs_rerun = True
                        
                        # 使用更复杂的唯一键，包含时间戳、随机数和索引
                        timestamp = int(time.time())
                        random_suffix = random.randint(100000, 999999)
                        unique_button_key = f"view_full_{file_id}_{i}_{timestamp}_{random_suffix}"
                        
                        # 使用普通按钮，但添加on_click回调
                        st.button("查看完整内容", key=unique_button_key, on_click=on_view_full_click)
                
                # 如果没有找到关键字，直接显示完整内容
                else:
                    # 使用自定义Markdown渲染函数或Streamlit的markdown组件
                    if MARKDOWN_IT_AVAILABLE:
                        rendered_html = render_markdown_with_highlight(content, keywords)
                        content_height = max(300, len(content.split('\n')) * 20)
                        styled_html = f"""
                        <style>
                        .markdown-content {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                            line-height: 1.6;
                            padding: 10px;
                            overflow-y: auto;
                            max-height: 100%;
                            border-radius: 5px;
                        }}
                        .markdown-content pre {{
                            background-color: #f5f5f5;
                            padding: 10px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                        </style>
                        <div class="markdown-content">
                        {rendered_html}
                        </div>
                        """
                        st.components.v1.html(styled_html, height=content_height, scrolling=True)
                    else:
                        st.markdown(content, unsafe_allow_html=True)
                    
                    # 如果需要高亮关键词，添加JavaScript
                    if highlight_keywords and keywords and not MARKDOWN_IT_AVAILABLE:
                        content_id = hashlib.md5(content.encode()).hexdigest()
                        
                        highlight_js = f"""
                        <script>
                            document.addEventListener('DOMContentLoaded', function() {{
                                const keywords = {str(keywords).lower()};
                                if (!keywords || keywords.length === 0) return;
                                
                                // 查找所有文本节点
                                function findTextNodes(node) {{
                                    const textNodes = [];
                                    if (node.nodeType === 3) {{ // 文本节点
                                        textNodes.push(node);
                                    }} else if (node.nodeType === 1 && !['CODE', 'PRE'].includes(node.tagName)) {{
                                        for (let i = 0; i < node.childNodes.length; i++) {{
                                            textNodes.push(...findTextNodes(node.childNodes[i]));
                                        }}
                                    }}
                                    return textNodes;
                                }}
                                
                                // 获取所有Markdown内容的容器
                                const containers = document.querySelectorAll('.stMarkdown');
                                containers.forEach(container => {{
                                    const textNodes = findTextNodes(container);
                                    
                                    // 高亮关键词
                                    textNodes.forEach(node => {{
                                        let text = node.nodeValue;
                                        let parent = node.parentNode;
                                        let highlightedText = text;
                                        let hasHighlight = false;
                                        
                                        keywords.forEach(keyword => {{
                                            if (keyword.length < 2) return;
                                            
                                            const regex = new RegExp('\\\\b' + keyword + '\\\\b', 'gi');
                                            highlightedText = highlightedText.replace(regex, match => {{
                                                hasHighlight = true;
                                                return `<span style="background-color: yellow; font-weight: bold;">${{match}}</span>`;
                                            }});
                                        }});
                                        
                                        if (hasHighlight) {{
                                            const span = document.createElement('span');
                                            span.innerHTML = highlightedText;
                                            parent.replaceChild(span, node);
                                        }}
                                    }});
                                }});
                            }});
                        </script>
                        """
                        
                        st.components.v1.html(highlight_js, height=0)
            
            # 添加分隔线
            st.markdown("---")
    
    else:
        st.warning("没有找到相关内容。")

# 检查是否需要重新渲染页面
if st.session_state.get('needs_rerun', False):
    st.session_state.needs_rerun = False
    st.rerun()