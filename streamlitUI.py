import streamlit as st
import subprocess
import json

# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="Obsidian 搜索", layout="wide")

# 导入配置文件
try:
    from config import (
        ROOT_DIR, COLLECTION_NAME, MODEL_NAME, 
        FORCE_CPU, OFFLINE_MODE, LOCAL_MODEL_PATH, 
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

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import re
import urllib.parse
from pathlib import Path
import torch
import os
import streamlit.components.v1 as components
import hashlib

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
@st.cache_resource
def load_model_and_client():
    """加载模型和数据库客户端"""
    try:
        # 检查本地模型目录是否存在（如果在离线模式下）
        if OFFLINE_MODE:
            st.info(f"正在离线模式下加载模型...")
            if os.path.exists(LOCAL_MODEL_PATH):
                st.success(f"找到本地模型: {LOCAL_MODEL_PATH}")
                # 使用本地模型路径
                model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
            else:
                st.error(f"未找到本地模型: {LOCAL_MODEL_PATH}")
                st.error("请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录")
                return None, None
        else:
            # 正常模式下加载在线模型
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        # 尝试连接本地数据库
        if os.path.exists("./qdrant_data"):
            client = QdrantClient(path="./qdrant_data")
            # 检查集合是否存在
            if not client.collection_exists(COLLECTION_NAME):
                st.warning(f"⚠️ 集合 {COLLECTION_NAME} 不存在，请先运行 scan_and_embed_notes.py 创建索引")
                # 创建临时内存数据库
                client = QdrantClient(":memory:")
                st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 创建索引。")
        else:
            st.warning("⚠️ 未找到向量数据库文件，使用临时内存数据库")
            client = QdrantClient(":memory:")
            st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 创建索引。")
        
        return model, client
    except Exception as e:
        st.error(f"加载模型或数据库时出错: {str(e)}")
        st.info("尝试重新初始化数据库...")
        try:
            # 尝试重新创建数据库连接
            client = QdrantClient(":memory:")  # 临时使用内存数据库
            st.warning("⚠️ 使用临时内存数据库。请先运行 scan_and_embed_notes.py 重建索引。")
            
            # 在离线模式下再次尝试加载本地模型
            if OFFLINE_MODE and os.path.exists(LOCAL_MODEL_PATH):
                model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
            else:
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

# 侧边栏配置
st.sidebar.header("⚙️ 搜索配置")
top_k = st.sidebar.slider("返回结果数量", 1, 20, 5)
score_threshold = st.sidebar.slider("相似度阈值", 0.0, 1.0, 0.45, 0.01)
max_display_lines = st.sidebar.slider("每个结果最大显示行数", 10, 100, 50)
highlight_keywords = st.sidebar.checkbox("高亮关键词", value=True)
show_full_path = st.sidebar.checkbox("显示完整文件路径", value=True)

# 添加高级选项折叠区
with st.sidebar.expander("🔧 高级选项"):
    use_original_file = st.checkbox("优先使用原始文件内容", value=True, 
                                  help="如果选中，将尝试读取原始文件内容而不仅仅使用向量数据库中的片段")
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
    
    # 2. 添加查询前缀，提高检索质量（BGE模型特性）
    enhanced_query = f"查询：{query}"
    
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
    
    参数:
        text (str): 要搜索的文本
        keywords (list): 关键字列表
        context_lines (int): 关键字前后要显示的行数
        
    返回:
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
    
    # 组合最终文本
    final_text = f"{prefix}\n{context_text}\n{suffix}" if prefix or suffix else context_text
    
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
    
    # 将查询文本转换为向量
    query_vector = model.encode(enhanced_query).tolist()
    
    with st.spinner("正在搜索..."):
        # 获取更多结果，后面会重排序
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k * 3,
            score_threshold=score_threshold  # 降低相似度阈值，增加召回率
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

        for hit in results:
            raw_path = hit.payload["source"]
            content = hit.payload["text"]
            
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
                if st.button("🔗 打开文件", key=f"link_{raw_path}"):
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
                        
                        # 提取文件的主要内容
                        lines = full_content.split('\n')
                        
                        # 查找包含关键字的行
                        keyword_lines = []
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            if any(keyword.lower() in line_lower for keyword in keywords if len(keyword) >= 2):
                                keyword_lines.append(i)
                        
                        # 如果找到了包含关键字的行，优先显示这部分内容
                        if keyword_lines and len(keyword_lines) > 0:
                            # 获取第一个关键字出现的位置
                            first_keyword_line = keyword_lines[0]
                            
                            # 计算要显示的行范围
                            context_lines = 15  # 关键字前后显示的行数
                            start_line = max(0, first_keyword_line - context_lines)
                            end_line = min(len(lines), first_keyword_line + context_lines + 1)
                            
                            # 提取包含关键字的上下文
                            preview_lines = lines[start_line:end_line]
                            preview_text = '\n'.join(preview_lines)
                            
                            # 添加提示，表明内容被截断
                            if start_line > 0:
                                preview_text = "...(前面还有内容)\n" + preview_text
                            if end_line < len(lines):
                                preview_text += "\n...(后面还有内容)"
                            
                            # 标记此内容包含关键字
                            content = preview_text
                            content_has_keywords = True
                            # 保存完整内容以便后续显示
                            full_file_content = full_content
                        else:
                            # 如果没有找到关键字，按照原来的方式处理
                            # 去除空行
                            meaningful_lines = [line for line in lines if line.strip()]
                            
                            # 根据用户设置的最大显示行数提取内容
                            if len(meaningful_lines) > max_display_lines:
                                preview_lines = meaningful_lines[:max_display_lines]
                                preview_text = '\n'.join(preview_lines)
                                preview_text += "\n...(更多内容)"
                                content_has_keywords = True  # 内容被截断，提供查看完整内容的选项
                                full_file_content = full_content
                            else:
                                preview_text = full_content
                                content_has_keywords = False
                                full_file_content = None
                            
                            content = preview_text
                except Exception as e:
                    st.warning(f"读取原始文件时出错: {str(e)}，将使用向量数据库中的内容片段")
                    content_has_keywords = False
                    full_file_content = None
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
            
            # 使用Streamlit的expander组件显示内容
            with st.expander("📝 笔记内容", expanded=True):
                # 如果找到了关键字，先显示包含关键字的上下文
                if keyword_info["has_keywords"]:
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
                    
                    # 添加查看完整内容的选项
                    if st.button("查看完整内容", key=f"full_{raw_path}"):
                        st.markdown("**📄 完整内容:**", unsafe_allow_html=True)
                        if full_file_content is not None:
                            if MARKDOWN_IT_AVAILABLE:
                                rendered_html = render_markdown_with_highlight(full_file_content, keywords)
                                content_height = max(500, len(full_file_content.split('\n')) * 20)
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
                                st.markdown(full_file_content, unsafe_allow_html=True)
                        else:
                            if MARKDOWN_IT_AVAILABLE:
                                rendered_html = render_markdown_with_highlight(keyword_info["full_text"], keywords)
                                content_height = max(500, len(keyword_info["full_text"].split('\n')) * 20)
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
                                st.markdown(keyword_info["full_text"], unsafe_allow_html=True)
                elif content_has_keywords and full_file_content is not None:
                    # 如果原始文件内容被截断，提供查看完整内容的选项
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
                    
                    # 添加查看完整内容的选项
                    if st.button("查看完整内容", key=f"full_{raw_path}"):
                        st.markdown("**📄 完整内容:**", unsafe_allow_html=True)
                        if MARKDOWN_IT_AVAILABLE:
                            rendered_html = render_markdown_with_highlight(full_file_content, keywords)
                            content_height = max(500, len(full_file_content.split('\n')) * 20)
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
                            st.markdown(full_file_content, unsafe_allow_html=True)
                else:
                    # 如果没有找到关键字，直接显示完整内容
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