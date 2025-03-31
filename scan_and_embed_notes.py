import os
import re
import uuid
import hashlib
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# === 配置项 ===
ROOT_DIR = Path("D:/Notes")  # <-- ⚠️ 修改为你的 Obsidian 根目录路径
EXTENSIONS = [".md"]
CHUNK_SIZE = 1024
COLLECTION_NAME = "obsidian_notes"
# 升级到更强大的模型
MODEL_NAME = "BAAI/bge-large-zh-noinstruct"  # 或者 "text2vec-large-chinese"
VECTOR_DIM = 1024  # 修改为模型的实际输出维度

# === 检测CUDA可用性 ===
def check_cuda_availability():
    """检测是否有可用的CUDA设备，特别针对Windows环境优化"""
    try:
        # 尝试直接获取CUDA设备信息
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
            print(f"✅ 检测到 {device_count} 个CUDA设备: {device_name}")
            print(f"✅ 将使用GPU进行加速处理")
            return "cuda"
        
        # 如果上面的检测失败，尝试直接创建CUDA张量
        try:
            # 尝试在CUDA上创建一个小张量
            test_tensor = torch.tensor([1.0], device="cuda")
            del test_tensor  # 清理
            print(f"✅ 通过测试张量检测到CUDA设备")
            print(f"✅ 将使用GPU进行加速处理")
            return "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"⚠️ 检测到错误: {e}")
                print("⚠️ 你的PyTorch没有CUDA支持")
            pass
            
        # 在Windows上，尝试使用系统命令检测NVIDIA显卡
        nvidia_detected = False
        if os.name == 'nt':  # Windows系统
            try:
                # 使用nvidia-smi命令检测显卡
                result = os.system('nvidia-smi >nul 2>&1')
                if result == 0:
                    print(f"✅ 通过nvidia-smi检测到NVIDIA显卡")
                    nvidia_detected = True
                    
                    # 检查PyTorch是否支持CUDA
                    if not torch.cuda.is_available():
                        print("⚠️ 检测到NVIDIA显卡，但当前PyTorch版本不支持CUDA")
                        print("⚠️ 请注意: 你使用的是Python 3.13，目前PyTorch官方尚未为此版本提供CUDA支持")
                        print("⚠️ 建议方案:")
                        print("⚠️ 1. 降级到Python 3.10或3.11，然后安装支持CUDA的PyTorch")
                        print("⚠️    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                        print("⚠️ 2. 或者继续使用CPU模式（速度较慢）")
                        print("⚠️ 将使用CPU处理（速度较慢）")
                        return "cpu"
                    
                    # 强制设置CUDA可见
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    # 重新初始化CUDA
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        print(f"✅ 已启用CUDA设备: {device_name}")
                        print(f"✅ 将使用GPU进行加速处理")
                        return "cuda"
            except Exception:
                pass
                
        # 所有检测方法都失败，使用CPU
        if nvidia_detected:
            print("⚠️ 检测到NVIDIA显卡，但无法启用CUDA")
            print("⚠️ 请确保安装了正确的CUDA版本和支持CUDA的PyTorch")
            print("⚠️ 运行: pip uninstall torch torchvision torchaudio")
            print("⚠️ 然后: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("⚠️ 未检测到CUDA设备，将使用CPU处理（速度较慢）")
            print("⚠️ 如果你有NVIDIA显卡，请确保已安装正确的CUDA和PyTorch版本")
            print("⚠️ 提示: 可以尝试运行 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'")
        
        return "cpu"
    except Exception as e:
        print(f"⚠️ CUDA检测出错: {e}")
        print("⚠️ 将使用CPU处理（速度较慢）")
        return "cpu"

# 确定设备
DEVICE = check_cuda_availability()

# === 加载模型 & 启动 Qdrant ===
print("🔍 加载模型与数据库...")
try:
    # 先加载模型，确保模型完全下载
    print("正在加载模型，首次运行可能需要下载模型文件...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("✓ 模型加载完成")
    
    # 然后初始化数据库
    print("正在初始化向量数据库...")
    client = QdrantClient(path="./qdrant_data")
    
    # 检查集合是否存在，不存在则创建
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"✓ 创建集合 {COLLECTION_NAME}")
    else:
        # 如果需要重建集合，取消下面的注释
        # client.delete_collection(collection_name=COLLECTION_NAME)
        # client.create_collection(
        #     collection_name=COLLECTION_NAME,
        #     vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        # )
        print(f"✓ 使用现有集合 {COLLECTION_NAME}")
    
    print("✓ 数据库初始化完成")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

# === Markdown 分段函数 ===
def split_markdown_chunks(markdown_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 100):
    """
    智能分割 Markdown 文本为多个语义块
    
    特点:
    1. 保留文档结构和标题层级
    2. 智能处理代码块、列表等特殊格式
    3. 添加上下文重叠以提高搜索质量
    4. 更好地处理中英文混合文本
    5. 避免在 Markdown 语法关键位置截断
    6. 正确处理连续的代码块
    """
    blocks = []
    current_headers = []  # 存储当前标题层级
    current_content = []
    in_code_block = False
    code_fence_marker = ""  # 记录代码块的围栏标记（可能是```或~~~）
    in_list = False
    list_indent = 0  # 记录列表的缩进级别
    lines = markdown_text.splitlines()
    
    # 第一步：按照文档结构分块，特别注意代码块的完整性
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # 处理代码块 - 改进识别方式，支持```和~~~两种围栏
        if (stripped_line.startswith("```") or stripped_line.startswith("~~~")) and not in_code_block:
            # 进入代码块
            in_code_block = True
            code_fence_marker = stripped_line[:3]  # 记录使用的围栏类型
            current_content.append(line)
            continue
        elif in_code_block and stripped_line.startswith(code_fence_marker):
            # 退出代码块
            in_code_block = False
            code_fence_marker = ""
            current_content.append(line)
            continue
            
        # 在代码块内，不做特殊处理
        if in_code_block:
            current_content.append(line)
            continue
            
        # 处理标题，但确保不在代码块中间分割
        header_match = re.match(r'^(\s*)(#{1,6})\s+(.*?)$', line)
        if header_match and not in_code_block:
            # 保存之前的内容块
            if current_content:
                header_text = " > ".join(current_headers) if current_headers else ""
                blocks.append((header_text, "\n".join(current_content).strip()))
                current_content = []
            
            # 更新当前标题层级
            level = len(header_match.group(2))
            title = header_match.group(3).strip()
            
            # 根据标题级别调整标题层级
            current_headers = current_headers[:level-1]  # 移除更低级别的标题
            if level == 1:  # 如果是一级标题，清空所有标题
                current_headers = [title]
            else:
                current_headers.append(title)
                
            # 将标题也添加到当前内容中，增强语义连贯性
            current_content.append(line)
            continue
            
        # 处理列表项 - 改进列表识别，支持多级缩进
        list_match = re.match(r'^(\s*)[-*+]\s+', line) or re.match(r'^(\s*)\d+\.\s+', line)
        if list_match:
            if not in_list:
                in_list = True
                # 记录列表的缩进级别
                list_indent = len(list_match.group(1)) if list_match.group(1) else 0
            current_content.append(line)
            continue
        elif in_list:
            # 检查是否仍在列表中 - 空行或者缩进符合列表结构
            if not stripped_line:
                # 空行可能是列表项之间的分隔，保留在列表上下文中
                current_content.append(line)
                continue
            # 检查缩进是否符合列表结构
            indent_match = re.match(r'^(\s+)', line)
            current_indent = len(indent_match.group(1)) if indent_match else 0
            if current_indent >= list_indent:
                # 仍在列表上下文中
                current_content.append(line)
                continue
            else:
                # 列表结束
                in_list = False
        
        # 添加当前行到内容
        current_content.append(line)
    
    # 处理最后一个块
    if current_content:
        header_text = " > ".join(current_headers) if current_headers else ""
        blocks.append((header_text, "\n".join(current_content).strip()))
    
    # 第二步：预处理块，确保代码块的完整性
    processed_blocks = []
    for header, content in blocks:
        # 检查代码块是否完整 - 分别检查```和~~~两种围栏
        backtick_count = content.count("```")
        tilde_count = content.count("~~~")
        
        if backtick_count % 2 != 0:
            # 代码块不完整，添加结束标记
            content += "\n```"
        
        if tilde_count % 2 != 0:
            # 代码块不完整，添加结束标记
            content += "\n~~~"
        
        processed_blocks.append((header, content))
    
    # 第三步：处理过长的内容块，使用智能分割方法
    final_chunks = []
    for header, content in processed_blocks:
        if len(content) <= chunk_size:
            final_chunks.append((header, content))
        else:
            # 识别需要保护的 Markdown 元素
            protected_regions = []
            
            # 1. 识别所有代码块
            # 使用状态机而不是简单正则表达式，更可靠地处理嵌套和特殊情况
            i = 0
            while i < len(content):
                # 查找代码块开始
                if content[i:i+3] == "```" or content[i:i+3] == "~~~":
                    fence_type = content[i:i+3]
                    start = i
                    # 查找对应的结束标记
                    i += 3
                    while i < len(content):
                        if content[i:i+3] == fence_type and (i+3 >= len(content) or content[i+3] in ['\n', ' ', '\t']):
                            # 找到结束标记
                            end = i + 3
                            protected_regions.append((start, end))
                            i = end
                            break
                        i += 1
                    if i >= len(content):  # 没找到结束标记
                        protected_regions.append((start, len(content)))
                else:
                    i += 1
            
            # 2. 识别其他 Markdown 语法元素
            # 链接和图片
            for match in re.finditer(r'\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]\([^()]*(?:\([^()]*\)[^()]*)*\)', content):
                protected_regions.append((match.start(), match.end()))
            
            # 行内代码
            for match in re.finditer(r'`[^`]+`', content):
                protected_regions.append((match.start(), match.end()))
            
            # 强调和加粗
            for pattern in [r'\*\*[^*]+\*\*', r'\*[^*]+\*', r'__[^_]+__', r'_[^_]+_']:
                for match in re.finditer(pattern, content):
                    protected_regions.append((match.start(), match.end()))
            
            # 表格
            table_start = -1
            for i, line in enumerate(content.split('\n')):
                if re.match(r'\s*\|.*\|\s*$', line):
                    if table_start == -1:
                        table_start = i
                elif table_start != -1:
                    # 表格结束
                    table_end = i
                    # 计算表格在原文中的位置
                    start_pos = content.find('\n') * table_start if table_start > 0 else 0
                    end_pos = content.find('\n', start_pos) * table_end if table_end > 0 else len(content)
                    protected_regions.append((start_pos, end_pos))
                    table_start = -1
            
            # 合并重叠的保护区域
            if protected_regions:
                protected_regions.sort()
                merged_regions = [protected_regions[0]]
                for current in protected_regions[1:]:
                    prev = merged_regions[-1]
                    if current[0] <= prev[1]:
                        # 区域重叠，合并
                        merged_regions[-1] = (prev[0], max(prev[1], current[1]))
                    else:
                        merged_regions.append(current)
                protected_regions = merged_regions
            
            # 按语义单元分割内容
            chunks = []
            last_pos = 0
            
            # 首先按段落分割
            paragraphs = []
            lines = content.split('\n')
            current_para = []
            
            for line in lines:
                if not line.strip() and current_para:
                    # 空行标志段落结束
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
                else:
                    current_para.append(line)
            
            # 添加最后一个段落
            if current_para:
                paragraphs.append('\n'.join(current_para))
            
            # 根据段落和保护区域构建块
            current_chunk = ""
            for para in paragraphs:
                # 检查段落是否与任何保护区域重叠
                para_start = content.find(para, last_pos)
                para_end = para_start + len(para)
                last_pos = para_end
                
                # 检查段落是否需要保持完整
                para_protected = False
                for start, end in protected_regions:
                    if (start < para_end and end > para_start):
                        para_protected = True
                        break
                
                # 如果添加这个段落会导致块过大，并且当前块不为空，则开始新块
                if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk and not para_protected:
                    # 确保当前块的语法完整性
                    if not is_markdown_syntax_complete(current_chunk):
                        current_chunk = fix_markdown_syntax(current_chunk)
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    # 添加段落到当前块
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                
                # 特殊情况：如果当前段落本身就超过了块大小，需要进一步分割
                if len(current_chunk) > chunk_size and not para_protected:
                    # 尝试在句子边界分割
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                            # 确保当前块的语法完整性
                            if not is_markdown_syntax_complete(current_chunk):
                                current_chunk = fix_markdown_syntax(current_chunk)
                            chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
            
            # 添加最后一个块
            if current_chunk:
                # 确保语法完整性
                if not is_markdown_syntax_complete(current_chunk):
                    current_chunk = fix_markdown_syntax(current_chunk)
                chunks.append(current_chunk)
            
            # 如果没有成功分割，使用原始内容
            if not chunks:
                chunks = [content]
            
            # 将分割后的块添加到最终结果
            for chunk in chunks:
                final_chunks.append((header, chunk))
    
    # 第四步：添加上下文重叠（可选）
    if overlap > 0 and len(final_chunks) > 1:
        overlapped_chunks = []
        for i, (header, content) in enumerate(final_chunks):
            if i > 0:
                # 从前一个块的末尾添加重叠内容
                prev_content = final_chunks[i-1][1]
                # 智能选择重叠内容：尽量在句子边界处截断
                overlap_text = get_smart_overlap(prev_content, overlap, is_end=True)
                content = overlap_text + "..." + content
            
            if i < len(final_chunks) - 1:
                # 从后一个块的开头添加重叠内容
                next_content = final_chunks[i+1][1]
                # 智能选择重叠内容：尽量在句子边界处截断
                overlap_text = get_smart_overlap(next_content, overlap, is_end=False)
                content = content + "..." + overlap_text
                
            overlapped_chunks.append((header, content))
        return overlapped_chunks
    
    return final_chunks

def get_smart_overlap(text, max_length, is_end=False):
    """
    智能选择重叠内容，尽量在句子边界处截断
    
    参数:
    - text: 要处理的文本
    - max_length: 最大重叠长度
    - is_end: 是否从文本末尾选择
    
    返回:
    - 选择的重叠文本
    """
    if len(text) <= max_length:
        return text
    
    if is_end:
        # 从末尾选择
        text_portion = text[-max_length*2:]  # 选择更长的部分来寻找合适的边界
        # 尝试在句子边界处分割
        sentences = re.split(r'(?<=[.!?])\s+', text_portion)
        result = ""
        for sentence in reversed(sentences):
            if len(result) + len(sentence) + 1 <= max_length or not result:
                result = sentence + " " + result if result else sentence
            else:
                break
        return result.strip()
    else:
        # 从开头选择
        text_portion = text[:max_length*2]  # 选择更长的部分来寻找合适的边界
        # 尝试在句子边界处分割
        sentences = re.split(r'(?<=[.!?])\s+', text_portion)
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) + 1 <= max_length or not result:
                result = result + " " + sentence if result else sentence
            else:
                break
        return result.strip()

def is_markdown_syntax_complete(text):
    """检查 Markdown 语法是否完整"""
    # 检查代码块
    if text.count("```") % 2 != 0:
        return False
    
    # 检查链接和图片
    if text.count("[") != text.count("]") or text.count("(") != text.count(")"):
        return False
    
    # 检查强调标记
    if (text.count("**") % 2 != 0 or 
        text.count("*") % 2 != 0 or
        text.count("__") % 2 != 0 or
        text.count("_") % 2 != 0):
        return False
    
    return True

def fix_markdown_syntax(text):
    """修复 Markdown 语法中可能的截断问题"""
    # 修复代码块
    if text.count("```") % 2 != 0:
        text += "\n```"
    
    # 修复链接和图片
    if text.count("[") > text.count("]"):
        text += "]"
    if text.count("(") > text.count(")"):
        text += ")"
    
    # 修复强调标记
    if text.count("**") % 2 != 0:
        text += "**"
    if text.count("*") % 2 != 0:
        text += "*"
    if text.count("__") % 2 != 0:
        text += "__"
    if text.count("_") % 2 != 0:
        text += "_"
    
    return text

# === 遍历笔记并写入向量数据库 ===
all_points = []
file_count = 0

print("📁 正在扫描并处理 Markdown 文件...")
for path in tqdm(list(ROOT_DIR.rglob("*"))):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue

    # 提取文件名（不含扩展名）作为额外的文本内容
    filename = path.stem
    
    # 处理文件名，将驼峰命名、下划线、连字符等分割为空格
    # 例如: "Windows开启bbr" -> "Windows 开启 bbr"
    # 例如: "windows_bbr_config" -> "windows bbr config"
    processed_filename = re.sub(r'([a-z])([A-Z])', r'\1 \2', filename)  # 驼峰转空格
    processed_filename = re.sub(r'[_\-.]', ' ', processed_filename)     # 下划线、连字符转空格
    
    # 将文件名添加到文本内容的开头，增加权重
    text_with_filename = f"# {processed_filename}\n\n{text}"
    
    chunks = split_markdown_chunks(text_with_filename)
    if not chunks:
        continue

    sentences = [f"{title}\n{content}" if title else content for title, content in chunks]
    vectors = model.encode(sentences, show_progress_bar=False)

    for i, vec in enumerate(vectors):
        # 使用 uuid5 基于 SHA1 哈希生成确定性 UUID
        namespace = uuid.NAMESPACE_DNS
        uid = uuid.uuid5(namespace, f"{path}-{i}")
        all_points.append(PointStruct(
            id=str(uid),  # 将 UUID 转换为字符串
            vector=vec.tolist(),
            payload={
                "text": sentences[i],
                "source": str(path),
                "filename": path.name  # 添加文件名到payload
            }
        ))

    # 额外添加一个仅包含文件名的向量点，提高文件名搜索的准确性
    filename_vector = model.encode(processed_filename)
    filename_uid = uuid.uuid5(namespace, f"{path}-filename")
    all_points.append(PointStruct(
        id=str(filename_uid),
        vector=filename_vector.tolist(),
        payload={
            "text": f"# {processed_filename}",
            "source": str(path),
            "filename": path.name,
            "is_filename_only": True  # 标记这是一个仅包含文件名的向量点
        }
    ))

    file_count += 1
    if len(all_points) >= 128:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        all_points = []

if all_points:
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)

print(f"✅ 完成：共处理 {file_count} 个文件，向量数据已写入 Qdrant 数据库。")