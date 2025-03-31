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
CHUNK_SIZE = 300
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
    """
    blocks = []
    current_headers = []  # 存储当前标题层级
    current_content = []
    in_code_block = False
    in_list = False
    lines = markdown_text.splitlines()
    
    # 第一步：按照文档结构分块
    for line in lines:
        # 处理代码块
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue
            
        # 在代码块内，不做特殊处理
        if in_code_block:
            current_content.append(line)
            continue
            
        # 处理标题
        header_match = re.match(r'^(\s*)(#{1,6})\s+(.*?)$', line)
        if header_match:
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
            
        # 处理列表项
        list_match = re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line)
        if list_match:
            in_list = True
        elif in_list and not line.strip():
            in_list = False
            
        # 添加当前行到内容
        current_content.append(line)
    
    # 处理最后一个块
    if current_content:
        header_text = " > ".join(current_headers) if current_headers else ""
        blocks.append((header_text, "\n".join(current_content).strip()))
    
    # 第二步：处理过长的内容块，使用滑动窗口而不是固定大小
    final_chunks = []
    for header, content in blocks:
        if len(content) <= chunk_size:
            final_chunks.append((header, content))
        else:
            # 优先按段落分割
            paragraphs = re.split(r'\n\s*\n', content)
            
            if len(max(paragraphs, key=len)) > chunk_size:
                # 如果段落仍然太长，使用滑动窗口
                buffer = ""
                # 同时处理中英文句号
                parts = re.split(r'(?<=[。！？.!?])\s*', content)
                
                window = []
                window_size = 0
                
                for p in parts:
                    if not p.strip():
                        continue
                    
                    window.append(p)
                    window_size += len(p)
                    
                    if window_size >= chunk_size:
                        final_chunks.append((header, "".join(window).strip()))
                        # 滑动窗口，保留后半部分
                        overlap_size = 0
                        while window and overlap_size < overlap:
                            part = window.pop(0)
                            overlap_size += len(part)
                        window_size = sum(len(p) for p in window)
                
                if window:  # 添加最后一个窗口
                    final_chunks.append((header, "".join(window).strip()))
            else:
                # 按段落分割并添加重叠
                buffer = ""
                for p in paragraphs:
                    if len(buffer) + len(p) + 1 < chunk_size:  # +1 for newline
                        buffer += p + "\n\n"
                    else:
                        if buffer:
                            final_chunks.append((header, buffer.strip()))
                        buffer = p + "\n\n"
                
                if buffer:  # 添加最后一个缓冲区
                    final_chunks.append((header, buffer.strip()))
    
    # 第三步：添加上下文重叠（可选）
    if overlap > 0 and len(final_chunks) > 1:
        overlapped_chunks = []
        for i, (header, content) in enumerate(final_chunks):
            if i > 0:
                # 从前一个块的末尾添加重叠内容
                prev_content = final_chunks[i-1][1]
                overlap_text = prev_content[-overlap:] if len(prev_content) > overlap else prev_content
                content = overlap_text + "..." + content
            
            if i < len(final_chunks) - 1:
                # 从后一个块的开头添加重叠内容
                next_content = final_chunks[i+1][1]
                overlap_text = next_content[:overlap] if len(next_content) > overlap else next_content
                content = content + "..." + overlap_text
                
            overlapped_chunks.append((header, content))
        return overlapped_chunks
    
    return final_chunks

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