import os
import re
import uuid
import hashlib
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
MODEL_NAME = "BAAI/bge-small-zh"
VECTOR_DIM = 512  # 修改为模型的实际输出维度

# === 加载模型 & 启动 Qdrant ===
print("🔍 加载模型与数据库...")
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path="./qdrant_data")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)

# === Markdown 分段函数 ===
def split_markdown_chunks(markdown_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 50):
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
    
    # 第二步：处理过长的内容块
    final_chunks = []
    for header, content in blocks:
        if len(content) <= chunk_size:
            final_chunks.append((header, content))
        else:
            # 优先按段落分割
            paragraphs = re.split(r'\n\s*\n', content)
            
            if len(max(paragraphs, key=len)) > chunk_size:
                # 如果段落仍然太长，按句子分割
                buffer = ""
                # 同时处理中英文句号
                parts = re.split(r'(?<=[。！？.!?])\s*', content)
                
                for p in parts:
                    if not p.strip():
                        continue
                    
                    if len(buffer) + len(p) < chunk_size:
                        buffer += p
                    else:
                        if buffer:
                            final_chunks.append((header, buffer.strip()))
                        buffer = p
                
                if buffer:  # 添加最后一个缓冲区
                    final_chunks.append((header, buffer.strip()))
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