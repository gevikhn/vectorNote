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
def split_markdown_chunks(markdown_text: str, chunk_size: int = CHUNK_SIZE):
    blocks = []
    current_header = ""
    current_content = []
    lines = markdown_text.splitlines()

    for line in lines:
        if re.match(r'^\s*#{1,6}\s', line):
            if current_header or current_content:
                blocks.append((current_header, "\n".join(current_content).strip()))
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_header or current_content:
        blocks.append((current_header, "\n".join(current_content).strip()))

    final_chunks = []
    for header, content in blocks:
        if len(content) <= chunk_size:
            final_chunks.append((header, content))
        else:
            parts = re.split(r'(?<=[。！？])\s*', content)
            buffer = ""
            for p in parts:
                if not p.strip():
                    continue
                if len(buffer) + len(p) < chunk_size:
                    buffer += p
                else:
                    final_chunks.append((header, buffer.strip()))
                    buffer = p
            if buffer:
                final_chunks.append((header, buffer.strip()))

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

    chunks = split_markdown_chunks(text)
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
                "source": str(path)
            }
        ))

    file_count += 1
    if len(all_points) >= 128:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        all_points = []

if all_points:
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)

print(f"✅ 完成：共处理 {file_count} 个文件，向量数据已写入 Qdrant 数据库。")