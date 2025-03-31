import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rich.console import Console
from rich.markdown import Markdown

# === 配置项 ===
COLLECTION_NAME = "obsidian_notes"
MODEL_NAME = "BAAI/bge-small-zh"
TOP_K = 5  # 返回最相关的结果数量

# === 初始化组件 ===
console = Console()
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path="./qdrant_data")

def search_notes(query: str):
    # 将查询文本转换为向量
    query_vector = model.encode(query)
    
    # 在 Qdrant 中搜索最相似的文档
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K
    )
    
    # 显示搜索结果
    console.print(f"\n🔍 搜索：[bold blue]{query}[/bold blue]\n")
    
    for i, result in enumerate(search_result, 1):
        score = result.score
        text = result.payload["text"]
        source = result.payload["source"]
        
        console.print(f"\n[bold green]结果 {i} (相似度: {score:.2f})[/bold green]")
        console.print(Markdown(text))
        console.print(f"[dim]来源: {source}[/dim]\n")
        console.print("─" * 80)

def main():
    if len(sys.argv) > 1:
        # 从命令行参数获取搜索查询
        query = " ".join(sys.argv[1:])
        search_notes(query)
    else:
        # 交互式模式
        console.print("[bold]✨ 笔记语义搜索[/bold]")
        console.print("输入 'q' 退出\n")
        
        while True:
            query = input("🔍 请输入搜索内容: ").strip()
            if query.lower() == 'q':
                break
            if query:
                search_notes(query)

if __name__ == "__main__":
    main()