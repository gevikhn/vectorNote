import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rich.console import Console
from rich.markdown import Markdown

# === 配置项 ===
COLLECTION_NAME = "obsidian_notes"
MODEL_NAME = "BAAI/bge-small-zh"
TOP_K = 5  # 返回最相关的结果数量
ROOT_DIR = Path("D:/Notes")  # 笔记根目录

# === 初始化组件 ===
console = Console()
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path="./qdrant_data")

def search_notes(query: str):
    # 将查询文本转换为向量
    query_vector = model.encode(query)
    
    # 在 Qdrant 中搜索最相似的文档
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K * 2,  # 获取更多结果，后面会过滤
        score_threshold=0.6  # 设置相似度阈值，过滤掉相关性较低的结果
    ).points
    
    # 显示搜索结果
    console.print(f"\n🔍 搜索：[bold blue]{query}[/bold blue]\n")
    
    # 文件名精确匹配搜索（优先显示）
    file_matches = []
    query_terms = query.lower().split()
    
    # 根据文件路径和文件名进行匹配
    for result in search_result:
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
    
    # 直接在文件系统中搜索匹配的文件（以防向量数据库中没有索引到）
    if not file_matches:  # 只有在向量搜索没有找到文件名匹配时才执行
        console.print("[dim]正在文件系统中搜索匹配文件...[/dim]")
        for root, dirs, files in os.walk(ROOT_DIR):
            for file in files:
                if file.lower().endswith('.md') and all(term in file.lower() for term in query_terms):
                    full_path = os.path.join(root, file)
                    try:
                        # 读取文件内容
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 直接显示文件系统搜索结果
                        console.print(f"\n[bold yellow]文件系统匹配[/bold yellow]")
                        console.print(f"[bold cyan]文件: {file}[/bold cyan]")
                        
                        # 提取文件的主要内容
                        lines = content.split('\n')
                        # 去除空行
                        lines = [line for line in lines if line.strip()]
                        
                        # 提取前10行非空内容作为预览
                        preview_lines = lines[:10]
                        preview = '\n'.join(preview_lines)
                        if len(lines) > 10:
                            preview += "\n..."
                        
                        # 显示预览内容
                        console.print("\n[bold]文件内容预览:[/bold]")
                        for line in preview_lines:
                            console.print(line)
                            
                        console.print(f"\n[dim]来源: {full_path}[/dim]\n")
                        console.print("─" * 80)
                        
                    except Exception as e:
                        console.print(f"[dim]读取文件 {full_path} 时出错: {e}[/dim]")
    
    # 合并结果，只使用向量搜索结果
    combined_results = file_matches + [r for r in search_result if r not in file_matches]
    
    # 去重并限制结果数量
    unique_results = []
    unique_paths = set()
    
    for result in combined_results:
        source = result.payload["source"]
        if source not in unique_paths and len(unique_results) < TOP_K:
            unique_paths.add(source)
            unique_results.append(result)
    
    # 显示结果
    if not unique_results:
        console.print("[yellow]未找到相关结果[/yellow]")
        return
    
    for i, result in enumerate(unique_results, 1):
        score = result.score
        text = result.payload["text"]
        source = result.payload["source"]
        
        # 提取文件名作为额外显示
        filename = Path(source).name
        
        console.print(f"\n[bold green]结果 {i} (相似度: {score:.2f})[/bold green]")
        console.print(f"[bold cyan]文件: {filename}[/bold cyan]")
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