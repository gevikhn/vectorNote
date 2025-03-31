import os
import sys
from pathlib import Path

# 导入配置文件
try:
    from config import (
        ROOT_DIR, COLLECTION_NAME, MODEL_NAME, 
        TOP_K, SCORE_THRESHOLD, FORCE_CPU,
        OFFLINE_MODE, LOCAL_MODEL_PATH, set_offline_mode
    )
except ImportError:
    print("错误: 未找到配置文件 config.py")
    sys.exit(1)

# 处理命令行参数
import argparse
parser = argparse.ArgumentParser(description="搜索向量化的笔记")
parser.add_argument("query", nargs="?", help="搜索关键词")
parser.add_argument("--offline", action="store_true", help="启用离线模式，使用本地缓存模型")
parser.add_argument("--cpu", action="store_true", help="强制使用CPU进行计算，即使有GPU可用")
args = parser.parse_args()

if args.offline:
    OFFLINE_MODE = True
    print("已启用离线模式")

if args.cpu:
    FORCE_CPU = True
    print("已启用强制CPU模式")

# 设置离线模式环境变量（必须在导入模块前设置）
if OFFLINE_MODE:
    set_offline_mode(verbose=True)  # 在主脚本中保留日志输出

# 导入其他模块
import torch
import hashlib
import uuid
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.markdown import Markdown
import re

console = Console()

# === 检测CUDA可用性 ===
def check_cuda_availability():
    """检测是否有可用的CUDA设备，特别针对Windows环境优化"""
    # 如果强制使用CPU，直接返回
    if FORCE_CPU:
        console.print("[bold yellow]⚠️ 已启用强制CPU模式，将使用CPU进行计算[/bold yellow]")
        return "cpu"
        
    try:
        # 尝试直接获取CUDA设备信息
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "未知"
            console.print(f"[bold green]✅ 检测到 {device_count} 个CUDA设备: {device_name}[/bold green]")
            console.print(f"[bold green]✅ 将使用GPU进行加速处理[/bold green]")
            return "cuda"
        
        # 如果上面的检测失败，尝试直接创建CUDA张量
        try:
            # 尝试在CUDA上创建一个小张量
            test_tensor = torch.tensor([1.0], device="cuda")
            del test_tensor  # 清理
            console.print(f"[bold green]✅ 通过测试张量检测到CUDA设备[/bold green]")
            console.print(f"[bold green]✅ 将使用GPU进行加速处理[/bold green]")
            return "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e):
                console.print(f"[bold yellow]⚠️ 检测到错误: {e}[/bold yellow]")
                console.print("[bold yellow]⚠️ 你的PyTorch没有CUDA支持[/bold yellow]")
            pass
            
        # 在Windows上，尝试使用系统命令检测NVIDIA显卡
        nvidia_detected = False
        if os.name == 'nt':  # Windows系统
            try:
                # 使用nvidia-smi命令检测显卡
                result = os.system('nvidia-smi >nul 2>&1')
                if result == 0:
                    console.print(f"[bold green]✅ 通过nvidia-smi检测到NVIDIA显卡[/bold green]")
                    nvidia_detected = True
                    
                    # 检查PyTorch是否支持CUDA
                    if not torch.cuda.is_available():
                        console.print("[bold yellow]⚠️ 检测到NVIDIA显卡，但当前PyTorch版本不支持CUDA[/bold yellow]")
                        console.print("[bold yellow]⚠️ 请注意: 你使用的是Python 3.13，目前PyTorch官方尚未为此版本提供CUDA支持[/bold yellow]")
                        console.print("[bold yellow]⚠️ 建议方案:[/bold yellow]")
                        console.print("[bold yellow]⚠️ 1. 降级到Python 3.10或3.11，然后安装支持CUDA的PyTorch[/bold yellow]")
                        console.print("[bold yellow]⚠️    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/bold yellow]")
                        console.print("[bold yellow]⚠️ 2. 或者继续使用CPU模式（速度较慢）[/bold yellow]")
                        console.print("[bold yellow]⚠️ 将使用CPU处理（速度较慢）[/bold yellow]")
                        return "cpu"
                    
                    # 强制设置CUDA可见
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    # 重新初始化CUDA
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        console.print(f"[bold green]✅ 已启用CUDA设备: {device_name}[/bold green]")
                        console.print(f"[bold green]✅ 将使用GPU进行加速处理[/bold green]")
                        return "cuda"
            except Exception:
                pass
                
        # 所有检测方法都失败，使用CPU
        if nvidia_detected:
            console.print("[bold yellow]⚠️ 检测到NVIDIA显卡，但无法启用CUDA[/bold yellow]")
            console.print("[bold yellow]⚠️ 请确保安装了正确的CUDA版本和支持CUDA的PyTorch[/bold yellow]")
            console.print("[bold yellow]⚠️ 运行: pip uninstall torch torchvision torchaudio[/bold yellow]")
            console.print("[bold yellow]⚠️ 然后: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/bold yellow]")
        else:
            console.print("[bold yellow]⚠️ 未检测到CUDA设备，将使用CPU处理（速度较慢）[/bold yellow]")
            console.print("[bold yellow]⚠️ 如果你有NVIDIA显卡，请确保已安装正确的CUDA和PyTorch版本[/bold yellow]")
            console.print("[bold yellow]⚠️ 提示: 可以尝试运行 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'[/bold yellow]")
        
        return "cpu"
    except Exception as e:
        console.print(f"[bold yellow]⚠️ CUDA检测出错: {e}[/bold yellow]")
        console.print("[bold yellow]⚠️ 将使用CPU处理（速度较慢）[/bold yellow]")
        return "cpu"

# === 初始化组件 ===
try:
    # 确定设备
    DEVICE = check_cuda_availability()
    
    print("正在加载模型，首次运行可能需要下载模型文件...")
    
    # 检查本地模型目录是否存在（如果在离线模式下）
    if OFFLINE_MODE:
        print("正在离线模式下加载模型...")
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"找到本地模型: {LOCAL_MODEL_PATH}")
            # 使用本地模型路径
            model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
        else:
            console.print(f"[bold red]错误: 未找到本地模型: {LOCAL_MODEL_PATH}[/bold red]")
            console.print("[bold yellow]请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录[/bold yellow]")
            sys.exit(1)
    else:
        # 正常模式下加载在线模型
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    print("✓ 模型加载完成")
    
    print("正在连接向量数据库...")
    client = QdrantClient(path="./qdrant_data")
    
    # 检查集合是否存在
    if not client.collection_exists(COLLECTION_NAME):
        console.print(f"[bold red]错误: 集合 {COLLECTION_NAME} 不存在，请先运行 scan_and_embed_notes.py[/bold red]")
        sys.exit(1)
    print("✓ 数据库连接成功")
except Exception as e:
    console.print(f"[bold red]初始化失败: {e}[/bold red]")
    sys.exit(1)

def enhance_query(query: str):
    """
    增强查询文本，提高检索效果
    """
    # 1. 去除多余空格和标点
    query = re.sub(r'\s+', ' ', query).strip()
    
    # 2. 添加查询前缀，提高检索质量（BGE模型特性）
    enhanced_query = f"查询：{query}"
    
    return enhanced_query

def search_notes(query: str, model=None, client=None):
    """搜索笔记"""
    # 如果没有传入模型和客户端，则使用全局变量
    if model is None or client is None:
        # 这里不再重新加载模型和客户端，而是使用全局已加载的
        console.print("[bold yellow]警告: 未传入模型或客户端，使用全局变量[/bold yellow]")
        # 确保全局变量已定义
        if 'model' not in globals() or 'client' not in globals():
            console.print("[bold red]错误: 全局模型或客户端未初始化[/bold red]")
            sys.exit(1)
        model = globals()['model']
        client = globals()['client']
    
    # 增强查询
    enhanced_query = enhance_query(query)
    
    # 将查询文本转换为向量
    query_vector = model.encode(enhanced_query)
    
    # 在 Qdrant 中搜索最相似的文档
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K * 3,  # 获取更多结果，后面会过滤和重排序
        score_threshold=SCORE_THRESHOLD  # 降低相似度阈值，增加召回率
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
    if not file_matches and len(search_result) < 2:  # 只有在向量搜索结果很少时才执行
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
    
    # 合并结果，只使用向量搜索结果
    combined_results = file_matches + [r for r in search_result if r not in file_matches]
    
    # 根据重排序分数排序
    combined_results.sort(key=rerank_score, reverse=True)
    
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
        
        # 显示结果标题
        console.print(f"\n[bold green]结果 {i} (相似度: {score:.2f})[/bold green]")
        console.print(f"[bold cyan]文件: {filename}[/bold cyan]")
        console.print("\n")
        
        # 尝试读取原始文件内容
        content_to_display = ""
        try:
            if os.path.exists(source):
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取文件的前30行非空内容
                lines = content.split('\n')
                non_empty_lines = []
                for line in lines:
                    if line.strip():
                        non_empty_lines.append(line)
                    if len(non_empty_lines) >= 30:
                        break
                
                if non_empty_lines:
                    content_to_display = '\n'.join(non_empty_lines)
                    if len(lines) > 30:
                        content_to_display += "\n..."
                else:
                    content_to_display = text
            else:
                content_to_display = text
        except Exception:
            # 如果读取失败，使用向量数据库中的文本
            content_to_display = text
        
        # 确保内容是字符串类型
        if not isinstance(content_to_display, str):
            try:
                content_to_display = str(content_to_display)
            except Exception:
                content_to_display = "错误：无法显示内容，内容格式不正确"
        
        console.print(Markdown(content_to_display))
        
        console.print(f"\n[dim]来源: {source}[/dim]")
        console.print("─" * 80)

def main():
    """主函数"""
    # 使用全局变量
    global model, client
    
    # 获取命令行参数
    if len(sys.argv) < 2 and not args.query:
        console.print("[bold red]请提供搜索关键词[/bold red]")
        console.print("用法: python search_notes.py \"搜索关键词\"")
        sys.exit(1)
    
    # 获取查询文本
    query = args.query if args.query else " ".join(sys.argv[1:])
    
    # 使用全局已加载的模型和客户端进行搜索
    search_notes(query, model, client)

if __name__ == "__main__":
    main()