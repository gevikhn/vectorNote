import os
import sys
from pathlib import Path

# 导入配置文件
try:
    from config import (
        ROOT_DIR, COLLECTION_NAME, MODEL_NAME, RERANKER_MODEL_NAME,
        TOP_K, RERANK_TOP_K, SCORE_THRESHOLD, FORCE_CPU,
        OFFLINE_MODE, LOCAL_MODEL_PATH, LOCAL_RERANKER_PATH, set_offline_mode
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
from sentence_transformers import SentenceTransformer, CrossEncoder
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
        # 加载嵌入模型
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"找到本地嵌入模型: {LOCAL_MODEL_PATH}")
            # 使用本地模型路径
            model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
        else:
            console.print(f"[bold red]错误: 未找到本地嵌入模型: {LOCAL_MODEL_PATH}[/bold red]")
            console.print("[bold yellow]请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录[/bold yellow]")
            sys.exit(1)
            
        # 加载重排序模型
        if os.path.exists(LOCAL_RERANKER_PATH):
            print(f"找到本地重排序模型: {LOCAL_RERANKER_PATH}")
            # 使用本地重排序模型路径
            reranker = CrossEncoder(LOCAL_RERANKER_PATH, device=DEVICE)
        else:
            console.print(f"[bold yellow]警告: 未找到本地重排序模型: {LOCAL_RERANKER_PATH}[/bold yellow]")
            console.print("[bold yellow]将仅使用向量检索，不进行重排序[/bold yellow]")
            reranker = None
    else:
        # 正常模式下加载在线模型
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        try:
            reranker = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
            print("✓ 重排序模型加载完成")
        except Exception as e:
            console.print(f"[bold yellow]警告: 加载重排序模型失败: {e}[/bold yellow]")
            console.print("[bold yellow]将仅使用向量检索，不进行重排序[/bold yellow]")
            reranker = None
    
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
    
    # 2. 不再添加查询前缀，因为BGE-M3不需要
    enhanced_query = query
    
    return enhanced_query

def search_notes(query: str, model=None, client=None, reranker=None):
    """搜索笔记，使用混合检索+重排序的推荐管道"""
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
        if 'reranker' in globals():
            reranker = globals()['reranker']
    
    # 增强查询
    enhanced_query = enhance_query(query)
    
    # 将查询文本转换为向量
    query_vector = model.encode(enhanced_query)
    
    # 在 Qdrant 中搜索最相似的文档 (第一阶段：检索)
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,  # 检索更多结果用于重排序
        with_payload=True,
    )
    
    # 调试信息
    console.print(f"[bold cyan]调试: 搜索结果类型: {type(search_result)}[/bold cyan]")
    
    # 如果没有找到结果，直接返回空列表
    if not search_result or not hasattr(search_result, 'points') or not search_result.points:
        console.print("[bold yellow]未找到相关结果[/bold yellow]")
        return []
    
    # 获取实际的点列表
    search_points = search_result.points
    console.print(f"[bold cyan]调试: 找到 {len(search_points)} 个结果[/bold cyan]")
    
    # 准备重排序 (第二阶段：重排序)
    if reranker is not None:
        try:
            # 提取检索到的文档和查询，准备重排序
            passages = []
            for point in search_points:
                try:
                    # 从 ScoredPoint 对象中提取 payload 和文本
                    payload = point.payload
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
                except Exception as e:
                    console.print(f"[bold yellow]警告: 提取文本时出错: {str(e)}，类型: {type(point)}[/bold yellow]")
                    passages.append("")  # 添加空字符串作为占位符
            
            # 创建查询-文档对，用于重排序
            query_passage_pairs = []
            for passage in passages:
                # 确保查询和文本都是字符串类型
                query_str = str(query)
                passage_str = str(passage)
                query_passage_pairs.append([query_str, passage_str])
            
            try:
                # 尝试使用GPU进行重排序
                rerank_scores = reranker.predict(query_passage_pairs)
            except Exception as e:
                console.print(f"[bold yellow]GPU重排序失败，尝试使用CPU: {str(e)}[/bold yellow]")
                # 尝试将模型移至CPU
                try:
                    import torch
                    device = torch.device("cpu")
                    reranker.to(device)
                    rerank_scores = reranker.predict(query_passage_pairs)
                except Exception as e2:
                    console.print(f"[bold red]重排序失败，跳过重排序步骤: {str(e2)}[/bold red]")
                    # 如果重排序失败，直接使用原始搜索结果
                    return format_search_results(search_points)
            
            # 将重排序分数与检索结果合并
            for i, point in enumerate(search_points):
                try:
                    # 更新分数
                    point.score = float(rerank_scores[i])
                except Exception as e:
                    console.print(f"[bold yellow]警告: 更新分数时出错: {str(e)}，类型: {type(point)}[/bold yellow]")
            
            # 按重排序分数重新排序
            search_points = sorted(search_points, key=lambda x: x.score, reverse=True)
            
            # 只保留前RERANK_TOP_K个结果
            search_points = search_points[:RERANK_TOP_K]
        except Exception as e:
            console.print(f"[bold red]重排序过程中出错，使用原始搜索结果: {str(e)}[/bold red]")
            # 如果重排序过程中出错，直接使用原始搜索结果
            return format_search_results(search_points)
    
    return format_search_results(search_points)

def format_search_results(search_points):
    """格式化搜索结果，处理不同类型的结果"""
    # 过滤掉低于阈值的结果
    filtered_results = []
    for point in search_points:
        try:
            # 添加调试信息，显示分数
            console.print(f"[bold cyan]调试: 结果分数: {point.score}[/bold cyan]")
            
            # 降低阈值，临时设置为0.01
            if point.score > 0.01:  # 原来是 SCORE_THRESHOLD (0.45)
                filtered_results.append(point)
        except Exception as e:
            console.print(f"[bold yellow]警告: 过滤结果时出错: {str(e)}，类型: {type(point)}[/bold yellow]")
    
    search_points = filtered_results
    console.print(f"[bold cyan]调试: 过滤后剩余 {len(search_points)} 个结果[/bold cyan]")
    
    # 格式化结果
    formatted_results = []
    for point in search_points:
        try:
            # 获取payload和分数
            score = point.score
            payload = point.payload
            
            # 处理不同类型的payload
            if isinstance(payload, dict):
                # 提取文件路径和文本内容
                file_path = payload.get("file_path", payload.get("source", "未知路径"))
                text = payload.get("text", "")
                
                # 计算相对路径（如果是绝对路径）
                if os.path.isabs(file_path) and str(ROOT_DIR) in file_path:
                    rel_path = os.path.relpath(file_path, ROOT_DIR)
                else:
                    rel_path = file_path
                    
                # 提取其他元数据
                chunk_id = payload.get("chunk_id", "")
                created_at = payload.get("created_at", "")
            elif isinstance(payload, list):
                # 如果payload是列表，尝试使用第一个元素作为文本
                text = str(payload[0]) if payload else ""
                file_path = "未知路径"
                rel_path = "未知路径"
                chunk_id = ""
                created_at = ""
            else:
                # 其他类型，转换为字符串
                text = str(payload)
                file_path = "未知路径"
                rel_path = "未知路径"
                chunk_id = ""
                created_at = ""
            
            # 添加到结果列表
            formatted_results.append({
                "score": score,
                "file_path": file_path,
                "rel_path": rel_path,
                "text": text,
                "chunk_id": chunk_id,
                "created_at": created_at
            })
        except Exception as e:
            console.print(f"[bold yellow]警告: 格式化结果时出错: {str(e)}，类型: {type(point)}[/bold yellow]")
    
    return formatted_results

def main():
    """主函数"""
    # 使用全局变量
    global model, client, reranker
    
    # 获取命令行参数
    if len(sys.argv) < 2 and not args.query:
        console.print("[bold red]请提供搜索关键词[/bold red]")
        console.print("用法: python search_notes.py \"搜索关键词\"")
        sys.exit(1)
    
    # 获取查询文本
    query = args.query if args.query else " ".join(sys.argv[1:])
    
    # 使用全局已加载的模型和客户端进行搜索
    results = search_notes(query, model, client, reranker)
    
    # 显示结果
    if not results:
        console.print("[yellow]未找到相关结果[/yellow]")
        return
    
    for i, result in enumerate(results, 1):
        score = result["score"]
        file_path = result["file_path"]
        rel_path = result["rel_path"]
        text = result["text"]
        chunk_id = result["chunk_id"]
        created_at = result["created_at"]
        
        # 显示结果标题
        console.print(f"\n[bold green]结果 {i} (相似度: {score:.2f})[/bold green]")
        console.print(f"[bold cyan]文件: {rel_path}[/bold cyan]")
        console.print("\n")
        
        # 显示文本内容
        console.print(Markdown(text))
        
        console.print(f"\n[dim]来源: {file_path}[/dim]")
        console.print("─" * 80)

if __name__ == "__main__":
    main()