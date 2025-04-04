#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# 导入配置文件
try:
    from config import (
        ROOT_DIR, EXTENSIONS, CHUNK_SIZE, COLLECTION_NAME, 
        MODEL_NAME, VECTOR_DIM, FORCE_CPU, 
        INDEX_FILE, FORCE_REINDEX, MD5_FILE_SIZE_THRESHOLD,
        OFFLINE_MODE, LOCAL_MODEL_PATH, set_offline_mode
    )
except ImportError:
    print("错误: 未找到配置文件 config.py")
    sys.exit(1)

# 处理命令行参数
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="扫描并向量化Markdown笔记")
    parser.add_argument("--force", action="store_true", help="强制重新索引所有文件")
    parser.add_argument("--offline", action="store_true", help="启用离线模式，使用本地缓存模型")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU进行计算，即使有GPU可用")
    parser.add_argument("--show-help", action="store_true", help="显示帮助信息")
    args = parser.parse_args()
    
    if args.force:
        FORCE_REINDEX = True
        print("⚠️ 已启用强制重新索引模式")
    
    if args.offline:
        OFFLINE_MODE = True
        print("⚠️ 已启用离线模式")
        
    if args.cpu:
        FORCE_CPU = True
        print("⚠️ 已启用强制CPU模式")
        
    if args.show_help:
        print("使用方法:")
        print("  python scan_and_embed_notes.py            # 增量更新，只处理新增或修改的文件")
        print("  python scan_and_embed_notes.py --force    # 强制重新索引所有文件")
        print("  python scan_and_embed_notes.py --offline  # 启用离线模式，使用本地缓存模型")
        print("  python scan_and_embed_notes.py --cpu      # 强制使用CPU进行计算，即使有GPU可用")
        print("  python scan_and_embed_notes.py --show-help # 显示帮助信息")
        sys.exit(0)

# 设置离线模式环境变量（必须在导入模块前设置）
if OFFLINE_MODE:
    set_offline_mode(verbose=True)  # 在主脚本中保留日志输出

# 导入其他模块
import hashlib
import uuid
import json
import re
import time
from datetime import datetime
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

# === 检测CUDA可用性 ===
def check_cuda_availability():
    """检测是否有可用的CUDA设备，特别针对Windows环境优化"""
    # 如果强制使用CPU，直接返回
    if FORCE_CPU:
        print("⚠️ 已启用强制CPU模式，将使用CPU进行计算")
        return "cpu"
        
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
    
    # 检查本地模型目录是否存在（如果在离线模式下）
    if OFFLINE_MODE:
        print("正在离线模式下加载模型...")
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"找到本地模型: {LOCAL_MODEL_PATH}")
            # 使用本地模型路径
            model = SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE)
        else:
            print(f"错误: 未找到本地模型: {LOCAL_MODEL_PATH}")
            print("请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录")
            print(f"BGE-M3模型可以从 https://huggingface.co/BAAI/bge-m3 下载")
            sys.exit(1)
    else:
        # 正常模式下加载在线模型
        # 确保模型完全下载完成
        try:
            # 添加一个简单的测试，确保模型已完全加载
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            # 测试模型是否可用，使用一个简单的文本进行编码测试
            test_text = "测试模型是否完全加载"
            test_vector = model.encode(test_text)
            if len(test_vector) != VECTOR_DIM:
                raise ValueError(f"模型测试失败：向量维度不匹配，期望 {VECTOR_DIM}，实际 {len(test_vector)}")
        except Exception as e:
            print(f"❌ 模型加载或测试失败: {e}")
            print("请确保网络连接正常，或尝试使用离线模式")
            sys.exit(1)
    
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

# === 增量更新功能 ===
def load_index_file():
    """加载文件索引"""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_index_file(index):
    """保存文件索引"""
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def get_file_mtime(path):
    """获取文件最后修改时间"""
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        return datetime.now().isoformat()

def is_file_modified(path, index):
    """检查文件是否修改"""
    path_str = str(path)
    if path_str not in index:
        return True
    
    try:
        current_mtime = get_file_mtime(path)
        return current_mtime != index[path_str]["mtime"]
    except Exception:
        return True

def update_index_file(path, index):
    """更新文件索引"""
    path_str = str(path)
    index[path_str] = {
        "mtime": get_file_mtime(path),
        "last_indexed": datetime.now().isoformat()
    }

def remove_deleted_files(client, index):
    """删除已经不存在的文件的向量"""
    deleted_count = 0
    deleted_paths = []
    
    # 检查索引中的文件是否仍然存在
    for path_str in list(index.keys()):
        if not os.path.exists(path_str):
            # 文件已删除，从索引中移除
            deleted_paths.append(path_str)
            del index[path_str]
            deleted_count += 1
    
    # 从向量数据库中删除对应的向量
    if deleted_paths:
        print(f"🗑️ 检测到 {deleted_count} 个文件已删除，正在从向量数据库中移除...")
        
        # 分批处理删除操作，避免一次性删除过多
        batch_size = 100
        for i in range(0, len(deleted_paths), batch_size):
            batch = deleted_paths[i:i+batch_size]
            for path in batch:
                try:
                    # 创建过滤条件，匹配source字段
                    filter_condition = Filter(
                        must=[
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=path)
                            )
                        ]
                    )
                    
                    # 删除匹配的点
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=filter_condition
                    )
                except Exception as e:
                    print(f"删除向量时出错: {e}")
        
        print(f"✅ 已从向量数据库中移除 {deleted_count} 个已删除文件的向量")
    
    return deleted_count

def get_file_md5(path):
    """获取文件的MD5哈希值"""
    if os.path.getsize(path) > MD5_FILE_SIZE_THRESHOLD:
        return None
    
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()

def is_file_content_modified(path, index):
    """检查文件内容是否修改"""
    path_str = str(path)
    if path_str not in index:
        return True
    
    try:
        current_md5 = get_file_md5(path)
        return current_md5 != index[path_str]["md5"]
    except Exception:
        return True

def update_index_file_with_md5(path, index):
    """更新文件索引，包括MD5哈希值"""
    path_str = str(path)
    index[path_str] = {
        "mtime": get_file_mtime(path),
        "md5": get_file_md5(path),
        "last_indexed": datetime.now().isoformat()
    }

# === 遍历笔记并写入向量数据库 ===
all_points = []
file_count = 0
modified_count = 0
skipped_count = 0

print("📁 正在扫描并处理 Markdown 文件...")
index = load_index_file()
deleted_count = remove_deleted_files(client, index)  # 删除已经不存在的文件的向量

# 扫描所有文件
all_files = [path for path in ROOT_DIR.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS]
print(f"找到 {len(all_files)} 个 Markdown 文件")

# 设置批处理大小
BATCH_SIZE = 32  # 每次处理的文件数量
VECTOR_BATCH_SIZE = 128  # 每次上传到数据库的向量数量

# 分批处理文件
for batch_start in range(0, len(all_files), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(all_files))
    batch_files = all_files[batch_start:batch_end]
    
    print(f"处理批次 {batch_start//BATCH_SIZE + 1}/{(len(all_files) + BATCH_SIZE - 1)//BATCH_SIZE}，文件 {batch_start+1}-{batch_end} / {len(all_files)}")
    
    for path in tqdm(batch_files, desc="处理文件"):
        if not FORCE_REINDEX and not is_file_modified(str(path), index) and not is_file_content_modified(path, index):
            skipped_count += 1
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ 读取文件 {path} 失败: {e}")
            continue

        # 提取文件名（不含扩展名）作为额外的文本内容
        filename = path.stem
        
        # 处理文件名，将驼峰命名、下划线、连字符等分割为空格
        processed_filename = re.sub(r'([a-z])([A-Z])', r'\1 \2', filename)  # 驼峰转空格
        processed_filename = re.sub(r'[_\-.]', ' ', processed_filename)     # 下划线、连字符转空格
        
        # 将文件名添加到文本内容的开头，增加权重
        text_with_filename = f"# {processed_filename}\n\n{text}"
        
        # 从向量数据库中删除该文件的旧向量（如果存在）
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=str(path))
                    )
                ]
            )
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=filter_condition
            )
        except Exception as e:
            print(f"删除旧向量时出错: {e}")
        
        chunks = split_markdown_chunks(text_with_filename)
        if not chunks:
            continue

        sentences = [f"{title}\n{content}" if title else content for title, content in chunks]
        
        # 分批编码，避免一次性处理太多文本导致内存溢出
        ENCODE_BATCH_SIZE = 16  # 每批编码的文本数量
        vectors = []
        
        for i in range(0, len(sentences), ENCODE_BATCH_SIZE):
            batch_sentences = sentences[i:i+ENCODE_BATCH_SIZE]
            batch_vectors = model.encode(batch_sentences, show_progress_bar=False)
            vectors.extend(batch_vectors)
            
            # 手动清理内存
            if i % (ENCODE_BATCH_SIZE * 4) == 0 and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        namespace = uuid.NAMESPACE_DNS
        for i, vec in enumerate(vectors):
            uid = uuid.uuid5(namespace, f"{path}-{i}")
            all_points.append(PointStruct(
                id=str(uid),
                vector=vec.tolist(),
                payload={
                    "text": sentences[i],
                    "source": str(path),
                    "file_path": str(path),
                    "filename": path.name,
                    "chunk_id": f"{path.name}-{i}",
                    "created_at": datetime.now().isoformat()
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
                "file_path": str(path),
                "filename": path.name,
                "chunk_id": f"{path.name}-filename",
                "created_at": datetime.now().isoformat(),
                "is_filename_only": True
            }
        ))

        update_index_file_with_md5(path, index)
        modified_count += 1
        file_count += 1
        
        # 当累积的向量点达到一定数量时，上传到数据库并清空
        if len(all_points) >= VECTOR_BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=all_points)
            all_points = []
            # 定期保存索引文件，避免中断导致索引丢失
            save_index_file(index)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 每批次处理完成后，强制清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确保当前批次的点已上传
    if all_points:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        all_points = []
        save_index_file(index)

# 最后保存索引文件
save_index_file(index)

print(f"✅ 完成：共扫描 {len(all_files)} 个文件")
print(f"   - 新增/修改: {modified_count} 个文件")
print(f"   - 跳过未修改: {skipped_count} 个文件")
print(f"   - 删除: {deleted_count} 个文件")
print(f"向量数据已写入 Qdrant 数据库。")