#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vectorNote 配置文件
包含所有脚本共用的配置项
"""

import os
from pathlib import Path

# === 基础配置 ===
ROOT_DIR = Path("D:/Notes")  # 笔记根目录路径
EXTENSIONS = [".md"]  # 支持的文件扩展名
COLLECTION_NAME = "obsidian_notes"  # 向量数据库集合名称

# === 模型配置 ===
MODEL_NAME = "BAAI/bge-m3"  # 向量化模型名称
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"  # 重排序模型名称
VECTOR_DIM = 1024  # 向量维度
FORCE_CPU = False  # 是否强制使用CPU，即使有GPU可用

# === 索引配置 ===
CHUNK_SIZE = 512  # 文本分块大小
INDEX_FILE = "./note_index.json"  # 文件索引路径
FORCE_REINDEX = False  # 是否强制重新索引所有文件
MD5_FILE_SIZE_THRESHOLD = 1024 * 1024 * 5  # 5MB，超过此大小的文件不计算MD5

# === 搜索配置 ===
TOP_K = 20  # 初始检索结果数量
RERANK_TOP_K = 8  # 重排序后保留的结果数量
SCORE_THRESHOLD = 0.45  # 相似度阈值
ENABLE_RERANKING = True  # 是否启用重排序功能

# === UI配置 ===
SHOW_OPEN_FILE_BUTTON = False  # 是否显示"打开文件"按钮

# === 离线模式配置 ===
OFFLINE_MODE = False  # 是否启用离线模式
LOCAL_MODEL_PATH = "./models/bge-m3"  # 本地模型路径
LOCAL_RERANKER_PATH = "./models/bge-reranker-large"  # 本地重排序模型路径

# === 设置离线模式环境变量 ===
def set_offline_mode(verbose=True):
    """设置离线模式环境变量"""
    # 检查是否已经设置过环境变量
    if os.environ.get("HF_HUB_OFFLINE_MODE_SET") == "1":
        return
        
    if OFFLINE_MODE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"  # 指定模型缓存目录
        # 设置标志，表示已经设置过环境变量
        os.environ["HF_HUB_OFFLINE_MODE_SET"] = "1"
        
        if verbose:
            print("已设置离线模式环境变量")
        
        # 检查本地模型目录是否存在
        global MODEL_NAME
        if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
            if verbose:
                print(f"使用本地模型: {LOCAL_MODEL_PATH}")
            MODEL_NAME = LOCAL_MODEL_PATH
        else:
            if verbose:
                print(f"警告: 未找到本地模型 {LOCAL_MODEL_PATH}")
                print("请先在联网状态下运行一次程序下载模型，或者手动下载模型到指定目录")
