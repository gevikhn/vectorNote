# vectorNote Docker 使用指南

## 项目Docker化说明

vectorNote项目已经配置为可以使用Docker容器运行，主要包含以下文件：

- `Dockerfile`：定义了Docker镜像的构建过程
- `docker-compose.yml`：定义了Docker服务的配置和运行方式

## 目录打包说明

Docker配置中，以下目录会被直接打包到镜像中：

- `model`：模型文件目录
- `doc`：文档目录
- `qdrant_data`：向量数据库存储目录
- `note_index.json`：笔记索引文件

这意味着您的数据将成为镜像的一部分，便于分发和部署。

## 使用方法

### 构建并启动容器

```bash
# 在项目根目录下运行
docker-compose up -d
```

这将构建Docker镜像并在后台启动容器。

### 查看应用日志

```bash
docker-compose logs -f
```

### 停止容器

```bash
docker-compose down
```

### 重新构建镜像

如果您修改了代码、依赖或数据文件，需要重新构建镜像：

```bash
docker-compose build
```

然后重新启动：

```bash
docker-compose up -d
```

## 访问应用

应用启动后，可以通过浏览器访问：

```
http://localhost:8501
```

## 注意事项

1. 由于数据目录已经打包到镜像中，如果您需要更新数据，需要重新构建镜像
2. 如果您使用的是GPU加速，可能需要修改Dockerfile以支持CUDA
3. 首次启动时，构建镜像可能需要一些时间，请耐心等待
