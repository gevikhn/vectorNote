$sourceDir = "C:\Users\17723\.cache\huggingface\hub\models--BAAI--bge-reranker-large\snapshots\55611d7bca2a7133960a6d3b71e083071bbfc312"
$targetDir = "D:\projects\python\vectorNote\model\bge-reranker-large"

# 确保目标目录存在
if (-not (Test-Path -Path $targetDir)) {
    New-Item -Path $targetDir -ItemType Directory
}

# 复制主目录下的软链接文件
Get-ChildItem -Path $sourceDir -File | ForEach-Object {
    if ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        # 获取软链接指向的目标
        $linkTarget = (Get-Item $_.FullName).Target
        if ($linkTarget) {
            # 提取blob文件名
            $blobFileName = $linkTarget -replace '.*\\blobs\\', ''
            $actualFile = Join-Path "C:\Users\17723\.cache\huggingface\hub\models--BAAI--bge-reranker-large\blobs" $blobFileName
            if (Test-Path -Path $actualFile) {
                Copy-Item -Path $actualFile -Destination (Join-Path $targetDir $_.Name)
                Write-Host "已复制: $actualFile -> $(Join-Path $targetDir $_.Name)"
            } else {
                Write-Host "警告: 找不到文件 $actualFile"
            }
        }
    } else {
        # 这是一个普通文件
        Copy-Item -Path $_.FullName -Destination (Join-Path $targetDir $_.Name)
        Write-Host "已复制: $($_.FullName) -> $(Join-Path $targetDir $_.Name)"
    }
}

Write-Host "复制完成！"
