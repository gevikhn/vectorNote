$sourceDir = "C:\Users\17723\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
$targetDir = "D:\projects\python\vectorNote\model\bge-m3"
$poolingDir = Join-Path $targetDir "1_Pooling"

# 确保目标目录存在
if (-not (Test-Path -Path $targetDir)) {
    New-Item -Path $targetDir -ItemType Directory
}

# 确保1_Pooling目录存在
if (-not (Test-Path -Path $poolingDir)) {
    New-Item -Path $poolingDir -ItemType Directory
}

# 复制1_Pooling目录下的文件
$sourcePoolingDir = Join-Path $sourceDir "1_Pooling"
if (Test-Path -Path $sourcePoolingDir) {
    Get-ChildItem -Path $sourcePoolingDir | ForEach-Object {
        if ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            # 这是一个软链接
            $linkTarget = [System.IO.File]::ReadAllText($_.FullName)
            $actualFile = (Join-Path "C:\Users\17723\.cache\huggingface\hub\models--BAAI--bge-m3\blobs" $linkTarget.Substring($linkTarget.LastIndexOf("\") + 1))
            if (Test-Path -Path $actualFile) {
                Copy-Item -Path $actualFile -Destination (Join-Path $poolingDir $_.Name)
                Write-Host "已复制: $actualFile -> $(Join-Path $poolingDir $_.Name)"
            }
        } else {
            # 这是一个普通文件
            Copy-Item -Path $_.FullName -Destination (Join-Path $poolingDir $_.Name)
            Write-Host "已复制: $($_.FullName) -> $(Join-Path $poolingDir $_.Name)"
        }
    }
}

# 复制主目录下的软链接文件
Get-ChildItem -Path $sourceDir -File | ForEach-Object {
    if ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        # 获取软链接指向的目标
        $linkTarget = (Get-Item $_.FullName).Target
        if ($linkTarget) {
            # 提取blob文件名
            $blobFileName = $linkTarget -replace '.*\\blobs\\', ''
            $actualFile = Join-Path "C:\Users\17723\.cache\huggingface\hub\models--BAAI--bge-m3\blobs" $blobFileName
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
