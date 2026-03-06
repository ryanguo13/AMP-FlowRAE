# DBAASP 数据库下载指南

## 方法一：网页直接导出（推荐）

1. **访问搜索页面**
   - 打开 https://dbaasp.org/search

2. **执行空搜索（获取所有数据）**
   - 页面会自动加载所有数据（无需填写任何搜索条件）
   - 等待表格加载完成（显示所有肽序列）

3. **导出 FASTA**
   - 在搜索结果表格上方，找到 **"Export FASTA Data"** 按钮
   - 点击按钮，浏览器会自动下载 FASTA 文件

4. **保存文件**
   - 将下载的文件重命名为 `dbaasp.fasta`
   - 移动到 `data/raw/dbaasp.fasta`

## 方法二：使用 API（如果方法一不可用）

DBAASP 提供 REST API，但需要查看 API 文档了解具体端点。

访问：https://dbaasp.org/api?page=rest

## 注意事项

- 如果导出按钮不可用，可能需要先执行一次搜索
- 如果数据量很大，导出可能需要一些时间
- 确保下载的是 FASTA 格式（不是 CSV）

## 下载后操作

下载完成后，运行以下命令合并到过滤库：

```bash
cd /home/ryankwok/Documents/AMP-FlowRAE
cat data/raw/uniprot_sprot.fasta \
    data/raw/dramp_Antimicrobial_amps.fasta \
    data/raw/dramp_general_amps.fasta \
    data/raw/naturalAMPs_APD2024a.fasta \
    data/raw/dbaasp.fasta \
  > data/processed/filter/merged.fasta

# 重建 MMseqs 数据库
mmseqs/bin/mmseqs createdb data/processed/filter/merged.fasta data/processed/filter/filterdb
mmseqs/bin/mmseqs createindex data/processed/filter/filterdb data/processed/filter/tmp --threads 16
```

