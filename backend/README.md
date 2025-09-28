# 数据库集成指南

本项目集成了两种高性能数据库：FAISS向量数据库和MongoDB JSON数据库。

## 安装的数据库和工具

### 向量数据库
- **FAISS (Facebook AI Similarity Search)**: 用于高效的向量相似性搜索
- **sentence-transformers**: 用于将文本转换为向量

### JSON数据库
- **MongoDB**: 灵活的文档数据库
- **PyMongo**: MongoDB的Python驱动

### 机器学习工具
- **scikit-learn**: 机器学习工具库
- **PyTorch**: 深度学习框架（已存在）

## 快速开始

1. **启动MongoDB服务**（如果需要本地MongoDB）:
   ```bash
   # Ubuntu/Debian
   sudo systemctl start mongodb
   # 或
   sudo service mongodb start

   # macOS (使用Homebrew)
   brew services start mongodb-community

   # Docker方式
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   ```

2. **运行示例代码**:
   ```python
   cd backend
   python database_example.py
   ```

## 使用方法

### 基本用法

```python
from database_example import DatabaseManager

# 初始化数据库管理器
db_manager = DatabaseManager()

# 添加文档
db_manager.add_document_with_vector(
    content="Your document content here",
    metadata={"category": "example", "author": "user"}
)

# 搜索相似文档
results = db_manager.search_similar_documents("search query", k=5)

# 获取所有文档
all_docs = db_manager.get_all_documents()
```

### 高级用法

#### 向量数据库操作
```python
from vector_database import VectorDatabase

# 创建向量数据库
vector_db = VectorDatabase(dimension=384)

# 添加文档
documents = [
    {"content": "Document 1", "metadata": {"id": 1}},
    {"content": "Document 2", "metadata": {"id": 2}}
]
vector_db.add_documents(documents)

# 搜索
results = vector_db.search("query text", k=3)
```

#### JSON数据库操作
```python
from json_database import JSONDatabase

# 连接数据库
json_db = JSONDatabase("mongodb://localhost:27017/", "my_database")

# 插入文档
doc_id = json_db.insert_document({
    "content": "Sample document",
    "metadata": {"category": "sample"}
})

# 查询文档
docs = json_db.find_documents({"metadata.category": "sample"})

# 更新文档
json_db.update_document(doc_id, {"content": "Updated content"})

# 删除文档
json_db.delete_document(doc_id)
```

## 配置说明

### 环境要求
- Python 3.8+
- CUDA (可选，用于GPU加速的向量计算)
- MongoDB (可选，如果只需要向量数据库)

### 依赖包
```
faiss-cpu==1.12.0
pymongo==4.15.1
sentence-transformers==5.1.1
scikit-learn==1.7.2
numpy==1.26.3
torch==2.4.1+cu118
```

## 注意事项

1. **FAISS索引持久化**: 向量数据库的索引默认存储在内存中，重启程序后需要重新构建
2. **MongoDB连接**: 默认连接本地MongoDB，可以修改连接字符串连接远程数据库
3. **向量维度**: 当前使用384维向量，可以根据需要调整模型和维度
4. **性能优化**: 大规模数据时考虑使用GPU版本的FAISS和合适的索引类型

## 故障排除

1. **ImportError**: 确保所有依赖都已正确安装
2. **ConnectionError**: 检查MongoDB服务是否运行
3. **MemoryError**: 对于大数据集，考虑使用索引压缩或分页处理