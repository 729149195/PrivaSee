#!/usr/bin/env python3
"""
数据库安装测试脚本

运行此脚本验证所有数据库和相关库是否正确安装
"""

import sys
import traceback

def test_imports():
    """测试基本导入"""
    try:
        import faiss
        print("✓ FAISS imported successfully")
    except ImportError as e:
        print(f"✗ FAISS import failed: {e}")
        return False

    try:
        import pymongo
        print("✓ PyMongo imported successfully")
    except ImportError as e:
        print(f"✗ PyMongo import failed: {e}")
        return False

    try:
        import sentence_transformers
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False

    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False

    return True

def test_functionality():
    """测试基本功能"""
    try:
        # 测试FAISS
        import faiss
        import numpy as np

        # 创建一个简单的索引
        dim = 128
        index = faiss.IndexFlatL2(dim)
        print("✓ FAISS index created successfully")

        # 测试向量操作
        vectors = np.random.random((10, dim)).astype('float32')
        index.add(vectors)
        print("✓ FAISS vectors added successfully")

        # 测试搜索
        query = np.random.random((1, dim)).astype('float32')
        distances, indices = index.search(query, 3)
        print("✓ FAISS search completed successfully")

    except Exception as e:
        print(f"✗ FAISS functionality test failed: {e}")
        traceback.print_exc()
        return False

    try:
        # 测试sentence-transformers
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Sentence transformer model loaded successfully")

        # 测试编码
        sentences = ["This is a test sentence", "Another test sentence"]
        embeddings = model.encode(sentences)
        print(f"✓ Text encoded successfully, shape: {embeddings.shape}")

    except Exception as e:
        print(f"✗ Sentence transformers test failed: {e}")
        traceback.print_exc()
        return False

    try:
        # 测试MongoDB连接（如果服务运行）
        try:
            from pymongo import MongoClient
            client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            print("✓ MongoDB connection successful")

            # 测试基本操作
            db = client.test_database
            collection = db.test_collection
            result = collection.insert_one({"test": "document"})
            print("✓ MongoDB insert operation successful")

            # 清理测试数据
            collection.delete_one({"test": "document"})
            client.close()

        except Exception as mongo_error:
            print(f"⚠ MongoDB not running or connection failed: {mongo_error}")
            print("  (This is normal if MongoDB is not installed/running)")

    except Exception as e:
        print(f"✗ MongoDB test failed: {e}")
        traceback.print_exc()
        return False

    return True

def main():
    """主函数"""
    print("=== 数据库安装测试 ===")
    print()

    # 测试导入
    print("1. 测试基本导入...")
    import_success = test_imports()

    if not import_success:
        print("\n❌ 基本导入测试失败")
        sys.exit(1)

    print("\n2. 测试功能...")
    function_success = test_functionality()

    if not function_success:
        print("\n❌ 功能测试失败")
        sys.exit(1)

    print("\n🎉 所有测试通过！数据库安装成功。")
    print("\n你可以开始使用以下功能：")
    print("- FAISS向量数据库进行相似性搜索")
    print("- MongoDB JSON数据库进行文档存储")
    print("- sentence-transformers进行文本向量化")
    print("- scikit-learn进行机器学习任务")

    return 0

if __name__ == "__main__":
    sys.exit(main())