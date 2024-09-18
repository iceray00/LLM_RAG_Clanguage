import os
import json
from tqdm import tqdm
from pathlib import Path
from haystack import Pipeline
# from haystack.schema import Document
# from haystack import Document
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# 获取JSON文件的路径
json_file_path = Path("data") / "Simulated_Test.json"  # 假设JSON文件名是knowledge_base.json

# 读取并解析JSON文件
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    exit(1)

# 创建一个 ChromaDocumentStore 实例
document_store = ChromaDocumentStore()

# 创建一个 Pipeline 实例用于文档索引
indexing = Pipeline()

# 生成Haystack的文档格式
documents = []

for item in knowledge_data:
    doc = Document(
        content=item["title"],  # 将JSON中的'title'字段作为文档内容
        meta={
            "id": item["id"],       # 你可以添加更多的元数据字段
            "content": item["content"],
            "section": item["part"]["section"],
            "classification": item["part"]["classification"],
            # "tags": item["tags"],  # 但是对于Chroma中，不能接受list类型，需要转化为str才能继续进行操作
            "tags": ", ".join(item["tags"]) if isinstance(item["tags"], list) else item["tags"]  # 将列表转换为逗号分隔的字符串
        }
    )
    documents.append(doc)

# # 不显示进度条
# document_store.write_documents(documents)  # 将这些文档写入到 ChromaDocumentStore

# 使用 tqdm 显示进度条
for doc in tqdm(documents, desc="Indexing documents", unit="document"):
    document_store.write_documents([doc])  # 每次写入一个文档，保持进度条更新

# 输出提示
print(f"Successfully indexed {len(documents)} documents to ChromaDocumentStore.")

indexing.add_component("writer", DocumentWriter(document_store))



#
# # 创建一个 ChromaDocumentStore 实例
# document_store = ChromaDocumentStore()
#
# # 创建一个Pipeline实例用于文档索引
# indexing = Pipeline()
#
# # 添加文本文件转换器组件
# indexing.add_component("converter", TextFileToDocument())
#
# # 添加文档写入组件，将转换后的文档写入 document_store
# indexing.add_component("writer", DocumentWriter(document_store))
#
# # 连接组件
# indexing.connect("converter", "writer")
#
# # 运行Pipeline
# indexing.run({"converter": {"sources": file_paths}})
