import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

try:
    # 获取 data 目录下的所有文件路径
    # file_paths = ["data" / Path(name) for name in os.listdir("data")]
    file_paths = [Path("data") / name for name in os.listdir("data")]

except FileNotFoundError:
    print("Error: data directory not found.")
    exit(1)

# 创建一个 ChromaDocumentStore 实例
document_store = ChromaDocumentStore()

# 创建一个Pipeline实例用于文档索引
indexing = Pipeline()

# 添加文本文件转换器组件
indexing.add_component("converter", TextFileToDocument())

# 添加文档写入组件，将转换后的文档写入 document_store
indexing.add_component("writer", DocumentWriter(document_store))

# 连接组件
indexing.connect("converter", "writer")

# 运行Pipeline
indexing.run({"converter": {"sources": file_paths}})
