"""
base on `test_json.py`
"""
import argparse
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
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

# 获取JSON文件的路径
json_file_path = Path("data") / "Simulated_Test.json"  # 假设JSON文件名是 Simulated_Test.json


query = """

我想知道，关于C语言学习，对于for循环，有什么适合的练习题目吗？

"""



# Build a RAG pipeline
# 定义提示模板
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

# 读取并解析JSON文件
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    exit(1)


def test_RAG():
    parser = argparse.ArgumentParser(description="Test ChromaDocumentStore with OllamaEmbeddingFunction")

    parser.add_argument("--model", type=str, default="nomic-embed-text:latest", help="input single one path of solidity file")
    # parser.add_argument("--input_file", type=str, help="input one file path for Contract source code")
    # parser.add_argument("--o", "-output", type=str, default="./vfcs/", help="output path for VFCS")
    # parser.add_argument("--model_name", type=str, default="Qwen1.5-32B-Q4", help="model name")
    # parser.add_argument("--ctx", type=int, default=default_ctx, help="The maximum length of the context")
    # parser.add_argument("--prompt5", type=str, default=None, help="Prompt5")
    # parser.add_argument("--prompt6", type=str, default=None, help="Prompt6")

    args = parser.parse_args()

    if args.model == "default":
        document_store = ChromaDocumentStore()
        print("Now, Embedding Model is: all-MiniLM-L6-v2")
    else:
        # 使用自定义嵌入函数创建 ChromaDocumentStore
        document_store = ChromaDocumentStore(
            embedding_function="OllamaEmbeddingFunction",
            url="http://localhost:11434/api/embeddings",
            model_name=args.model
        )
        print(f"Now, Embedding Model is: {args.model}")

    # 生成Haystack的文档格式
    documents = []

    for item in knowledge_data:
        doc = Document(
            content=item["title"],  # 将JSON中的'title'字段作为文档内容
            # 可选择的给出展示
            meta={
                "id": item["id"],
                "问题/内容": item["title"],
                "答案/内容": item["content"],
                "部分": item["part"]["section"],
                "题型/类型": item["part"]["classification"],
                # "tags": item["tags"],  # 但是对于Chroma中，不能接受list类型，需要转化为str才能继续进行操作
                "关键词": ", ".join(item["tags"]) if isinstance(item["tags"], list) else item["tags"]  # 将列表转换为逗号分隔的字符串
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


    # 创建一个 Pipeline 实例用于文档索引
    indexing = Pipeline()
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.run({"writer": {"documents": documents}})



    # 创建一个 Pipeline 实例用于文档检索
    querying = Pipeline()
    querying.add_component("retriever", ChromaQueryTextRetriever(document_store=document_store))

    def run_query(top_k):
        results = querying.run({"retriever": {"query": query, "top_k": top_k}})
        for d in results["retriever"]["documents"]:
            # print(prompt_template.format(documents=d.content, question=query))
            print(d.meta, d.score)

    # /* Testing */
    for i in range(7):  # [1, 7]
        print(f"\nTop {i+1} results:")
        run_query(i+1)



if __name__ == "__main__":
    test_RAG()









