import os
from pathlib import Path
import argparse
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

# import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils import embedding_functions  # /* new */


# 创建嵌入函数对象
# ollama_ef = embedding_functions.OllamaEmbeddingFunction(
#     url="http://localhost:11434/api/embeddings",
#     model_name="nomic-embed-text:latest"
# )
# /* test */
# print(ollama_ef("Variable declarations"))


def test_ollama():
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


    documents = [
        Document(content="This contains variable declarations", meta={"title": "one"}),
        Document(content="This contains another sort of variable declarations", meta={"title": "two"}),
        Document(content="This has nothing to do with variable declarations", meta={"title": "three"}),
        Document(content="A random doc", meta={"title": "four"}),
        Document(content="Another random doc", meta={"title": "five"}),
        Document(content="A third random doc", meta={"title": "six"}),
        Document(content="A fourth random doc", meta={"title": "seven"}),
    ]

    indexing = Pipeline()
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.run({"writer": {"documents": documents}})

    querying = Pipeline()
    querying.add_component("retriever", ChromaQueryTextRetriever(document_store=document_store))

    def run_query(top_k):
        results = querying.run({"retriever": {"query": "Variable declarations", "top_k": top_k}})
        for d in results["retriever"]["documents"]:
            print(d.meta, d.score)

    # /* Testing */
    for i in range(7):  # [1, 7]
        print(f"\nTop {i+1} results:")
        run_query(i+1)


if __name__ == "__main__":
    test_ollama()



