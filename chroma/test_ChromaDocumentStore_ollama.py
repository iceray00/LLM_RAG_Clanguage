import os
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

import chromadb.utils.embedding_functions as embedding_functions  #/* new */
from chromadb.utils import embedding_functions


ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)
document_store = ChromaDocumentStore(embedding_function=ollama_ef)

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


print("\nTop 1 result:")
run_query(1)
print("\nTop 2 results:")
run_query(2)
print("Top 3 results:")
run_query(3)
print("Top 4 results:")
run_query(4)
print("\nTop 5 result:")
run_query(5)
print("\nTop 6 results:")
run_query(6)



