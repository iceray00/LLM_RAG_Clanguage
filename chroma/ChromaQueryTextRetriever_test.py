import os
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

# Chroma is used in-memory, so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

documents = [
    Document(content="This contains variable declarations", meta={"title": "one"}),
    Document(content="This contains another sort of variable declarations", meta={"title": "two"}),
    Document(content="This has nothing to do with variable declarations", meta={"title": "three"}),
    Document(content="A random doc", meta={"title": "four"}),
]

indexing = Pipeline()
indexing.add_component("writer", DocumentWriter(document_store))
indexing.run({"writer": {"documents": documents}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryTextRetriever(document_store))
results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

for d in results["retriever"]["documents"]:
    print(d.meta, d.score)