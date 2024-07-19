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


# In here, Chroma's documentation library is created using the default ChromaDocumentStore().
# With no additional configuration

################
# EXEC:
# root@autodl-container-9b614dbe9b-c060355c:~/LLM_RAG_Clanguage/chroma# python3 ChromaQueryTextRetriever_test.py
# /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz: 100%|███████████████████| 79.3M/79.3M [00:09<00:00, 9.14MiB/s]
# {'title': 'three'} 0.22100931406021118
# {'title': 'two'} 0.28266316652297974
# {'title': 'one'} 0.3107161521911621


# By default, it appears that the model ChromaQueryTextRetriever uses to compute
# the similarity between a document and a query is: `all-MiniLM-L6-v2`




