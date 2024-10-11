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
# print("\nTop 7 results:")
# run_query(7)
# print("\nTop 8 results:")
# run_query(8)
# print("\nTop 9 results:")
# run_query(9)
# print("\nTop 10 results:")
# run_query(10)

####################
# EXEC:
# root@autodl-container-9b614dbe9b-c060355c:~/LLM_RAG_Clanguage/chroma# python3 test_top_k.py
# Top 3 results:
# {'title': 'three'} 0.22100931406021118
# {'title': 'two'} 0.28266316652297974
# {'title': 'one'} 0.3107161521911621
#
# Top 2 results:
# {'title': 'three'} 0.22100931406021118
# {'title': 'two'} 0.28266316652297974
#
# Top 1 result:
# {'title': 'three'} 0.22100931406021118
#
#-----------------
#
# That'Open-webui-Pipeline not correct!
#
# So, I decide to fix it successfully.
#



# Sad, that'Open-webui-Pipeline the correct one..

