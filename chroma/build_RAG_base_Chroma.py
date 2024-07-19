# /* Build RAG on top of Chroma */

from haystack import Pipeline
from chroma_haystack.retriever import ChromaQueryRetriever
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.builders import PromptBuilder

prompt = """
Answer the query based on the provided context.
If the context does not contain the answer, say 'Answer not found'.
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt)

llm = HuggingFaceTGIGenerator(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token='YOUR_HF_TOKEN')
llm.warm_up()
retriever = ChromaQueryRetriever(document_store=document_store)

querying = Pipeline()
querying.add_component("retriever", retriever)
querying.add_component("prompt_builder", prompt_builder)
querying.add_component("llm", llm)

querying.connect("retriever.documents", "prompt_builder.documents")
querying.connect("prompt_builder", "llm")

results = querying.run({"retriever": {"queries": [query], "top_k": 3},
                        "prompt_builder": {"query": query}})
