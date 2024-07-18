import os
import json
import requests
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
# from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

#* use in Haystack Intergration with Ollama *#
from haystack_integrations.components.generators.ollama import OllamaGenerator


model_name="Qwen1.5-32B-Q4"
model_url="http://localhost:11434/api/generate"


# # Set the environment variable OPENAI_API_KEY
# os.environ['OPENAI_API_KEY'] = "Your OpenAI API Key"


# Write documents to InMemoryDocumentStore
# 创建内存文档存储
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

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


# 创建检索器和自定义生成器
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
# llm = OpenAIGenerator()
# ollama_generator = OllamaGenerator(model_url="http://localhost:11434/api/generate")

generator = OllamaGenerator(model=model_name,
                            url=model_url,
                            generation_kwargs={
                              "num_predict": 80,
                              "temperature": 0.9,
                              "num_ctx": 8192
                              })


# 创建并配置管道
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
# rag_pipeline.add_component("llm", llm)
# rag_pipeline.add_component("llm", ollama_generator)
rag_pipeline.add_component("llm", generator)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question
question = "Who lives in Paris?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])
