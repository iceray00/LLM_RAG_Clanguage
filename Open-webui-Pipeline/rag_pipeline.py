import json
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class RAGPipeline:
    def __init__(self):
        # Load and parse the JSON file
        json_file_path = Path("data") / "Simulated_Test.json"
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.knowledge_data = json.load(f)

        # Initialize document store
        self.document_store = ChromaDocumentStore(
            embedding_function="OllamaEmbeddingFunction",
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest"
        )

        # Create documents and write them to the document store
        self.index_documents()

        # Initialize retriever
        self.retriever = ChromaQueryTextRetriever(document_store=self.document_store)

        # Initialize generator
        self.generator = OllamaGenerator(
            model="qwen2:latest",
            url="http://localhost:11434/",
            generation_kwargs={
                "temperature": 0.8,
                "num_ctx": 4096
            }
        )

        # Set up the pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("generator", self.generator)
        self.pipeline.connect("retriever", "generator")

    def index_documents(self):
        documents = []
        for item in self.knowledge_data:
            doc = Document(
                content=item["title"],
                meta={
                    "问题/内容": item["title"],
                    "答案/内容": item["content"],
                    "部分": item["part"]["section"],
                    "题型/类型": item["part"]["classification"],
                    "关键词": ", ".join(item["tags"]) if isinstance(item["tags"], list) else item["tags"]
                }
            )
            documents.append(doc)
        # Write documents to the document store
        self.document_store.write_documents(documents)










