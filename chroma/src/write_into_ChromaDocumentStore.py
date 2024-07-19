# /* Write documents into a ChromaDocumentStore */

import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from chroma_haystack import ChromaDocumentStore

file_paths = ["data" / Path(name) for name in os.listdir("data")]

document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))

indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})
