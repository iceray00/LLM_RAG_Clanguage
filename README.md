# C Language RAG with LLM and HayStack



```bash
sh initisl_rag.sh
```


在DocumentStore方面，我们选择开源的[Chroma](https://docs.trychroma.com).

其中，使用Ollama提供的与Haystack和Chroma的接口，从Ollama Model Library中，选取了:

* `snowflake-arctic-embed`
* `nomic-embed-text`
* `mxbai-embed-large`

这3个Embedding模型，用于RAG过程中的自定义嵌入模型使用。 或者在后面可以综合使用这三个向量嵌入模型，在我们的知识库中达到更好的效果。





prompt


fine-tuning-SFT

Agent


RAG
检索增加生成





