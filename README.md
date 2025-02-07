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


## Dependencies

```bash
pip3 install -r requirements.txt
```

## Download and Install

* 修改Ollama模型下载地址
```bash
export OLLAMA_MODELS=/root/autodl-tmp/models
echo 'export OLLAMA_MODELS=/root/autodl-tmp/models' >> ~/.bashrc
source ~/.bashrc
```

* 下载 ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

* 启动 ollama 服务
```bash
ollama serve
```

* 下载模型
```bash
ollama pull qwen2
ollama pull nomic-embed-text:latest
```

```bash
cd chat
```

## Quick Start

```bash
python3 full.py -m knowledge -g qwen2 -e nomic-embed-text
```

## Usage

```bash
# python3 full.py -h
usage: full.py [-h] [-e EMODEL] [-g GMODEL] [-k TOP_K] [-m MODE]

RAG Chat Bot for C Language Learning!

options:
  -h, --help            show this help message and exit
  -e EMODEL, --Emodel EMODEL
                        Embedding Model
  -g GMODEL, --Gmodel GMODEL
                        Generator Model
  -k TOP_K, --top_k TOP_K
                        Top-k for retrieval
  -m MODE, --mode MODE  `test` or `knowledge`. Default is `knowledge`
```






