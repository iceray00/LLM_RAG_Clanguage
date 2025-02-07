import json
import ollama
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# 实现OllamaLLM类
class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama2"):
        # 正确设置 model_name 和调用基类构造函数
        self.model_name = model_name
        super().__init__()

    def _generate(self, prompt: str, stop: list = None) -> str:
        # 使用Ollama的Python接口来调用本地模型
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response['text']

    @property
    def _llm_type(self) -> str:
        # 返回模型的类型名称
        return "ollama"

    @property
    def _identifying_params(self) -> dict:
        # 用于标识模型的参数
        return {"model_name": self.model_name}

    def _get_tokens(self, text: str) -> int:
        # 可选的：计算模型token数量
        # 如果Ollama不支持直接计算token数，可以留空或者实现
        return len(text.split())

# 实例化OllamaLLM
ollama_llm = OllamaLLM(model_name="llama2")  # 你可以根据需要选择合适的模型

# 加载JSON数据（假设你的JSON格式类似于下面的结构）
with open('your_documents.json', 'r') as file:
    documents_json = json.load(file)

# 假设JSON数据是一个包含多个文档的列表，每个文档有一个'text'字段
documents = [{"text": doc["text"]} for doc in documents_json]  # 根据JSON格式调整解析方式

# 使用CharacterTextSplitter分割文档，避免过长的文本
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 创建FAISS索引
embedding = OpenAIEmbeddings()  # 也可以使用其他嵌入模型
faiss_index = FAISS.from_documents(docs, embedding)

# 创建RetrievalQA链，结合Ollama本地模型
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,  # 使用本地Ollama模型
    chain_type="map_reduce",  # map_reduce更适合较长的文本处理
    retriever=faiss_index.as_retriever()
)

# 运行查询
query = "What is the main topic of the document?"
response = qa_chain.run(query)

print(response)
