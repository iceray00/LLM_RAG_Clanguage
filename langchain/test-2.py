import json
import faiss
import numpy as np
import requests

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer


# 加载JSON数据（假设你的JSON格式类似于下面的结构）
with open('data/Simulated_Test.json', 'r') as file:
    documents_json = json.load(file)


# 初始化 Ollama LLM
llm = OllamaLLM(model="qwen2", host="localhost", port=11434)


# 加载预训练的嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 为每个文档生成嵌入
embeddings = [model.encode(doc["title"]) for doc in documents_json]

valid_embeddings = [embedding for embedding in embeddings if embedding is not None]
if not valid_embeddings:
    raise ValueError("No valid embeddings were generated.")


# 转换为 NumPy 数组
embedding_matrix = np.array(valid_embeddings, dtype="float32")

# 确保矩阵的形状正确
if len(embedding_matrix.shape) != 2:
    raise ValueError(f"Embedding matrix has incorrect shape: {embedding_matrix.shape}")

# 修改后的生成嵌入函数，直接返回嵌入
def generate_embedding(text):
    return model.encode(text)

# 创建 FAISS 索引
embedding_dim = embedding_matrix.shape[1]  # 嵌入维度
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

# 假设我们有一个查询
query = "我想知道，关于C语言学习，对于for循环，有什么相关定义吗？"

# 生成查询的嵌入
query_embedding = generate_embedding(query)

# 使用 FAISS 检索最相似的文档
D, I = index.search(np.array([query_embedding]).astype("float32"), k=1)  # k 是返回的最近邻个数

# 找到最相关的文档
retrieved_doc = documents_json[I[0][0]]  # I[0][0] 是文档的索引
print(f"Retrieved document: {retrieved_doc['title']}")


# 将检索到的文档与查询一起传入模型，生成答案
context = retrieved_doc["title"] + "\n" + retrieved_doc["content"]
prompt = f"Given the context: {context} \nAnswer the following question: {query}"

# 获取模型生成的答案
response = llm(prompt)
print(response)



