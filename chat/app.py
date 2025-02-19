# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import time
# import json
# from pathlib import Path
# from haystack.dataclasses import ChatMessage, Document
# from haystack.components.builders import ChatPromptBuilder
# from haystack_integrations.components.generators.ollama import OllamaChatGenerator
# from haystack_integrations.document_stores.chroma import ChromaDocumentStore
# from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
# from haystack import Pipeline

# app = Flask(__name__)
# CORS(app)

# # 初始化文档存储、检索器、生成器等组件，只需初始化一次
# document_store = ChromaDocumentStore(
#     embedding_function="OllamaEmbeddingFunction",
#     url="http://localhost:11434/api/embeddings",
#     model_name="nomic-embed-text:latest"
# )
# retriever = ChromaQueryTextRetriever(document_store=document_store)

# prompt_builder = ChatPromptBuilder()
# generator = OllamaChatGenerator(
#     model="qwen2:latest",
#     url="http://localhost:11434",
#     generation_kwargs={"temperature": 0.9}
# )

# conversation_history = [
#     ChatMessage.from_system("你是一个精通C语言的智能助教。用户会提问一些关于C语言编程和技术方面的问题，你要提供有帮助的答案。同时，这是一个RAG的场景，会给出对应的RAG的内容。")
# ]

# pipe = Pipeline()
# pipe.add_component("prompt_builder", prompt_builder)
# pipe.add_component("llm", generator)
# pipe.connect("prompt_builder.prompt", "llm.messages")


# def chat_with_RAG(question, conversation_history, retriever, top_k=5):
#     # 检索相关文档
#     retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

#     # 将检索到的文档添加到历史对话上下文
#     conversation_history.append(ChatMessage.from_system("相关文档如下："))
#     for doc in retrieved_documents:
#         conversation_history.append(ChatMessage.from_system(f"{doc.meta['问题/内容']}: {doc.content}"))

#     # 生成回复
#     querying = pipe.run(data={
#         "prompt_builder": {
#             "template_variables": {"question": question},
#             "template": conversation_history + [ChatMessage.from_user(question)]
#         }
#     })

#     response_text = querying["llm"]["replies"][0].text
#     conversation_history.append(ChatMessage.from_assistant(response_text))
#     return response_text


# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_question = data.get('message', '')

#     if not user_question:
#         return jsonify({'reply': '请输入问题！'}), 400

#     try:
#         answer = chat_with_RAG(user_question, conversation_history, retriever)
#         return jsonify({'reply': answer})
#     except Exception as e:
#         return jsonify({'reply': f'出错了：{str(e)}'}), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import json
import os
from pathlib import Path

from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

# 获取当前时间戳
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
output_folder = Path("chat_records") / timestamp
output_folder.mkdir(parents=True, exist_ok=True)

chat_records = []

# 初始化你的Pipeline和相关组件
Emodel = "nomic-embed-text:latest"
Gmodel = "qwen2:latest"
top_k = 5
mode = "knowledge"

if mode == "knowledge":
    with open(Path("data") / "Summary_of_Knowledge_Points.json", "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)

if Emodel == "default":
    document_store = ChromaDocumentStore()
else:
    document_store = ChromaDocumentStore(
        embedding_function="OllamaEmbeddingFunction",
        url="http://localhost:11434/api/embeddings",
        model_name=Emodel
    )

retriever = ChromaQueryTextRetriever(document_store=document_store)

documents = []
for item in knowledge_data:
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

for doc in tqdm(documents, desc="Indexing documents", unit="document"):
    document_store.write_documents([doc])

prompt_builder = ChatPromptBuilder()
generator = OllamaChatGenerator(
    model="qwen2:latest",
    url="http://localhost:11434/api/embeddings",
    generation_kwargs={
        "temperature": 0.9,
    }
)

conversation_history = [
    ChatMessage.from_system("你是一个精通C语言的智能助教。用户会提问一些关于C语言编程和技术方面的问题，你要提供有帮助的答案。同时，这是一个RAG的场景，会给出对应的RAG的内容。")
]

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)
pipe.connect("prompt_builder.prompt", "llm.messages")

def chat_with_RAG(question):
    global conversation_history
    retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

    conversation_history.append(ChatMessage.from_system("相关文档如下："))
    for doc in retrieved_documents:
        conversation_history.append(ChatMessage.from_system(f"{doc.meta['问题/内容']}: {doc.content}"))

    querying = pipe.run(data={
        "prompt_builder": {
            "template_variables": {"question": question},
            "template": conversation_history + [ChatMessage.from_user(question)]
        }
    })

    response = querying["llm"]["replies"]
    response_text = response[0].text
    conversation_history.append(ChatMessage.from_assistant(response_text))

    return response_text, retrieved_documents

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('message', '')

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    try:
        answer, retrieved_docs = chat_with_RAG(question)

        docs_output = [{
            'title': doc.meta['问题/内容'],
            'content': doc.content
        } for doc in retrieved_docs]

        chat_records.append({
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            "user_input": question,
            "system_response": answer
        })

        with open(os.path.join(output_folder, "chat_history.json"), "w", encoding="utf-8") as f:
            json.dump(chat_records, f, ensure_ascii=False, indent=2)

        return jsonify({"reply": answer, "retrieved_docs": docs_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
