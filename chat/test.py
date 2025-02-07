from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack import Pipeline

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
import json
from pathlib import Path
from tqdm import tqdm

# 获取JSON文件的路径
json_file_path = Path("data") / "Simulated_Test.json"  # 假设JSON文件名是 Simulated_Test.json

# 读取并解析JSON文件
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        knowledge_data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    exit(1)

# 创建 ChromaDocumentStore 和 检索器
document_store = ChromaDocumentStore()
retriever = ChromaQueryTextRetriever(document_store=document_store)

# 生成 Haystack 文档
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

# 将文档写入 ChromaDocumentStore
for doc in tqdm(documents, desc="Indexing documents", unit="document"):
    document_store.write_documents([doc])

# 创建 ChatPromptBuilder 和 OllamaChatGenerator
prompt_builder = ChatPromptBuilder()
generator = OllamaChatGenerator(
    model="qwen2:latest",
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.9,

    }
)

# 设置多轮对话的上下文
conversation_history = [
    ChatMessage.from_system("你是一个智能助手，我会提问一些关于编程和技术方面的问题，你要提供有帮助的答案。")
]


# 创建 Pipeline
pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)
pipe.connect("prompt_builder.prompt", "llm.messages")


def chat_with_RAG(question, conversation_history, retriever, top_k=4):
    # 首先，使用 retriever 获取相关文档
    retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

    # print(f"\n## Retriever Answer: (top_k = {top_k})##\n")
    # for d in retrieved_documents:
    #     print(d.meta, d.score)

    # 将检索到的文档添加到历史对话中
    conversation_history.append(ChatMessage.from_system("相关文档如下："))
    for doc in retrieved_documents:
        conversation_history.append(ChatMessage.from_system(f"{doc.meta['问题/内容']}: {doc.content}"))

    # 获取生成器的回答
    querying = pipe.run(data={
        "prompt_builder": {
            "template_variables": {"question": question},
            "template": conversation_history + [ChatMessage.from_user(question)]
        }
    })

    # 获取回答并更新历史对话
    response = querying["llm"]["replies"]
    response_text = response[0].text
    conversation_history.append(ChatMessage.from_assistant(response_text))

    return response_text, conversation_history


# 交互式对话循环
print("\n🟢 对话RAG已就绪（输入'exit'退出）")
i = 1
while True:
    try:
        print(f"\n\n## Round {i} ##\n")
        i += 1
        question = input("\n❓ 你的问题: ")
        if question.lower() in ["exit", "quit"]:
            print("退出对话。")
            break

        # 处理问题并调用 retriever 进行文档检索
        answer, conversation_history = chat_with_RAG(
            question=question,
            conversation_history=conversation_history,
            retriever=retriever
        )

        # 输出回答
        print(f"\n🤖 助手回答:\n{answer}")

    except KeyboardInterrupt:
        print("\n对话中断。")
        break
