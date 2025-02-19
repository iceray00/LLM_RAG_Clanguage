import argparse
import time
import json
import os

from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from pathlib import Path
from tqdm import tqdm

# 获取当前时间戳
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
output_folder = Path("chat_records") / timestamp
output_folder.mkdir(parents=True, exist_ok=True)

# 初始化聊天记录列表
chat_records = []


def main():
    parser = argparse.ArgumentParser(description="RAG Chat Bot for C Language Learning!")

    parser.add_argument("-e", "--Emodel", type=str, default="nomic-embed-text:latest", help="Embedding Model")
    parser.add_argument("-g", "--Gmodel", type=str, default="qwen2:latest", help="Generator Model")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("-m", "--mode", type=str, default="knowledge", help="`test` or `knowledge`. Default is `knowledge`")

    # parser.add_argument("--input_file", type=str, help="input one file path for Contract source code")
    # parser.add_argument("--o", "-output", type=str, default="./vfcs/", help="output path for VFCS")
    # parser.add_argument("--model_name", type=str, default="Qwen1.5-32B-Q4", help="model name")
    # parser.add_argument("--ctx", type=int, default=default_ctx, help="The maximum length of the context")
    # parser.add_argument("--prompt5", type=str, default=None, help="Prompt5")
    # parser.add_argument("--prompt6", type=str, default=None, help="Prompt6")

    args = parser.parse_args()

    if args.mode == "knowledge":
        try:
            with open(Path("data") / "Summary_of_Knowledge_Points.json", "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)
        except FileNotFoundError:
            print("Error: JSON file not found.")
            exit(1)
    elif args.mode == "test":
        try:
            with open(Path("data") / "Simulated_Test.json", "r", encoding="utf-8") as f:
                knowledge_data = json.load(f)
        except FileNotFoundError:
            print("Error: JSON file not found.")
            exit(1)
    else:
        print("Error: Invalid mode. '-m' '--mode' Should input `test` or `knowledge`.")
        exit(1)

    # Embedding Model
    if args.Emodel == "default":
        document_store = ChromaDocumentStore()
        print("Now, Embedding Model is: all-MiniLM-L6-v2")
    else:
        # 使用自定义嵌入函数创建 ChromaDocumentStore
        document_store = ChromaDocumentStore(
            embedding_function="OllamaEmbeddingFunction",
            url="http://localhost:11434/api/embeddings",
            model_name=args.Emodel
        )
        print(f"### Embedding Model is: {args.Emodel} ###")

    print(f"### Generator Model is: {args.Gmodel} ###")

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

    # 将文档写入 ChromaDocumentStore的部分保持不变
    for doc in tqdm(documents, desc="Indexing documents", unit="document"):
        document_store.write_documents([doc])

    # 创建 ChatPromptBuilder 和 OllamaChatGenerator的部分保持不变
    prompt_builder = ChatPromptBuilder()
    generator = OllamaChatGenerator(
        model="qwen2:latest",
        url="http://localhost:11434",
        generation_kwargs={
            "temperature": 0.9,
        }
    )

    # 设置多轮对话的上下文的部分保持不变
    conversation_history = [
        ChatMessage.from_system("你是一个精通C语言的智能助教。用户会提问一些关于C语言编程和技术方面的问题，你要提供有帮助的答案。同时，这是一个RAG的场景，会给出对应的RAG的内容。")
    ]

    # 创建 Pipeline的部分保持不变
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", generator)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    def chat_with_RAG(question, conversation_history, retriever, top_k):
        # 首先，使用 retriever 获取相关文档
        retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

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
            print(f"\n## Round {i} ##")
            i += 1
            question = input("\n❓ 你的问题: \n")
            if question.lower() in ["exit", "quit"]:
                print("退出对话。")
                break

            # 保存当前的系统信息到聊天记录
            chat_record = {
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                "user_input": question,
                "system_response": ""
            }

            # 处理问题并调用 retriever 进行文档检索
            answer, conversation_history = chat_with_RAG(
                question=question,
                conversation_history=conversation_history,
                retriever=retriever,
                top_k=args.top_k
            )

            # 更新聊天记录中的系统响应
            chat_record["system_response"] = answer

            # 将当前聊天记录添加到列表中
            chat_records.append(chat_record)

            # 将聊天记录保存到 JSON 文件
            with open(os.path.join(output_folder, "chat_history.json"), "w", encoding="utf-8") as f:
                json.dump(chat_records, f, ensure_ascii=False, indent=2)

            # 输出回答
            print(f"\n🤖 助手回答:\n{answer}")

        except KeyboardInterrupt:
            print("\n对话中断。")
            break


if __name__ == "__main__":
    main()
