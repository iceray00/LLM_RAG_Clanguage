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

    # knowledge 就是对应：“知识点回答”
    # test 对应的就是：“智能出题”

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

    # 将文档写入 ChromaDocumentStore
    for doc in tqdm(documents, desc="Indexing documents", unit="document"):
        document_store.write_documents([doc])

    # 创建 ChatPromptBuilder 和 OllamaChatGenerator
    prompt_builder = ChatPromptBuilder()
    generator = OllamaChatGenerator(
        model="qwen2:latest",
        # url="http://localhost:11434/api/chat",
        url="http://localhost:11434/",
        generation_kwargs={
            "temperature": 0.9,
        }
    )

    # 创建 Pipeline的部分保持不变
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", generator)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    # 设置多轮对话的上下文
    conversation_history = [
        ChatMessage.from_system("你是一个精通C语言的智能助教。用户会提问一些关于C语言编程和技术方面的问题，你要提供有帮助的答案。同时，这是一个RAG的场景，会给出对应的RAG的内容。而这个场景，正是“智能出题”，用户一般会问一个知识点，然后需要你给出一些题目！而这里RAG就是有一个标准题库，会检索出可能匹配的问题（也有可能出错），请你对合适的例子给出！")
    ]

    def chat_with_RAG_first_time(question, conversation_history, retriever, top_k):
        # 首先，使用 retriever 获取相关文档
        retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

        template = (
"## **问题**:\n" + question + "\n" +
"""
   ### 注意事项：（如果并没有给出具体问题，而是进行问候或者其他行为，请重申当前是“智能出题”场景，需要给出明确问题或者知识点！除非要求，请不要直接按照RAG固定输出的题目继续虚空答题！）
   1. 根据用户指定题目类型（如选择题、填空题、判断题、改错题、代码题等）、难度和数量出题。若未指定，随机选择题型，默认生成一道中等难度的题目。 
   2. 文档包含多个题目条目，请根据关键词标签检索出与问题最相关的5条内容作为参考，但这并不一定是最符合、贴切问题的答案，而是一些确切的、准确的相关题目。 
   3. 确保所参考条目与**用户问题**保持一致，避免仅靠关键词匹配而忽略上下文意义。
   4. 核查检索结果是否包含与问题要求无关的题目，避免误导答案的生成。  

   ### 输出格式：  
   - **题干**：清晰描述题目要求和背景信息，确保无歧义或错误，且有典型错误的干扰项。  
   - **参考答案**：提供明确的参考答案，如有多种正确答案，尽可能地包含。
   - **解释**：
      - 说明解题思路和关键知识点。  
      - 指出干扰项的常见错误及其原因。
**注意**：
如果并没有给出具体问题，而是进行问候或者其他行为，请重申当前是“智能出题”场景，需要给出明确问题或者知识点！除非要求，请不要直接按照RAG固定输出的题目继续虚空答题！
""")
        conversation_history.append(ChatMessage.from_user(template))

        # 将检索到的文档添加到历史对话中
        conversation_history.append(
            ChatMessage.from_system("RAG检索到的相关文档如下：（不一定是百分百匹配问题的！请你进行甄别）"))
        for doc in retrieved_documents:
            conversation_history.append(ChatMessage.from_system(f"{doc.meta['问题/内容']}: {doc.content}"))

        # 获取生成器的回答
        querying = pipe.run(data={
            "prompt_builder": {
                "template_variables": {"question": question},
                "template": conversation_history
            }
        })

        # 获取回答并更新历史对话
        response = querying["llm"]["replies"]
        response_text = response[0].text
        conversation_history.append(ChatMessage.from_assistant(response_text))

        return response_text, conversation_history


    def chat_with_RAG_second_time(question, conversation_history, retriever, top_k):
        # 首先，使用 retriever 获取相关文档
        retrieved_documents = retriever.run(query=question, top_k=top_k)["documents"]

        template = (
"## **问题**:\n" + question + "\n" +
"""
   ### 注意事项：（如果并没有给出具体问题，而是进行问候或者其他行为，请重申当前是“智能出题”场景，需要给出明确问题或者知识点！除非要求，请不要直接按照RAG固定输出的题目继续虚空答题！）
   1. 根据用户指定题目类型（如选择题、填空题、判断题、改错题、代码题等）、难度和数量出题。若未指定，随机选择题型，默认生成一道中等难度的题目。 
   2. 文档包含多个题目条目，请根据关键词标签检索出与问题最相关的5条内容作为参考，但这并不一定是最符合、贴切问题的答案，而是一些确切的、准确的相关题目。 
   3. 确保所参考条目与**用户问题**保持一致，避免仅靠关键词匹配而忽略上下文意义。
   4. 核查检索结果是否包含与问题要求无关的题目，避免误导答案的生成。  

   ### 输出格式：  
   - **题干**：清晰描述题目要求和背景信息，确保无歧义或错误，且有典型错误的干扰项。  
   - **参考答案**：提供明确的参考答案，如有多种正确答案，尽可能地包含。
   - **解释**：
      - 说明解题思路和关键知识点。  
      - 指出干扰项的常见错误及其原因。
**注意**：
如果并没有给出具体问题，而是进行问候或者其他行为，请重申当前是“智能出题”场景，需要给出明确问题或者知识点！除非要求，请不要直接按照RAG固定输出的题目继续虚空答题！
""")
        conversation_history.append(ChatMessage.from_user(template))

        # 将检索到的文档添加到历史对话中
        conversation_history.append(
            ChatMessage.from_system("RAG检索到的相关文档如下：（不一定是百分百匹配问题的！请你进行甄别）"))
        for doc in retrieved_documents:
            conversation_history.append(ChatMessage.from_system(f"{doc.meta['问题/内容']}: {doc.content}"))

        # 获取生成器的回答
        querying = pipe.run(data={
            "prompt_builder": {
                "template_variables": {"question": question},
                "template": conversation_history
            }
        })

        # 获取回答并更新历史对话
        response = querying["llm"]["replies"]
        response_text = response[0].text
        conversation_history.append(ChatMessage.from_assistant(response_text))

        return response_text, conversation_history



    # 交互式对话循环
    print("\n🟢 超级 C 语言助教已就绪（输入'exit'退出）")
    i = 1

    while True:
        try:
            print(f"\n## Round {i} ##")
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

            if i == 1:
                # 开始第一轮对话
                answer, conversation_history = chat_with_RAG_first_time(
                    question=question,
                    conversation_history=conversation_history,
                    retriever=retriever,
                    top_k=args.top_k
                )

            else:
                # 非第一轮对话
                answer, conversation_history = chat_with_RAG_second_time(
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

            i += 1

        except KeyboardInterrupt:
            print("\n对话中断。")
            break


if __name__ == "__main__":
    main()
