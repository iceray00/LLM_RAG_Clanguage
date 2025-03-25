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

    # parser.add_argument("--input_file", type=str, help="input one file path for Contract source code")
    # parser.add_argument("--o", "-output", type=str, default="./vfcs/", help="output path for VFCS")
    # parser.add_argument("--model_name", type=str, default="Qwen1.5-32B-Q4", help="model name")
    # parser.add_argument("--ctx", type=int, default=default_ctx, help="The maximum length of the context")
    # parser.add_argument("--prompt5", type=str, default=None, help="Prompt5")
    # parser.add_argument("--prompt6", type=str, default=None, help="Prompt6")

    args = parser.parse_args()
    #
    # if args.mode == "knowledge":
    #     try:
    #         with open(Path("data") / "Summary_of_Knowledge_Points.json", "r", encoding="utf-8") as f:
    #             knowledge_data = json.load(f)
    #     except FileNotFoundError:
    #         print("Error: JSON file not found.")
    #         exit(1)
    # elif args.mode == "test":
    #     try:
    #         with open(Path("data") / "Simulated_Test.json", "r", encoding="utf-8") as f:
    #             knowledge_data = json.load(f)
    #     except FileNotFoundError:
    #         print("Error: JSON file not found.")
    #         exit(1)
    # else:
    #     print("Error: Invalid mode. '-m' '--mode' Should input `test` or `knowledge`.")
    #     exit(1)

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

    # retriever = ChromaQueryTextRetriever(document_store=document_store)

    # documents = []
    # for item in knowledge_data:
    #     doc = Document(
    #         content=item["title"],
    #         meta={
    #             "问题/内容": item["title"],
    #             "答案/内容": item["content"],
    #             "部分": item["part"]["section"],
    #             "题型/类型": item["part"]["classification"],
    #             "关键词": ", ".join(item["tags"]) if isinstance(item["tags"], list) else item["tags"]
    #         }
    #     )
    #     documents.append(doc)

    # # 将文档写入 ChromaDocumentStore的部分保持不变
    # for doc in tqdm(documents, desc="Indexing documents", unit="document"):
    #     document_store.write_documents([doc])

    # 创建 ChatPromptBuilder 和 OllamaChatGenerator的部分保持不变
    prompt_builder = ChatPromptBuilder()
    generator = OllamaChatGenerator(
        model="qwen2:latest",
        # url="http://localhost:11434/api/chat",
        url="http://localhost:11434/",
        generation_kwargs={
            "temperature": 0.9,
        }
    )

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", generator)
    pipe.connect("prompt_builder.prompt", "llm.messages")


    # 第一轮
    conversation_history = [
        ChatMessage.from_system("你是一个精通C语言的智能助教。用户会提问一些关于C语言编程和技术方面的问题，你要提供有帮助的答案。同时，当前场景是“代码解释”，只需要专注于对用户提出的代码片段进行详尽的解释即可！")
    ]

    def chat_with_RAG_first_time(question, conversation_history):

        template = (
"### **问题**\n" + question + """\n### **注意事项**：  
   1. 简要说明代码的主要功能和核心逻辑，包括关键部分如函数、变量、循环、条件判断等。  
   2. 梳理函数的调用关系及其作用，分析循环和条件判断逻辑对程序流程的影响。  
   3. 如涉及输入输出或数据处理，请描述其方式及意义。  
   4. 识别可能存在的重复逻辑或功能，若有，提出优化建议。  
   5. 请确保描述清晰明了，避免冗长和复杂表述。  

**注意**：
如果并没有给出代码，而是进行问号或者其他行为，请重申当前是“代码解释”场景，需要输入代码！除非要求，请不要进行虚空代码捏造！

""")

        # 获取生成器的回答
        querying = pipe.run(data={
            "prompt_builder": {
                "template_variables": {"question": question},
                "template": conversation_history + [ChatMessage.from_user(template)]
            }
        })

        # 获取回答并更新历史对话
        response = querying["llm"]["replies"]
        response_text = response[0].text
        conversation_history.append(ChatMessage.from_assistant(response_text))

        return response_text, conversation_history

    def chat_with_RAG_second_time(question, conversation_history):
        template = (
"请你结合此前给出的语言代码格式，并根据用户的问题需求进行详细的解释和分析。请结合前面的代码对新代码进行分析：\n" +
"### **问题**\n" + question + """\n
   请判断新代码与前面代码的关系并分析：  
   1. **延续或衔接**：说明新增部分与前面代码的逻辑衔接及扩展功能，分析新增变量或函数的作用。  
   2. **新功能或模块**：如为新增内容，请独立分析其功能意义及其在整体程序中的作用。  
   3. **性能优化**：如涉及优化，评估优化的必要性及其对执行效率或功能的提升。  
   4. **修改或重构**：如为修改或重构代码，请找到错误并修改，分析改动对可读性及性能的影响。  
   5. **调试与修复**：如为调试代码，需说明其对正常程序的影响，确认是否存在进一步清理或优化空间。

**注意**：
如果并没有给出代码，而是进行问号或者其他行为，请重申当前是“代码解释”场景，需要输入代码！除非要求，请不要进行虚空代码捏造！

""")
        # 完整的prompt
        querying = pipe.run(data={
            "prompt_builder": {
                "template_variables": {"question": question},
                "template": conversation_history + [ChatMessage.from_user(template)]
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

            if i == 1:
                # 开始第一轮对话
                answer, conversation_history = chat_with_RAG_first_time(
                    question=question,
                    conversation_history=conversation_history
                )

            else:
                # 非第一轮对话
                answer, conversation_history = chat_with_RAG_second_time(
                    question=question,
                    conversation_history=conversation_history,
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
