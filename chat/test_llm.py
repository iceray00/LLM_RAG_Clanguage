from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage

generator = OllamaChatGenerator(model="zephyr",
                            url = "http://localhost:11434",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              })

messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
ChatMessage.from_user("你好！")]

print(generator.run(messages=messages))

# root@autodl-container-a63e40be4b-9396484e:~/iceray/chat# python3 test_llm.py
# {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='你好！很高兴能为你服务。有什么问题或需要帮助的吗？')], _name=None, _meta={'model': 'qwen2', 'created_at': '2025-02-07T02:39:38.537243871Z', 'done': True, 'done_reason': 'stop', 'total_duration': 263116317, 'load_duration': 28583897, 'prompt_eval_count': 24, 'prompt_eval_duration': 33000000, 'eval_count': 16, 'eval_duration': 188000000})]}



