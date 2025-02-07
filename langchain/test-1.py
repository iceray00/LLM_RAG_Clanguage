from langchain_ollama import OllamaLLM

# 使用正确的本地端口设置
llm = OllamaLLM(model="qwen2", host="localhost", port=11434)

# 输入提示
prompt = "Hello, how are you?"

# 获取模型响应
response = llm(prompt)

print(response)
