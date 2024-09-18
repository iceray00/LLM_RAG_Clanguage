import json
import requests

model_url = "http://localhost:11434/api/generate"


def generate_prompt_response(prompt, model_name="Qwen1.5-32B-Q4", max_tokens_length=400, max_context_length=2048, mode="print"):
    """
    Generate a response from the LLM
    :param prompt: Prompt to send
    :param model_name: Model to use
    :param max_tokens_length: Generate up to this many tokens
    :param max_context_length: Max context length
    :return:
    """
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "max_tokens": max_tokens_length,
            "num_ctx": max_context_length,
            "temperature":0.02,
        }
    }

    if mode == "silence":
        response = requests.post(model_url, json=data, stream=True)
        if response.status_code == 200:
            response_text = ""
            prompt_eval_count = None
            eval_count = 0

            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode('utf-8'))

                    if prompt_eval_count is None:
                        prompt_eval_count = line_data.get("prompt_eval_count", None)

                    if "response" in line_data:
                        chunk = line_data["response"]
                        response_text += chunk

                    if line_data.get("done", False):
                        eval_count = line_data.get("eval_count", 0)
                        break

            # Once the processing is complete, you can use the response_text variable to further process the generated text
            # print("Generate: \n", response_text)
            print(f"*** Prompt tokens: {prompt_eval_count} ***")
            print(f"*** Generated tokens: {eval_count} ***")
            return response_text
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)

    elif mode == "print":
        try:
            response = requests.post(model_url, json=data, stream=True)
            response_text = ""
            prompt_eval_count = None
            eval_count = 0

            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode('utf-8'))

                    if prompt_eval_count is None:
                        # Extract prompt_eval_count from the first response line
                        prompt_eval_count = line_data.get("prompt_eval_count", None)
                        if prompt_eval_count is not None:
                            print(f"\n*****INPUT: Prompt token count: {prompt_eval_count} ***")

                    if "response" in line_data:
                        chunk = line_data["response"]
                        response_text += chunk
                        print(chunk, end='', flush=True)

                    if line_data.get("done", False):
                        eval_count = line_data.get("eval_count", 0)
                        break

            if prompt_eval_count is not None:
                total_tokens = prompt_eval_count + eval_count
                print(f"*****Generated token count: {eval_count} ***")
                print(f"*****Total token count (input + generated): {total_tokens} ***")
                print("******************************************************************\n")

            return response_text
        except Exception as e:
            print(f"\nRequest failed: {e}", flush=True)
            return ""

    else:
        print("Invalid mode. Please choose 'silence' or 'print'!!")
        return ""


if __name__ == "__main__":
    print(generate_prompt_response("Hello, how are you?", "qwen2:latest"))