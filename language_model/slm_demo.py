from llama_cpp import Llama 
from typing import List
import time
import os

def chat(llm: Llama, question: str, message_history: List[str] = [], temperature: float = 0.0, max_tokens: int = 250) -> str:
    user_prompt = dict(role="user", content=question)
    message_history.append(user_prompt)
    response = llm.create_chat_completion(messages=message_history, 
                                          stream=True, 
                                          temperature=temperature, 
                                          max_tokens=max_tokens)
    text = []
    for token in response:
        line = token["choices"][0]["delta"].get("content", "")
        print(line, end="", flush=True)
        text.append(line)
    
    return " ".join(text)


if __name__ == "__main__":
    model_path = os.environ["LANGUAGE_MODEL"]
    llm = Llama(model_path=model_path, n_threads=4, use_mmap=False, use_mlock=True, n_gpu_layers=-1, offload_kqv=True)
    
    question = input("Ask me a question:\t")
    chat(llm, question=question, temperature=0.0, max_tokens=500)
