from llama_cpp import Llama 
from typing import List
import time
import os

def chat(llm: Llama, question: str, message_history: List[str] = [], temperature: float = 0.0, max_tokens: int = 250) -> str:
    sys_prompt = dict(role="system", 
                      content="Do not answer this question using previous context unless explicitly instructed to do so")
    message_history.append(sys_prompt)
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
    llm = Llama(model_path=model_path, n_threads=3, use_mmap=False, use_mlock=True)

    while True:
        prompt = input("Enter prompt:\t")
        llm = Llama(model_path=model_path, n_threads=3, use_mmap=False, use_mlock=True)
        if prompt.lower() in ("quit", "exit", "q", "end"):
            print("Goodbye!")
            break
        
        chat(llm, question=prompt, temperature=0.0)
        del llm
        continue
