from llama_cpp import Llama
from typing import List, Dict, Any
from datetime import datetime as dt
import time
import json
import os

    
def chat(llm: Llama, 
         question: str, 
         message_history: List[str] = [], 
         temperature: float = 0.0, 
         max_tokens: int = 250) -> Dict[str, Any]:
    task = ("You are a helpful chatbot. "
            "You are given a question and a set of answer choices. "
            "Select the single best answer to the question as your response.")
    # sys_prompt = dict(role="system", content=task)
    user_prompt = dict(role="user", content=question)
    # message_history.append(sys_prompt)
    message_history.append(user_prompt)
    response = llm.create_chat_completion(messages=message_history,
                                          stream=True, 
                                          temperature=temperature, 
                                          max_tokens=max_tokens)
    text = []
    # Measure time to first token, tokens / second, total time, total tokens
    metrics = dict(model_name=(llm.model_path).split("/")[-1], 
                   input_tokens=len(llm.tokenize(question.encode("utf-8"))),
                   question=question)
    print("Beginning inference...")
    start = time.perf_counter()
    for idx, token in enumerate(response):
        if idx == 0:
            # Time to first token
            metrics["ttft"] = round(time.perf_counter() - start, 3)
        line = token["choices"][0]["delta"].get("content", "")
        text.append(line)
    print("Inference complete!")
    metrics["total_time"] = round(time.perf_counter() - start, 3)
    metrics["response"] = "".join(text)
    metrics["output_tokens"] = len(llm.tokenize((metrics["response"]).encode("utf-8")))
    metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
    metrics["inference_time"] = metrics["total_time"] - metrics["ttft"]
    metrics["tps"] = round(metrics["output_tokens"] / metrics["inference_time"], 3)
    
    return metrics


if __name__ == "__main__":
    # Load in evaluation data
    with open("../commonsenseqa.json", "r") as f:
        eval_ds = json.load(f)
    
    prompts = []
    for row in eval_ds:
        question = f"Question: {row['question']}\n"
        choices = '\n'.join([f'{label}:{text}' for label, text in 
                            zip(row["choices"]["label"], 
                                row["choices"]["text"])])
        prompt = f"{question}\n{choices}"
        prompts.append(prompt)

    model_path = os.environ["LANGUAGE_MODEL"]
    model_name = model_path.split("/")[-1]
    # For the raspberry pi 4 we need to set n_threads=4 to match the cores to speed up inference.
    # Likewise, we need to explicitly tell Llama to load the model into memory by disabling 
    # memory mapping and enabling memory lock:
    # chat_format = "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
    llm = Llama(model_path=model_path, 
                n_threads=4, 
                use_mmap=False, 
                use_mlock=True, 
                verbose=True, 
                n_ctx=800, 
                stop=["<|end|>"])
    results = []
    print(f"Evaluating {model_name}:")
    for prompt in prompts:
        try:
            # Record metrics
            metrics = chat(llm, prompt, temperature=0.0, max_tokens=250)
            # Update results
            results.append(metrics)
            llm.reset()
        except Exception as ex:
            print(f"Couldn't answer question: {prompt}")
            print(str(ex))
            continue

    print(f"Removing {model_path} from memory...")
    del llm
    # Write results out to disk
    with open(f"../results/commonsense/{model_name}_eval_results_{dt.now()}.json", "w") as f:
        json.dump(results, f)