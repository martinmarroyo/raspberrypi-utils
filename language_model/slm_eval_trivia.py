from llama_cpp import Llama
from typing import List, Dict, Any
from datetime import datetime as dt
import time
import json
import os
import sys

# class Llama32ChatCompletionHandler:
#     """A custom implementation of LlamaChatCompletionHandler."""

#     def __call__(
#         self,
#         *,
#         llama: llama.Llama,
#         messages: List[llama_types.ChatCompletionRequestMessage],
#         functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
#         function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
#         tools: Optional[List[llama_types.ChatCompletionTool]] = None,
#         tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
#         temperature: float = 0.0,
#         top_p: float = 0.95,
#         top_k: int = 40,
#         stream: bool = False,
#         stop: Optional[Union[str, List[str]]] = [],
#         seed: Optional[int] = None,
#         response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
#         max_tokens: Optional[int] = None,
#         presence_penalty: float = 0.0,
#         frequency_penalty: float = 0.0,
#         repeat_penalty: float = 1.1,
#         model: Optional[str] = None,
#         logit_bias: Optional[Dict[str, float]] = None,
#         min_p: float = 0.05,
#         typical_p: float = 1.0,
#         tfs_z: float = 1.0,
#         mirostat_mode: int = 0,
#         mirostat_tau: float = 5.0,
#         mirostat_eta: float = 0.1,
#         logits_processor: Optional[llama.LogitsProcessorList] = None,
#         grammar: Optional[llama.LlamaGrammar] = None,
#         logprobs: Optional[bool] = None,
#         top_logprobs: Optional[int] = None,
#         use_mmap: bool = False, #rpi4 settings
#         use_mlock: bool = True, #rpi4 settings
#         n_threads: int = 4, #rpi4 settings
#         **kwargs,
#     ) -> Union[
#         llama_types.CreateChatCompletionResponse,
#         Iterator[llama_types.CreateChatCompletionStreamResponse],
#     ]:
#         # Initialize response processing based on `stream` parameter
#         if stream:
#             # When streaming, yield response chunks iteratively
#             return self._stream_response(llama, messages, temperature, top_p, top_k, **kwargs)
#         else:
#             # When not streaming, generate and return a full response
#             return self._generate_response(llama, messages, temperature, top_p, top_k, **kwargs)

#     def _generate_response(
#         self,
#         llama: llama.Llama,
#         messages: List[llama_types.ChatCompletionRequestMessage],
#         temperature: float,
#         top_p: float,
#         top_k: int,
#         **kwargs
#     ) -> llama_types.CreateChatCompletionResponse:
#         # Implement response generation logic here
#         # Example: llama generates a full response based on messages
#         print("Generating non-streamed response...")
#         response = llama.create_chat_completion(
#             messages=messages,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             **kwargs
#         )
#         return response

#     def _stream_response(
#         self,
#         llama: llama.Llama,
#         messages: List[llama_types.ChatCompletionRequestMessage],
#         temperature: float,
#         top_p: float,
#         top_k: int,
#         **kwargs
#     ) -> Iterator[llama_types.CreateChatCompletionStreamResponse]:
#         # Implement streaming response logic here
#         print("Starting streaming response...")
#         stream = llama.create_chat_completion_stream(
#             messages=messages,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             **kwargs
#         )
#         for chunk in stream:
#             yield chunk

#     def apply_chat_template(self, messages):
#         prompt = []
#         for message in messages:
#             if message["role"] == "system":
#                 sys_msg = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(sys_msg)
#             if message["role"] in ("user", "human"):
#                 query = f"<|start_header_id|>user<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(query)
#             if message["role"] in ("ai", "assistant"):
#                 ai = f"<|start_header_id|>assistant<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(ai)
#         if prompt[-1]["role"] in ("user", "human"):
#             prompt[-1] += "<|start_header_id|>assistant<|end_header_id|>"

#         return " ".join(prompt)


# class Llama32ChatFormat(LlamaChatCompletionHandler):
#     def apply_chat_template(self, messages):
#         prompt = []
#         for message in messages:
#             if message["role"] == "system":
#                 sys_msg = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(sys_msg)
#             if message["role"] in ("user", "human"):
#                 query = f"<|start_header_id|>user<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(query)
#             if message["role"] in ("ai", "assistant"):
#                 ai = f"<|start_header_id|>assistant<|end_header_id|>{message['content']}<|eot_id|>"
#                 prompt.append(ai)
#         if prompt[-1]["role"] in ("user", "human"):
#             prompt[-1] += "<|start_header_id|>assistant<|end_header_id|>"

#         return " ".join(prompt)

    
def chat(llm: Llama, 
         question: str, 
         message_history: List[str] = [], 
         temperature: float = 0.0, 
         max_tokens: int = 250) -> Dict[str, Any]:
    sys_prompt = dict(role="system", content="You are a helpful chat assistant.")
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
    metrics["inference_time"] = metrics["total_time"] - metrics["ttft"]
    metrics["response"] = "".join(text)
    metrics["output_tokens"] = len(llm.tokenize((metrics["response"]).encode("utf-8")))
    metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
    metrics["tps"] = round(metrics["output_tokens"] / metrics["inference_time"], 3)
    
    return metrics


if __name__ == "__main__":
    # Load in evaluation data
    with open("../qatrivia.json", "r") as f:
        eval_ds = json.load(f)

    model_path = os.environ["LANGUAGE_MODEL"]
    model_name = model_path.split("/")[-1]
    # For the raspberry pi 4 we need to set n_threads=4 to match the cores to speed up inference.
    # Likewise, we need to explicitly tell Llama to load the model into memory by disabling 
    # memory mapping and enabling memory lock:
    chat_format = "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
    llm = Llama(model_path=model_path, n_threads=4, use_mmap=False, use_mlock=True, verbose=True)
    results = []
    print(f"Evaluating {model_name}:")
    for question in eval_ds[:10]:
        try:
            # Record metrics
            metrics = chat(llm, question["question"], temperature=0.0, max_tokens=250)
            # Update results
            results.append(metrics)
            llm.reset()
        except Exception:
            print(f"Couldn't answer question: {question['question']}")
            continue

    print(f"Removing {model_path} from memory...")
    del llm
    # Write results out to disk
    with open(f"../results/{model_name}_eval_results_{dt.now()}.json", "w") as f:
        json.dump(results, f)

    sys.exit(0)