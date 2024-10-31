"""A demo application that takes in microphone input, generates a response 
from a language model, and returns a spoken response to the input."""
from llama_cpp import Llama 
from typing import List
import speech_recognition as SR 
import subprocess
import time
import os 


def speak(text: str):
    """Runs the Piper Text-To-Speech system to playback the given text"""
    # Get the TTS model
    tts_model = os.environ.get("TTS_MODEL")
    # Build the command (include the BUFFERING: buffer for now due to weird issue with ALSA cutting off first word)
    echo_command = ["echo", f"BUFFERING: {text.strip()}"]
    piper_command = ["piper", "--model", tts_model, "--output-raw"]
    aplay_command = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]

    try:
        # Echo the text to stdout
        with subprocess.Popen(echo_command, stdout=subprocess.PIPE) as echo_proc:
            # Create the audio of the text using Piper
            with subprocess.Popen(piper_command, stdin=echo_proc.stdout, stdout=subprocess.PIPE) as piper_proc:
                # Stream the audio playblack
                with subprocess.Popen(aplay_command, stdin=piper_proc.stdout) as aplay_proc:
                    aplay_proc.wait()
    except subprocess.CallProcessError as e:
        print(f"Command failed with error: {e.stderr}")


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
        speak(line)
        print(line, end="", flush=True)
        text.append(line)

    return " ".join(text)

def get_input_from_mic(recognizer: SR.Recognizer):
    """Reads input from the microphone and returns the transcribed text."""
    with SR.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)

    return audio


if __name__ == "__main__":
    # Speech Recognition
    recognizer = SR.Recognizer()
    audio = get_input_from_mic(recognizer)
    transcribed_text = recognizer.recognize_whisper(audio, language="english") 
    # Response generation + TTS output
    model_path = os.environ.get("LANGUAGE_MODEL")
    llm = Llama(model_path=model_path, 
                n_threads=4, 
                use_mmap=False, 
                use_mlock=True)
    chat(llm, transcribed_text)