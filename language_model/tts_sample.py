import subprocess
import os

def tts(text: str):
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



if __name__ == "__main__":
    with open("sample.txt", "r") as f:
        text = f.read().split(" ")
        for i in range(0, len(text)-3, 3):
            print(text[i:i+3])
            tts(" ".join(text[i:i+3]))