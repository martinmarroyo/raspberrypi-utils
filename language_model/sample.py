import speech_recognition as SR 

def get_input_from_mic(recognizer: SR.Recognizer):
    """Reads input from the microphone and returns the transcribed text."""
    with SR.Microphone() as source:
        print("Say something!")
        audio = recognizer.listen(source)

    return audio


if __name__ == "__main__":

    recognizer = SR.Recognizer()
    audio = get_input_from_mic(recognizer)
    transcribed_text = recognizer.recognize_whisper(audio, language="english")
    print(f"Whisper heard: {transcribed_text}")


