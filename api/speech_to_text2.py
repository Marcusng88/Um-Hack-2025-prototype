import speech_recognition as sr

def speech_prompt():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=2, phrase_time_limit=15)
        print("Processing your audio...")

    try:
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        text = ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        text = ""