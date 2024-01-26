import speech_recognition as sr

recognizer = sr.Recognizer()


def record_audio(timeout=None,phrase_time_limit=None):
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                if audio:
                    return audio
            except sr.WaitTimeoutError:
                print("Timed out. Listening again...")


def recognize_speech(audio):
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        # Add conditions based on recognized commands to perform specific actions
        # For example, you can have different if statements for different commands
        if "turn on the lights" in text.lower():
            print("Turning on the lights...")
            # Add logic to control lights
        elif "play music" in text.lower():
            print("Playing music...")
            # Add logic to play music
        # Add more conditions as needed
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Sorry, there was an error processing your request.")


if __name__ == "__main__":
    while True:
        audio = record_audio()
        try:
            trigger_phrase = recognizer.recognize_google(audio)
            print(f"You said: {trigger_phrase}")
            if "adjutant" in trigger_phrase.lower():
                print("Listening for commands...")
                audio = record_audio(phrase_time_limit=7)  # Listen for 7 seconds
                recognize_speech(audio)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the trigger phrase.")
        except sr.RequestError:
            print("Sorry, there was an error processing the trigger phrase.")
