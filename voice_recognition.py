import time
import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def record_audio(timeout=None, phrase_time_limit=None):
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
        handle_command(text)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Sorry, there was an error processing your request.")

def handle_command(text):
    print("here")
    text = text.lower()
    print("here")
    if "turn on the lights" in text:
        respond("Turning on the lights...")
    elif "play music" in text:
        respond("Playing music...")
    else:
        respond("Sorry, I don't know how to do that yet.")

def respond(message):
    try:
        print(message)
        engine.say(message)
        print("Running TTS engine...")
        engine.runAndWait()
        print("TTS engine finished.")
        time.sleep(0.5)
    except Exception as e:
        print(f"Error in TTS: {e}")

if __name__ == "__main__":
    while True:
        audio = record_audio()
        try:
            trigger_phrase = recognizer.recognize_google(audio)
            print(f"You said: {trigger_phrase}")
            if "adjutant" in trigger_phrase.lower():
                respond("Listening for commands...")
                audio = record_audio(phrase_time_limit=7)
                recognize_speech(audio)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the trigger phrase.")
        except sr.RequestError:
            print("Sorry, there was an error processing the trigger phrase.")
