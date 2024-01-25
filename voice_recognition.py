# import speech_recognition as sr
#
#
# def recognize_voice_command():
#     recognizer = sr.Recognizer()
#
#     with sr.Microphone() as source:
#         print("Listening for a voice command...")
#         audio = recognizer.listen(source)
#
#     try:
#         # Using Sphinx (offline) for Russian speech recognition
#         command = recognizer.recognize_sphinx(audio, language="ru-RU")
#         print(f"Recognized command: {command}")
#         # Add your logic for handling the recognized command here
#
#     except sr.UnknownValueError:
#         print("Sorry, I couldn't understand the audio.")
#
#     except sr.RequestError as e:
#         print(f"Error with the speech recognition service; {e}")
#
#
# if __name__ == "__main__":
#     recognize_voice_command()
import speech_recognition as sr

recognizer = sr.Recognizer()


def record_audio():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    return audio


def recognize_speech(audio):
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Sorry, there was an error processing your request.")


if __name__ == "__main__":
    audio = record_audio()
    recognize_speech(audio)
