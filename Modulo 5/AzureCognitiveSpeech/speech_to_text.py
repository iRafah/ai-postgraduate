import os
from dotenv import load_dotenv
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig

load_dotenv()

KEY = os.getenv("KEY")
REGION = os.getenv("REGION")

speech_config = SpeechConfig(subscription=KEY, region=REGION)
speech_config.speech_recognition_language = "pt-br"

script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_dir, 'audio1.wav')
audio_config = AudioConfig(filename=audio_path)

speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

def recognize_speech():
    print("Reconhecendo fala pelo arquivo de áudio...")
    result = speech_recognizer.recognize_once()

    if result.reason == result.reason.RecognizedSpeech:
        print("Reconhecido: {}".format(result.text))
    elif result.reason == result.reason.NoMatch:
        print("Nenhuma fala pôde ser reconhecida")
    elif result.reason == result.reason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconhecimento de fala cancelado: {}".format(cancellation_details.reason))
        print(f"Detalhes do erro: {cancellation_details.error_details}")

if __name__ == "__main__":
    recognize_speech()