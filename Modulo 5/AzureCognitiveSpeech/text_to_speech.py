import os
from dotenv import load_dotenv
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig

load_dotenv()

# Carregar variáveis de ambiente
KEY = os.getenv("KEY")
REGION = os.getenv("REGION")

speech_config = SpeechConfig(subscription=KEY, region=REGION)
speech_config.speech_synthesis_language = "pt-BR"
# speech_config.speech_synthesis_voice_name = "pt-BR-JulioNeural"
speech_config.speech_synthesis_voice_name = "pt-BR-LeilaNeural"
# https://learn.microsoft.com/pt-br/azure/ai-services/speech-service/language-support?tabs=tts


audio_filename = "output_audio_female.wav"
audio_config = AudioConfig(filename=audio_filename)

synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

text = "O Luquinhas acabou de perder de 14 à 1 para o time das meninas!"

def text_to_speech():
    print("Convertendo texto para fala...")
    result = synthesizer.speak_text(text)

    if result.reason == result.reason.SynthesizingAudioCompleted:
        print(f"Fala sintetizada e salva em '{audio_filename}'")
    elif result.reason == result.reason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Conversão de texto para fala cancelada: '{cancellation_details.reason}'")
        print(f"Detalhes do erro: '{cancellation_details.error_details}'")
        
if __name__ == '__main__':
    text_to_speech()