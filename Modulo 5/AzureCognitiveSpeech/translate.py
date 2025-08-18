import os
from dotenv import load_dotenv
from azure.cognitiveservices.speech import AudioConfig
from azure.cognitiveservices.speech.translation import SpeechTranslationConfig, TranslationRecognizer

# Carregar variáveis de ambiente
load_dotenv()
KEY = os.getenv("KEY")
REGION = os.getenv("REGION")

# Configuração do serviço de tradução de fala
translation_config = SpeechTranslationConfig(subscription=KEY, region=REGION)
translation_config.speech_recognition_language = "pt-BR"  # Idioma da fala original
translation_config.add_target_language("en")  # Idioma para tradução

# Configuração de áudio (use um arquivo de áudio local)
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_dir, 'audio1.wav')  # Entrada do audio em .wav
audio_config = AudioConfig(filename=audio_path)

# Inicialização do reconhecedor de tradução
recognizer = TranslationRecognizer(translation_config=translation_config, audio_config=audio_config)

# Função de tradução de fala
def translate_speech():
    print("Traduzindo fala do arquivo de áudio...")
    result = recognizer.recognize_once()
    if result.reason == result.reason.TranslatedSpeech:
        print("Fala reconhecida: {}".format(result.text))
        print("Tradução para inglês: {}".format(result.translations["en"]))
    elif result.reason == result.reason.NoMatch:
        print("Nenhuma fala foi reconhecida")
    elif result.reason == result.reason.Canceled:
        cancellation_details = result.cancellation_details
        print("Tradução de fala cancelada: {}".format(cancellation_details.reason))
        print(f"Detalhes do erro: {cancellation_details.error_details}")

# Executar a função de tradução de fala
if __name__ == "__main__":
    translate_speech()