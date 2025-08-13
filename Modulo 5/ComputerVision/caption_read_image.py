import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

KEY = os.environ.get('KEY')
ENDPOINT = os.environ.get('ENDPOINT')

client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'image.png')

with open(image_path, 'rb') as image_stream:
    try:
        result = client.analyze(
            image_data = image_stream,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
            gender_neutral_caption=True
        )

        print("Image analysis results:")

        if result.caption is not None:
            print(f"Caption: '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

        if result.read is not None and len(result.read.blocks) > 0:
            print("Read:")
            for line in result.read.blocks[0].lines:
                print(f"Line: '{line.text}', Bounding box {line.bounding_polygon}")
                for word in line.words:
                    print(f"Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

    except HttpResponseError as e:
        print(f"Erro na análise da image: {e}")