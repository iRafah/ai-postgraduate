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
        
        visual_features = [
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.READ,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.PEOPLE,
        ]
        
        result = client.analyze(
            image_data = image_stream,
            visual_features=visual_features,
            smart_crops_aspect_ratios=[0.9, 1.33],
            gender_neutral_caption=True,
            language="en"
        )
        
        # Print all analysis results to the console
        print("Image analysis results:")

        if result.caption is not None:
            print(" Caption:")
            print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

        if result.dense_captions is not None:
            print(" Dense Captions:")
            for caption in result.dense_captions.list:
                print(f"   '{caption.text}', {caption.bounding_box}, Confidence: {caption.confidence:.4f}")

        if result.read is not None and len(result.read.blocks) > 0:
            print(" Read:")
            for line in result.read.blocks[0].lines:
                print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
                for word in line.words:
                    print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

        if result.tags is not None:
            print(" Tags:")
            for tag in result.tags.list:
                print(f"   '{tag.name}', Confidence {tag.confidence:.4f}")

        if result.objects is not None:
            print(" Objects:")
            for object in result.objects.list:
                print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")

        if result.people is not None:
            print(" People:")
            for person in result.people.list:
                print(f"   {person.bounding_box}, Confidence {person.confidence:.4f}")

        if result.smart_crops is not None:
            print(" Smart Cropping:")
            for smart_crop in result.smart_crops.list:
                print(f"   Aspect ratio {smart_crop.aspect_ratio}: Smart crop {smart_crop.bounding_box}")

        print(f" Image height: {result.metadata.height}")
        print(f" Image width: {result.metadata.width}")
        print(f" Model version: {result.model_version}")
       

    except HttpResponseError as e:
        print(f"Erro na análise da image: {e}")