import requests
import json
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv
import os
import sys

load_dotenv()
KEY = os.environ.get("KEY")
ENDPOINT = os.environ.get("ENDPOINT")

detect_url = f"{ENDPOINT}face/v1.0/detect"

headers = {
    'Ocp-Apim-Subscription-Key': KEY,
    'Content-Type': 'application/json'
}

image_url = "https://img.buzzfeed.com/buzzfeed-static/static/2020-01/24/15/asset/a4a439fc5e1f/sub-buzz-1096-1579879662-3.jpg"
body = {
    'url': image_url
}

params = {
    'detectionModel': 'detection_03',
    'recognitionModel': 'recognition_04',
    'returnFaceId': 'false',
    'returnFaceAttributes': 'blur,glasses,headpose,mask,qualityforrecognition',
    'returnFaceLandmarks': 'true' # Pega os dados dos pontos do rosto
}

response = requests.post(detect_url, headers=headers, json=body, params=params)
# response = requests.post(detect_url, headers=headers, json=body)

if response.status_code == 200:
    faces = response.json()
    print(json.dumps(faces, indent=2))
else:
    print(f"Erro {response.status_code}: {response.json()}")
    sys.exit()

response_image = requests.get(image_url)
image = Image.open(BytesIO(response_image.content))

draw = ImageDraw.Draw(image)

for face in faces:
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
    
    landmarks = face['faceLandmarks']
    for landmark, point in landmarks.items():
        x, y = point['x'], point['y']
        draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0))

image.show()
image.save("output_image_with_landmark.png")