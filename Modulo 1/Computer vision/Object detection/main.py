import cv2

# Carregador o classificador cascade pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Carregar a image. 
imagem = cv2.imread('image.jpg')

# Converter a imagem para tons de cinza.
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar faces na images
faces = face_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55))

# Descomente para 
#faces = eye_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55))

# for (x, y, w, h) in faces:
#    cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)


for (x, y, w, h) in faces:
    hair_y = max(0, y - int(h / 2)) # ajustar a região acima do rosto
    combined_height = h + (y - hair_y) # altura combinada do cabelo e rosto

    cv2.rectangle(imagem, (x, hair_y), (x + w, hair_y + combined_height), (255, 0, 0), 2)

# Mostrar a image com a face detectada
cv2.imshow('Imagem com Detecções', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()