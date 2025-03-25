import pytesseract 
import cv2

img = cv2.imread('quote.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', img)