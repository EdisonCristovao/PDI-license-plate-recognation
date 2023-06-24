import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


img = cv2.imread('./src/img.jpg')
plate_image = cv2.convertScaleAbs(img, alpha=(255.0))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Imagem em Escala de Cinza', gray)

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.axis(False)
# plt.imshow(img)

# plt.subplot(1,2,2)
# plt.axis(False)
# plt.imshow(gray)



# plt.tight_layout()
# plt.show()
cv2.waitKey(0)
