import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder

import pytesseract

def load_model(path):
  try:
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    print("Loading model successfully...")
    return model
  except Exception as e:
    print(e)

wpod_net_path = "./model/wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255
  if resize:
    img = cv2.resize(img, (224,224))
  return img


def get_plate(image_path, Dmax=608, Dmin=256):
  vehicle = preprocess_image(image_path)
  ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
  side = int(ratio * Dmin)
  bound_dim = min(side, Dmax)
  _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
  return LpImg, cor

def sort_contours(cnts,reverse = False):
  i = 0
  boundingBoxes = [cv2.boundingRect(c) for c in cnts]
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
  return cnts

def predict_from_model(image,model,labels):
  image = cv2.resize(image,(80,80))
  image = np.stack((image,)*3, axis=-1)
  prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
  return prediction


def main():
  # Create a list of image paths 
  # image_paths = glob.glob("./dataset/images/*.png")
  image_paths = glob.glob("./dataset/br_plates/*.jpeg")
  print("Found %i images..."%(len(image_paths)))
# pre-processing input images and pedict with model

  for i in range(len(image_paths)):
  # for i in range(20):
    try:
      test_image = image_paths[i+20]
      LpImg,cor = get_plate(test_image)
      
      plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
      gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
      blur = cv2.GaussianBlur(gray,(7,7),0)
      binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
      kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
      thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

      cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      test_roi = plate_image.copy()
      crop_characters = []
      # digit_w, digit_h = 30, 60

      for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w

        if 0.5<=ratio<=1.5 or 3<=ratio<=4: # Only select contour with defined ratio
          if h/plate_image.shape[0]>=0.4: # Select contour which has the height larger than 40% of the plate

            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            # curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

      fig = plt.figure(figsize=(14,4))
      grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)


      json_file = open('./model/MobileNets_character_recognition.json', 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      model = model_from_json(loaded_model_json)
      model.load_weights("./model/License_character_recognition_weight.h5")

      labels = LabelEncoder()
      labels.classes_ = np.load('./model/license_character_classes.npy')
      fig = plt.figure(figsize=(15,3))
      cols = len(crop_characters)
      grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

      final_string = ''
      for i,character in enumerate(crop_characters):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character,model,labels))
        plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        plt.axis(False)
        plt.imshow(character,cmap='gray')

      # print(final_string)

      # cv2.imshow('Imagem em Escala de Cinza', test_roi)
      # cv2.waitKey(0)

  
    # Visualize results
      # plt.figure(figsize=(10,5))
      # plt.subplot(1,2,1)
      # plt.axis(False)
      # plt.imshow(binary, cmap="gray")

      # plt.subplot(1,2,2)
      # plt.axis(False)
      # plt.imshow(test_roi)
      
      # encontrarRoiPlaca('./dataset/images/Car01.jpeg')
    except Exception as e:
      print(e)
  
  # print(listOfPlate)

  plt.tight_layout()
  plt.show()



if __name__ == '__main__':
  main()


