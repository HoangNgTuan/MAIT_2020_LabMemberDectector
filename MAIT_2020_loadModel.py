import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
image_size = (128, 128)

# Load model
model = load_model('/content/drive/Shareddrives/Lâm_Trần/Lan_Anh/model5_1/my_model.h5')
face_cascade = cv2.CascadeClassifier('/content/drive/Shareddrives/Lâm_Trần/MAIT2020/temp/haarcascade_frontalface_alt.xml')

def faceRecognize(imgPath):
  img = keras.preprocessing.image.load_img(
    imgPath, target_size=image_size
  )
  # img = cv2.imread(imgPath, 1)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # img = cv2.resize(img, image_size)
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Create batch axis
  predictions = model.predict(img_array)

  img1 = mpimg.imread(imgPath)
  imgplot = plt.imshow(img1)
  plt.show()

  score = predictions[0][0]
  print(
    "This image is %.2f percent Lan Anh."
    % (100 * score)
  
