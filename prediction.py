from keras.models import load_model
from OpenFiles import OpenFiles
import numpy as np

model = load_model('first_model2.h5')
stream = "/Users/Joel/Desktop/robotics research/Open"
stream1 = "/Users/Joel/Desktop/robotics research/test_img.jpg"


def prediction(model,stream):
    #img = OpenFiles(stream).transform_images()
    images = OpenFiles(stream).get_images()
    #print(np.argmax(model.predict(img)[0]))
    images = np.array(images)
    #print(type(images))
    images = images.reshape(images.shape[0], 256,256,3)
    for i in range(len(images)):
        temp = images[i]
        temp = temp.reshape(1,256,256,3)
        print(np.argmax((model.predict(temp)[0])))

prediction(model, stream)
