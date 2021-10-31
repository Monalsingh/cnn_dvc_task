import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class chestXray:
    def __init__(self,filename):
        self.filename =filename


    def predictionchestxray(self):
        # load model
        model = load_model('saved_models/model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        #result = model.predict(test_image)
        yhat_class = np.argmax(model.predict(test_image), axis=1)

        if yhat_class == 0:
            prediction = 'covid'
            return [{"image": prediction}]
        if yhat_class == 1:
            prediction = 'normal'
            return [{"image": prediction}]
        if yhat_class == 2:
            prediction = 'viral'
            return [{"image": prediction}]

