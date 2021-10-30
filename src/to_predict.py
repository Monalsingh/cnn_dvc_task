from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import get_data

def test_data_seperator(test_generator):
    data = []     # store all the generated data batches
    labels = []   # store all the generated label batches
    max_iter = 10  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in test_generator:
        data.append(d)
        labels.append(l)
        #i += 1
        #if i == max_iter:
        #    break
    return data, labels

def data_label_formatter(data, labels):
    data = np.array(data, dtype=object)
    print(data.shape)
    data = np.reshape(data, (data.shape[0] * data.shape[1],) + data.shape[2:])

    labels = np.array(labels, dtype=object)
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1],) + labels.shape[2:])

    return data, labels

if __name__=="__main__":
    model = load_model('saved_models/model_vgg16.h5')
    _, _, test = get_data.main()
    print(test)
    data, labels = test_data_seperator(test)
    data, labels = data_label_formatter(data, labels)
    print(data.shape)
    scores = model.evaluate(data, labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


