from keras.preprocessing.image import ImageDataGenerator


def our_preprocessing_function(filename):
    #Combines all the transformations
    img = cv2.imread(filename)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe2.apply(img)
    cv2.imwrite('tmp/clahe.jpg', image)
    img = cv2.imread("tmp/clahe.jpg")
    return img

def preprocess_data():
    img_datagen = ImageDataGenerator(rescale=1. / 255,
                                     preprocessing_function=our_preprocessing_function)
    return img_datagen


def get_data(data_path):
    img_datagen = preprocess_data()
    train_it = img_datagen.flow_from_directory(data_path[0], class_mode='categorical')
    test_it = img_datagen.flow_from_directory(data_path[1], class_mode='categorical')
    return train_it, test_it