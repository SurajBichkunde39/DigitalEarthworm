import os
import pickle
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model as load
from .utils import plot_graph


class Model_clf(object):
    def __init__(self):
        cwd = os.getcwd()
        # self.model_path = os.path.join(cwd, 'models/vgg16_model.h5')
        self.model_path = os.path.join(cwd, 'models/mobilenet-fine-tuned.h5')
        self.class_indices = None
        self.model = None
        self.load_class_indices()

    def load_class_indices(self):
        cwd = os.getcwd()
        pkl_path = os.path.join(cwd, 'application/class_indices.pkl')
        with open(pkl_path, 'rb') as f:
            self.class_indices = pickle.load(f)

    def preprocess_img(self, path):
        pass

    def load_model(self):
        self.model = load(self.model_path)

    def predict(self, path):
        image = Image.open(path)
        image = image.resize(size=(224, 224))
        np_array = np.asarray(image)
        np_array = np_array.reshape(1, 224, 224, 3)
        predictions = self.model.predict(np_array)
        label = predictions[0].argmax()
        prob_dict = {}
        for i, j in enumerate(predictions[0]):
            prob_dict[self.class_indices[i]] = j
        # print(prob_dict)
        plot_path = plot_graph(prob_dict, title="Weight of Each Lable", size=(10, 10))
        return self.class_indices[label], plot_path, prob_dict
