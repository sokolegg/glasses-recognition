import sys
import os
import numpy as np
import cv2
from keras.models import model_from_json
import time
import pickle
import pandas as pd

images = []
casc_path = 'cascades/haarcascade_frontalface_alt2.xml'
num_of_result = len(os.listdir('results')) + 1
results_path = 'results/' + str(num_of_result) + '/'
os.mkdir(results_path)


def get_full_path(file_name, file='.pkl'):
    folder = 'models/'
    full_path = folder + file_name + file
    return full_path


def load_keras_model(file_name):
    f = get_full_path(file_name, '.json')
    json_file = open(f, 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    weights_f = get_full_path(file_name, '.h5')
    loaded_model.load_weights(weights_f)
    return loaded_model


def load_sklearn_estimator(file_name):
    f = get_full_path(file_name)
    with open(f, 'rb') as file:
        estimator = pickle.load(file)
    return estimator


pca = load_sklearn_estimator('pca')
svc = load_sklearn_estimator('svc')
forest = load_sklearn_estimator('forest')
cnn = load_keras_model('cnn')


class Image:

    def __init__(self, path):
        self.path = path
        self.rgb_data = None
        self.gray_data = None
        self.faces_number = 0
        self.faces_xywh = []
        self.eyes_xywh = []
        self.eyes_data = []
        self.classes = []

    def predict(self):
        self.load_image()
        self.detect_faces()
        self.analyse_eyes()

    def load_image(self):
        self.rgb_data = cv2.imread(self.path)
        self.gray_data = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        face_cascade = cv2.CascadeClassifier(casc_path)
        self.faces_xywh = face_cascade.detectMultiScale(self.gray_data)
        self.faces_number = len(self.faces_xywh)

    def analyse_eyes(self):
        for (x, y, w, h) in self.faces_xywh:
            eyes_data = self.detect_eyes_on_face(x, y, w, h)
            self.predict_eyes_data(eyes_data)

    def detect_eyes_on_face(self, x, y, w, h):
        y_top = y + int(15 / 64. * h)
        y_bottom = y + int(35 / 64. * h)
        y_h = y_bottom - y_top
        [e_x, e_y, e_w, e_h] = (x, y_top, w, y_h)
        self.eyes_xywh.append((e_x, e_y, e_w, e_h))
        return self.gray_data[e_y: e_y + e_h, e_x: e_x + e_w]

    def predict_eyes_data(self, eyes_data):
        eyes_resized = np.array([cv2.resize(eyes_data, (64, 20))])
        eyes_x = eyes_resized.reshape(eyes_resized.shape[0], eyes_resized.shape[1], eyes_resized.shape[2], 1) / 255.
        self.eyes_data.append(eyes_x)
        predicted = self.predict_cnn(eyes_x)
        self.classes.append(predicted)

    def draw_image(self):
        for i in range(self.faces_number):
            x, y, w, h = self.faces_xywh[i]
            cv2.rectangle(self.rgb_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
            e_x, e_y, e_w, e_h = self.eyes_xywh[i]
            color = (255, 0, 0) if self.classes[i] == 1 else (0, 0, 255)
            cv2.rectangle(self.rgb_data, (e_x, e_y), (e_x + e_w, e_y + e_h), color, 2)
            name = self.path.split('/')[-1].split('.')[0]
            full_name = results_path + name + '_' + str(i) + '.jpg'
            cv2.imwrite(full_name, self.rgb_data)

    def predict_svc(self, eyes_x):
        eyes_x = eyes_x.reshape(1, -1)
        eyes_x = pca.transform(eyes_x)
        predicted = svc.predict(eyes_x)
        return predicted

    def predict_forest(self, eyes_x):
        eyes_x = eyes_x.reshape(1, -1)
        return forest.predict(eyes_x)

    def predict_cnn(self, eyes_x):
        return cnn.predict_classes(eyes_x)[0]


def predict_glasses():
    start = time.time()
    info = []
    for image in images:
        image.predict()
        image.draw_image()
        info.append(pd.DataFrame({'original_image': image.path,
                                  'predicted_class': image.classes,
                                  'eyes_position:': image.eyes_xywh}))
    csv_result = pd.concat(info, ignore_index=True)
    csv_result.to_csv(results_path + 'results.csv')
    print(csv_result)
    print('End of recognition. Time: %f s' % ((time.time() - start) * 1.))


def get_files_from_path(path):
    parser = '' if path[-1] == '/' else '/'
    for file in os.listdir(path):
        full_image_path = path + parser + file
        images.append(Image(full_image_path))


def get_images(paths):
    print("Analyse images from paths : ", paths)
    for path in paths:
        if os.path.isdir(path):
            get_files_from_path(path)
        else:
            images.append(Image(path))


def run():
    paths = sys.argv[1:]
    get_images(paths)
    predict_glasses()


if __name__ == '__main__':
    run()
