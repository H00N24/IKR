#!/usr/bin/env python3
import os

from skimage.feature import hog
from skimage.io import imread


class data_loader(object):

    def __init__(self):
        self.__root = os.path.dirname(os.path.realpath(__file__))
        self.__train = os.path.join(self.__root, 'train')
        self.__test = os.path.join(self.__root, 'dev')

        self.files = {}
        self.train = {'names': [],
                      'targets': [],
                      'img_raw': [],
                      'img_hog': [],
                      'img_hog_d': []}

        self.test = {'names': [],
                     'targets': [],
                     'img_raw': [],
                     'img_hog': [],
                     'img_hog_d': []}

    def __load_dir(self, data_dir, train=True):
        targets = sorted([int(x) for x in os.listdir(data_dir)])
        for x in targets:
            target = x
            x = os.path.join(data_dir, str(x))
            files = sorted(os.listdir(x))
            for f in files:
                file_path = os.path.join(x, f)
                name = f[:-4]
                if name not in self.files:
                    self.files[name] = {}
                self.files[name]['target'] = target
                self.files[name]['train'] = train
                if f[-4:] == '.png':
                    self.files[name]['image'] = imread(file_path,
                                                       as_grey=True,
                                                       plugin='pil')

    def load_train(self, train_dir=False):
        if not train_dir:
            train_dir = self.__train
        self.__load_dir(train_dir, train=True)

    def load_test(self, test_dir=False):
        if not test_dir:
            test_dir = self.__test
        self.__load_dir(test_dir, train=False)

    def load_data(self, test=True):
        if test:
            self.load_train()
            self.load_test()
        else:
            self.load_train()
            self.load_train(self.__test)

    def load_eval(self, data_dir='eval'):
        files = sorted(os.listdir(data_dir))
        for f in files:
            file_path = os.path.join(data_dir, f)
            name = f[:-4]
            if name not in self.files:
                self.files[name] = {}
            self.files[name]['target'] = None
            self.files[name]['train'] = False
            if f[-4:] == '.png':
                self.files[name]['image'] = imread(file_path,
                                                   as_grey=True,
                                                   plugin='pil')

    @staticmethod
    def __hog_wraper(image):
        return hog(image,
                   orientations=8,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(1, 1),
                   block_norm='L1',
                   visualise=True)

    def prepare_lists(self):
        for key, item in self.files.items():
            di = self.train if item['train'] else self.test
            di['names'].append(key)
            di['targets'].append(item['target'])
            fd, img = self.__hog_wraper(item['image'])
            di['img_hog'].append(img)
            di['img_hog_d'].append(fd)
