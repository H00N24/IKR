#!/usr/bin/env python3

import sys
import numpy as np
from time import time

from data_loader import data_loader

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


TEST = False
if len(sys.argv) > 1:
    TEST = sys.argv[1]

classifiers = (('svc', SVC(C=1e2, probability=True)),
               ('mlp', MLPClassifier(hidden_layer_sizes=(800,),
                                     activation='logistic',
                                     learning_rate='adaptive',
                                     alpha=1e-4,
                                     max_iter=400,
                                     random_state=1)))

loader = data_loader()
if TEST:
    loader.load_data()
else:
    loader.load_data(test=False)
    loader.load_eval()

loader.prepare_lists()

X_train = loader.train['img_hog_d']
y_train = loader.train['targets']
names_train = loader.train['names']
X_test = loader.test['img_hog_d']
y_test = loader.test['targets']
names_test = loader.test['names']

for name, clf in classifiers:
    print('\n', clf)
    start_time = time()
    clf.fit(X_train, y_train)
    print('\n\tTraining time:', time() - start_time)

    if TEST:
        print('\tScore:', clf.score(X_test, y_test))
    else:
        start_time = time()
        pred_proba = clf.predict_log_proba(X_test)
        print('\tPrediction time:', time() - start_time)
        with open('image_{}.txt'.format(name), 'w') as out:
            for name, proba in zip(names_test, pred_proba):
                print(name,
                      1 + np.argmax(proba),
                      ' '.join([str(x) for x in proba.tolist()]),
                      file=out)
