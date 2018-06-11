#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  test_all.py <imagesPath> <modelsDir> [--width=<wd>] [--feat=<f>]\
   [--rule=<r>] [--kValue=<kv>] [--norm=<n>]
  test_all.py -h | --help
Options:
  --kValue=<kv>       Value of k (number of neightbors to use)  [default: 3]
  --width=<wd>        Images will be rescaled to this width     [default: 40]
  --norm=<n>          Distance norm 1,2,...                     [default: 2]
  --feat=<f>          hist, dct, pca                            [default: hist]
  --rule=<r>          knn                                       [default: knn]
"""

import os

from docopt import docopt
import imageio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import model
import features
import classifiers

if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    images_path = args['<imagesPath>']      #
    models_dir = args['<modelsDir>']       #
    k = int(args['--kValue'])    # k value for knn
    width = int(args['--width'])     # width of rescalated image
    order = int(args['--norm'])      # norm order
    feat = args['--feat']
    rule = args['--rule']

    print('Searching photos in ' + images_path)
    print('Searching models in ' + models_dir)
    print('Resizing to ' + str(width))
    print('Using ' + feat)
    print('Using ' + rule + ' classification rule')

    model_list = model.load_models(models_dir)
    # All models should have consistent feature vectors.
    # It should be safe to take the lenght of the first vector
    num_coeffs = len(model_list[0](0))

    if feat == 'hist':
        extract = features.hist
        extract_opt = {'num_coeffs': num_coeffs, 'width': width}
    elif feat == 'dct':
        extract = features.my_dct
        extract_opt = {'num_coeffs': num_coeffs, 'width': width}
    elif feat == 'pca':
        pca = features.fit_pca('data/train', num_coeffs, width)
        extract = features.my_pca
        extract_opt = {'pca': pca, 'width': width}

    if rule == 'knn':
        classify = classifiers.knn
        rule_opt = {'k': k, 'ord': order}
    elif rule == 'svm':
        classify = classifiers.svm

    correctly_classified = 0
    total_images = 0

    y_true = []  # True labels
    y_pred = []  # Predicted labels

    # Read all images in the given folder.
    # All images should be cropped faces from the same individual
    for dirName, subdirList, fileList in os.walk(images_path):
        for fname in sorted(fileList):
            extension = os.path.splitext(fname)[1]
            if (extension == '.jpg' or extension == '.JPG' or
                    extension == '.png' or extension == '.PNG'):
                total_images = total_images + 1

                ima = imageio.imread('{}/{}'.format(dirName, fname))

                ##########################################
                # TODO: Extract the same feature as models
                ##########################################
                coeffs = extract(ima, **extract_opt)

                ##########################################
                # TODO: Extract best model
                ##########################################
                best_id = classify(coeffs, model_list, **rule_opt)

                # Test/validation images should be stored in directories
                # named according to the person name.
                # (the name of the directory is the label for this class)
                ground_truth_name = os.path.basename(dirName)

                y_true.append(ground_truth_name)
                y_pred.append(model_list[best_id].name())

                if model_list[best_id].name() == ground_truth_name:
                    correctly_classified = correctly_classified + 1
                else:
                    print ("""ERROR: Image {}, true name = {}, '
                        hypothesis name = {}
                        """.format(fname,
                                   ground_truth_name,
                                   model_list[best_id].name()))

    print ('Total classification error: {0:.2f}%'.format(
        (1.0 - correctly_classified / total_images) * 100.0))

    # Advanced reporting:
    print (confusion_matrix(y_true, y_pred))
    print (classification_report(y_true, y_pred, digits=3))
