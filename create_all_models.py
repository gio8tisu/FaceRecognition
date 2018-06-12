#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  create_all_models.py <rootName> <outPath> [--width=<wd>]\
   [--feat=<f>] [--numCoeffs=<nc>]
  create_all_models.py -h | --help
Options:
  --width=<wd>            Images will be rescaled to this width  [default: 40]
  --numCoeffs=<nc>        Number of coefficients to return       [default: 40]
  --feat=<f>              hist, dct, pca                       [default: hist]
"""

import os

from docopt import docopt
import imageio
import model

import features


def main():
    # read arguments
    args = docopt(__doc__)

    path_name = args['<rootName>']
    output_dir = args['<outPath>']
    num_coeffs = int(args['--numCoeffs'])  # size of feature vector
    width = int(args['--width'])  # width at which to be resized
    feat = args['--feat']

    print('Searching photos in ' + path_name)
    print('Resizing to ' + str(width))
    print('Using ' + feat + ' with ' + str(num_coeffs) + ' coefficients')

    if feat == 'hist':
        extract = features.hist
        extract_opt = {'num_coeffs': num_coeffs, 'width': width}
    elif feat == 'dct':
        extract = features.my_dct
        extract_opt = {'num_coeffs': num_coeffs, 'width': width}
    elif feat == 'pca':
        pca = features.fit_pca(path_name, num_coeffs, width)
        extract = features.my_pca
        extract_opt = {'pca': pca, 'width': width}

    ident = 0
    model_dict = {}

    # Create a dictionary with the different persons
    for dirName, subdirList, fileList in os.walk(path_name):
        print('Found directory: %s' % dirName)
        for fname in sorted(fileList):
            base, extension = os.path.splitext(fname)
            person_name = os.path.basename(dirName)

            # If model has not been created, create it
            if not(person_name in model_dict):
                model_dict[person_name] = model.Model(id=ident,
                                                      name=person_name)
                ident = ident + 1

            # Add images to model
            if (extension == '.jpg' or extension == '.JPG' or
                    extension == '.png' or extension == '.PNG'):

                # Read image and convert it to GRAY
                ima = imageio.imread('{}/{}'.format(dirName, fname))

                ###########################################
                # TODO: Create feature and use it here
                ###########################################
                coeffs = extract(ima, **extract_opt)

                # Add to model
                model_dict[person_name].add(coeffs)

    # Save all models
    for name in model_dict:
        model_name = '{}/{}_model.bin'.format(output_dir, name)
        model_dict[name].save(model_name)


if __name__ == '__main__':
    main()
