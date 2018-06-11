import os

import numpy as np
from scipy.fftpack import dct
import imageio
from skimage import color
from skimage import transform
from sklearn.decomposition import PCA


def hist(ima, num_coeffs=100, width=40):
    """Extracts num_coeffs bin (normalized) histogram of greyscale image."""
    # Convert to grayscale if color
    if len(ima.shape) == 3:
        ima = color.rgb2gray(ima)

    # Resize to width x width
    x = transform.resize(ima, (width, width))

    h = np.histogram(x, num_coeffs)[0]

    return h / np.sum(h)


def my_dct(ima, num_coeffs=100, width=40):
    """Extracts num_coeffs size DCT of greyscale image."""
    # Convert to grayscale if color
    if len(ima.shape) == 3:
        ima = color.rgb2gray(ima)

    # Resize to width x width
    x = transform.resize(ima, (width, width))

    # Compute sqrt(num_coeffs)*sqrt(num_coeffs) DCT and flatten
    coeffs = dct(dct(x, n=int(np.sqrt(num_coeffs)), axis=0),
                 n=int(np.sqrt(num_coeffs)), axis=1)
    return coeffs.flatten()


def my_pca(ima, pca=None, width=40):
    """Projects greyscale image to PCA space."""
    # Convert to grayscale if color
    if len(ima.shape) == 3:
        ima = color.rgb2gray(ima)
    # Resize to width x width
    x = transform.resize(ima, (width, width))
    x = x.flatten().reshape(1, -1)
    # Project
    return pca.transform(x)


def fit_pca(path_name, num_coeffs=100, width=40):
    """Fit PCA"""
    X = []
    # Read all images
    for dirName, subdirList, fileList in os.walk(path_name):
        for fname in sorted(fileList):
            base, extension = os.path.splitext(fname)
            # Add images to model
            if (extension == '.jpg' or extension == '.JPG' or
                    extension == '.png' or extension == '.PNG'):
                ima = imageio.imread('{}/{}'.format(dirName, fname))
                if len(ima.shape) == 3:
                    ima = color.rgb2gray(ima)
                x = transform.resize(ima, (width, width))
                X.append(x.flatten())

    X = np.array(X)
    # Fit PCA
    pca = PCA(n_components=num_coeffs, svd_solver='randomized', whiten=True)
    X = X - X.mean(axis=0)  # center at origin
    return pca.fit(X)
