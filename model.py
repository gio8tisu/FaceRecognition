import numpy as np
import h5py
import os


class Model():
    def __init__(self, id=-1, name='', model_name=''):
        """ Constructor."""
        if model_name != '':
            with h5py.File(model_name, 'r') as hf:
                self.__id__ = np.int64(hf.get('id'))
                dset = hf.get('name')
                self.__name__ = dset.attrs['name']
                self.__data__ = np.array(hf.get('vects'))
        else:
            self.__id__ = id
            self.__data__ = []
            self.__name__ = name

    def save(self, model_name):
        """ Save model in file."""
        hf = h5py.File(model_name, 'w')
        hf.create_dataset('vects', data=self.__data__)
        hf.create_dataset('id', data=self.__id__)

        dt = h5py.special_dtype(vlen=bytes)
        dset = hf.create_dataset("name", (100,), dtype=dt)
        dset.attrs["name"] = self.__name__
        hf.close()

    def add(self, vec):
        """ Add feature vector to model."""
        self.__data__.append(vec)

    def size(self):
        """ Returns vector size, i.e. numer of faces."""
        return len(self.__data__)

    def __call__(self, num_vec):
        """ Access to feature vector of a face."""
        return self.__data__[num_vec]

    def id(self):
        """ Return model ID."""
        return self.__id__

    def name(self):
        """ Return model name"""
        return self.__name__


# Read all vectors in models_dir into an array
def load_models(models_dir):

    models = []
    for dirName, subdirList, fileList in os.walk(models_dir):
        for fname in sorted(fileList):
            extension = os.path.splitext(fname)[1]
            if extension == '.bin':
                print ('Reading model {}/{}'.format(dirName, fname))
                models.append(Model(model_name='{}/{}'.format(dirName, fname)))

    return models
