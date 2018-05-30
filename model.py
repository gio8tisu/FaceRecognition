import numpy as np
import h5py
import os


class Model():
    # Constructor 
    def __init__(self, id = -1, name = '', model_name = ''):

        if model_name != '':
            with h5py.File(model_name, 'r') as hf:
                self.__id__   = np.int64(hf.get('id'))
                dset = hf.get('name')
                self.__name__ = dset.attrs['name']
                self.__data__  = np.array(hf.get('vects'))
        else:
            self.__id__   = id
            self.__data__ = []
            self.__name__ = name
        
    # Guarda el modelo en un fichero dentro del directorio especificado
    def save(self, model_name):
        
        hf = h5py.File(model_name, 'w')
        hf.create_dataset('vects', data=self.__data__)
        hf.create_dataset('id'   , data=self.__id__)
        
        dt = h5py.special_dtype(vlen=bytes)   # http://docs.h5py.org/en/latest/strings.html
        dset = hf.create_dataset("name", (100,), dtype=dt)
        dset.attrs["name"] = self.__name__
        hf.close()

        
    # Añade un vector de características al modelo (en el vector _data).
    def add(self, vec):
        self.__data__.append(vec)
        
    # Devuelve el tamaño del vector (el número de caras en el modelo)
    def size(self):
        return len(self.__data__)
    
    # Acceso al vector de características de una cara determinada
    def __call__(self, num_vec):
        return self.__data__[num_vec]
    
    # Método que retorna el identificador numérico
    def id(self):
        return self.__id__

    def name(self):
        return self.__name__


# Read all vectors in models_dir into an array
def load_models(models_dir):

    models = []
    for dirName, subdirList, fileList in os.walk(models_dir):
        for fname in sorted(fileList):
            extension = os.path.splitext(fname)[1]
            if extension == '.bin':
                print ('Reading model {}/{}'.format(dirName, fname))
                models.append(Model(model_name = '{}/{}'.format(dirName, fname)))

    return models
