from numpy import array,save,load,argmax,swapaxes
import re
from glob import glob
from matplotlib.pyplot import imshow,plot,figure,show
import h5py

class DataXRD():
    """
    Class for processing XRD data.
    """

    def __init__(self,path = './'):
        self.path = path

    def read_params(self,name):
        """
        Process scanning parameters.

        Parameters
        ---------
        name: str
            name of the parameters file. e.g. Scanning_Parameters.txt
        
        Returns
        -------
        dictionary
            a dictionary of parameters
        """

        params = {}

        with open(name,'r') as f:
            for line in f:
                try:
                    key,value = line.split('=')
                    params[key] = value

                except:
                    x = re.search('AXIS: (\S)',line)
                    n = re.search('STEP: (\d+)',line)
                    if n and x:
                        params[x.group(1)] = int(n.group(1))

        self.params = params

    def read_xrd(self):
        names = sorted(glob(self.path + '/*.dat'))

        print("Reading data")

        self.__read_xrd__(names)

        print("Done")

    def __read_xrd__(self,names):
        """
        Reads the source data.

        Parameters
        ---------
        names: list
            a list of file names.

        Returns
        -------
        numpy array
            3 dimmensional array x,y,spectra
        """

        z = []
        for file in names:

            y = []
            with open(file,mode='r') as f:
                for line in f:
                    x = line.split()[1]
                    y += [int(x)]

            z += [array(y)]

        self.source = array(z)

    def reshape_source(self):
        self.reshaped = self.source.reshape(self.params['y'],self.params['x'],-1)

    def invert_reshaped(self):
        self.inverted = self.invert(self.reshaped)

    @property
    def all_spectra(self):
        if hasattr(self,'__all_spectra'):
            return self.__all_spectra
        else:
            self.__all_spectra = self.inverted.sum(axis = 0).sum(axis = 0)
            return self.__all_spectra

    @property
    def integrated_spectra(self):
        if hasattr(self,'__all_spectra'):
            return self.__integrated_spectra
        else:
            self.__integrated_spectra = self.inverted.sum(axis = 2)
            return self.__integrated_spectra

    @staticmethod
    def invert(z):
        """
        Invert every second row

        Parameters
        ---------
        z: numpy array
            3d numpy array

        Returns
        -------
        numpy array
            3 dimmensional array x,y,spectra
        """

        s = 1
        x = []
        for v in z:
            if s == 1:
                x += [v]
            else:
                x += [v[::-1]]
            s *= -1

        return array(x)

    def save_h5(self,name = 'data.h5'):
        with h5py.File(name,'w') as f:
            f.create_dataset('inverted',data = self.inverted)
            f.create_dataset('reshaped',data = self.reshaped)
            f.create_dataset('source',data = self.source)

        return self

    def load_h5(self,name = 'data.h5'):
        with h5py.File(name,'r') as f:
            x = f['inverted']
            self.inverted = x[:]

            x = f['reshaped']
            self.reshaped = x[:]

            x = f['source']
            self.source = x[:]

        return self
