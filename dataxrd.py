from numpy import array,save,load,argmax,swapaxes,loadtxt,arange
from matplotlib.pyplot import imshow,plot,figure,show
from scipy.optimize import curve_fit

from glob import glob
import re
import h5py

class DataXRD():
    """
    Class for processing XRD data.
    """
    def __init__(self,path = './',parameters = 'Scanning_Parameters.txt',calibration='Calibration.ini'):
        self.path = path
        self.parameters = parameters
        self.calibration = calibration

    def read_calibration_file(self,name=None):
        if name == None:
            name = self.path + '/' + self.calibration

        self.calib_data = loadtxt(name,unpack=True)

    @staticmethod
    def fce_second(x,a,b,c,d):
        return a * x**3 + b * x**2 + c * x +d 

    @staticmethod
    def fce_second(x,a,b,c):
        return a * x**2 + b * x + c 

    @staticmethod
    def fce_linear(x,a,b):
        return a * x + b 

    def calibrate_channels(self):
        x,y = self.calib_data
        self.opt,self.opt_var = curve_fit(self.fce_second,x,y)

        print('Calibrated data:',self.opt)

        self.c0 = arange(0,1280)
        self.cx = self.fce_second(self.c0,*self.opt)

    def read_params(self,name=None):
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
        if name == None:
            name = self.path + '/' + self.parameters

        print('Reading parameters from:',name)
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
        print(self.params)

    def read_xrd(self):
        names = sorted(glob(self.path + '/Frame*.dat'))

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

        self.source = array(z)[::-1]

    def reshape_source(self):
        self.reshaped = self.source.reshape(self.params['y'],self.params['x'],-1)
        self.shape = self.reshaped.shape

    def invert_reshaped(self):
        self.inverted = self.invert(self.reshaped)

    @property
    def all_spectra(self):
        if hasattr(self,'__all_spectra'):
            return self.__all_spectra
        else:
            self.__all_spectra = self.inverted.sum(axis = 0).sum(axis = 0) / (self.shape[0] * self.shape[1])
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

    def from_source(self):
        """
        Read data from source
        """
        
        self.read_params()
        self.read_xrd()
        self.reshape_source()
        self.invert_reshaped()

        return self

    def save_h5(self,name = None):

        if name == None:
            name = self.path + '/' + 'data.h5'

        print('Saving:',name)

        with h5py.File(name,'w') as f:
            f.create_dataset('inverted',data = self.inverted)
            f.create_dataset('reshaped',data = self.reshaped)
            f.create_dataset('source',data = self.source)

        return self

    def load_h5(self,name = None):

        if name == None:
            name = self.path + '/' + 'data.h5'

        print('Loading:',name)

        with h5py.File(name,'r') as f:
            x = f['inverted']
            self.inverted = x[:]

            x = f['reshaped']
            self.reshaped = x[:]
            self.shape = self.reshaped.shape

            x = f['source']
            self.source = x[:]

        return self
