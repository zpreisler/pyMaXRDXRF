from numpy import array,save,load,argmax,swapaxes,loadtxt,arange,pad,roll,minimum,sqrt,expand_dims,log,unravel_index,asarray
from matplotlib.pyplot import imshow,plot,figure,show
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import kurtosis
from numpy import fft,uint8

from glob import glob
import re
import h5py
from scipy import signal

class Calibration():
    """
    Channels Calibration Class.
    """
    def __init__(self,name,parent=None):

        self.data = array([[1,2,3],[1,2,3]])

        try:
            self.data = loadtxt(name,unpack=True)
            print('Calibration data:',self.data)

        except:
            print(name,'is missing.')
            print('Calibration data:',self.data)

        self.calibrate()


    @staticmethod
    def fce_second(x,a,b,c):
        return a * x**2 + b * x + c 

    def calibrate(self):

        x,y = self.data
        self.opt,self.opt_var = curve_fit(self.fce_second,x,y)

        print('Calibrated data:',self.opt)

        self.c0 = arange(0,1280)
        self.cx = self.fce_second(self.c0,*self.opt)
        self.ic = interp1d(self.cx,self.c0,fill_value='extrapolate')

    def c(self,x):
        return self.fce_second(x,*self.opt)

class DataXRD():
    """
    Class for processing XRD data.
    """
    def __init__(self,path = './',parameters = 'Scanning_Parameters.txt',calibration='Calibration.ini'):
        self.path = path
        self.parameters = parameters
        self.calibration = calibration

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
        names = sorted(glob(self.path + '/[F,f]rame*.dat'), key=lambda x: int(re.sub('\D','',x)))

        print("Reading data")
        self.__read_xrd(names)
        print("Done")

    def __read_xrd(self,names):
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

    def calibrate(self):
        self.calibration = Calibration(self.path + '/' + self.calibration,self)

    def reshape(self):
        self.reshaped = self.source.reshape(self.params['y'],self.params['x'],-1)
        self.shape = self.reshaped.shape

    def invert(self):
        def invert(z):
            """
            Invert every second row
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

        self.inverted = invert(self.reshaped)

    def from_source(self):
        """
        Read data from source
        """
        self.read_params()
        self.read_xrd()

        self.reshape()
        self.invert()

        self.convoluted = Preprocessing.convolve(self.inverted)

        return self

    @property
    def avg_spectra(self):
        #if hasattr(self,'__all_spectra'):
        #    return self.__all_spectra
        #else:
        #    self.__all_spectra = self.inverted.sum(axis = 0).sum(axis = 0) / (self.shape[0] * self.shape[1])
        #    return self.__all_spectra
        return self.inverted.sum(axis = 0).sum(axis = 0) / (self.shape[0] * self.shape[1])

    @property
    def integrated_spectra(self):
        return self.inverted.sum(axis = 2)

    @property
    def normalized_spectra(self):
        spectra = self.integrated_spectra
        return (spectra / spectra.max() * 255).astype(uint8)

    def crop_spectra(self,left,right):
        """
        FIXME set bounds maybe.
        """
        left,right = int(left),int(right)

        crop = self.inverted[:,:,left:right]
        crop = crop.sum(axis=2)

        if crop.max() == 0:
            return crop
        else:
            return (crop / crop.max() * 255).astype(uint8)

    def save_h5(self,name = None):

        if name == None:
            name = self.path + '/' + 'data.h5'

        print('Saving:',name)
        with h5py.File(name,'w') as f:
            f.create_dataset('inverted',data = self.inverted)
            f.create_dataset('convoluted',data = self.convoluted)

        return self

    def load_h5(self,name = None):

        if name == None:
            name = self.path + '/' + 'data.h5'

        print('Loading:',name)

        with h5py.File(name,'r') as f:

            print('Load inverted')
            x = f['inverted']
            self.inverted = x[:]
            self.shape = self.inverted.shape

            if 'convoluted' in f:
                print('Load convoluted')
                x = f['convoluted']
                self.convoluted = x[:]
            else:
                print('Preprocess convoluted')
                self.convoluted = Preprocessing.convolve(self.inverted)

        return self

    def shift_z(self,channel = 555):

        off = 24 
        win = signal.windows.gaussian(off*2,sqrt(8.1))

        b = []
        for x,_x in enumerate(self.inverted):
            a = []
            for y,_y in enumerate(_x):
                select = _y[channel - off : channel + off].copy()

                y = pad(select,(off,off),'edge')
                filtered = signal.convolve(y, win, mode='valid') / sum(win)
                filtered = filtered[:-1]

                f = filtered.argmax() - off

                if f > -off and f < off:
                    a += [roll(_y,-f)]
                else:
                    a += [_y]

            b += [array(a)]

        self.inverted = array(b)

class Preprocessing():

    def convolve(data,off = 48):
        """
        FIXME

        The gaussian convolution is only good for gaussian peaks i.e. low noise/signal ration
        """

        def select(d,off):
            c1 = []
            for i in range(d.shape[0]):
                c2 = []
                for j in range(d.shape[1]):
                    k = kurtosis(d[i,j])
                    if k < 2:
                        sigma = sqrt(d[i,j].std())
                        _w = signal.windows.gaussian(off * 2 - 1 ,sigma)
                    else:
                        _w = signal.windows.exponential(off * 2 - 1 ,tau = 1)
                    c2 += [_w]
                c1 += [array(c2)]

            return array(c1)

        win = signal.windows.gaussian(off * 2 - 1 ,3) 
        pad_data = pad(data,((0,0),(0,0),(off,off)),'edge')

        f = fft.rfft(pad_data)
        w = fft.rfft(win,pad_data.shape[-1])
        x = fft.irfft(f * w)

        x = x[:,:,off*2-1:-1]
        x = x / sum(win)

        for i in range(2):
            d = data - x
            win = select(d,off)
   
            w = fft.rfft(win,pad_data.shape[-1])
            x = fft.irfft(f * w)

            x = x[:,:,off*2-1:-1]
            sum_win = expand_dims(win.sum(axis=2),2)
            x = x / sum_win

        return x

    def shift_y(data,n):

        def pad_left(x,n):
            y_odd = pad(x,((0,0),(n,0),(0,0)),'edge')
            y_even = pad(x,((0,0),(0,n),(0,0)),'edge')
            y_odd[::2,:,:] = y_even[::2,:,:]
            return y_odd

        def pad_right(x,n):
            y_odd = pad(x,((0,0),(-n,0),(0,0)),'edge')
            y_even = pad(x,((0,0),(0,-n),(0,0)),'edge')
            y_odd[1::2,:,:] = y_even[1::2,:,:]
            return y_odd

        if n == 0:
            return data
        elif n > 0:
            return pad_left(data,n)
        elif n < 0:
            return pad_right(data,n)

    def shift_z(data,off=24,channel=555):
        
        b = []
        for i,x in enumerate(data):
            a = []
            for j,y in enumerate(x):
                select = y[channel - off : channel + off]
                f = select.argmax() - off

                if f > -off/2 and f < off/2:
                    a += [-f]
                else:
                    a += [0]
            b += [asarray(a)]

        return asarray(b)

    def apply_shift_z(data,shift):
        u = []
        for _y,_x in zip(data,shift):
            v = []
            for y,x in zip(_y,_x):
                v += [roll(y,x)]
            u += [asarray(v)]
        return asarray(u)

