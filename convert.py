#!/usr/bin/env python
from src.xrd_data import DataXRD,Preprocessing
from src.roi import MyROI
from src.mainwindow import MainWindow

from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt
from numpy.random import random,randint
from itertools import cycle

from pathlib import Path
from scipy import signal
import os

from matplotlib.image import imsave

from argparse import ArgumentParser
import h5py

def main():
    """
    ArgParse
    """
    parser = ArgumentParser()

    parser.add_argument('path')
    parser.add_argument('--parameters',default='Scanning_Parameters.txt',help='scanning parameters file')
    parser.add_argument('-c','--calibration',default='calibration.ini',help='calibration file')
    parser.add_argument('-s','--shift-y',default=0,help='shift correction',type=int)
    parser.add_argument('-l','--load',action='store_true')
    parser.add_argument('-z','--shift-z',default = 0,type=int)
    parser.add_argument('--asci',action='store_true')

    args = parser.parse_args()
    kwargs = vars(args)

    print(args)
    print('Source data directory:',args.path)
    """
    Reading data
    """

    load = kwargs.pop('load')
    shift_y = kwargs.pop('shift_y')
    shift_z = kwargs.pop('shift_z')
    #save_h5 = kwargs.pop('h5')
    save_asci = kwargs.pop('asci')

    if load is False:
        data = DataXRD(**kwargs).from_source()
        data.save_h5()

        if shift_z != 0:
            shift = Preprocessing.shift_z(data.convoluted,channel = shift_z)
            data.convoluted = Preprocessing.apply_shift_z(data.convoluted,shift)
            data.inverted = Preprocessing.apply_shift_z(data.inverted,shift)

    else:
        data = DataXRD(**kwargs).load_h5()

        if shift_z != 0:
            shift = Preprocessing.shift_z(data.convoluted,channel = shift_z)
            data.convoluted = Preprocessing.apply_shift_z(data.convoluted,shift)
            data.inverted = Preprocessing.apply_shift_z(data.inverted,shift)

    data.calibrate(n_channels=data.shape[-1])

    print(data.inverted.shape)
    print(data.calibration.cx,len(data.calibration.cx))

    tmp_data = data.inverted.reshape(-1,1280)
    
    try:
        os.mkdir('converted')
    except:
        pass

    #print(tmp_data.shape,data.inverted.shape)
    #print(len(tmp_data))

    #if save_h5:
    print('Saving h5')
    tmp_cx = array([data.calibration.cx] * len(tmp_data))
    cx = tmp_cx.reshape(*data.inverted.shape)

    with h5py.File('converted/cdata.h5','w') as f:

        f.create_dataset('flat_inverted',data = tmp_data)
        f.create_dataset('flat_calibration_inverted',data = tmp_cx)

        f.create_dataset('inverted',data = data.inverted)
        f.create_dataset('calibration_inverted',data = cx)

    if save_asci:
        print('Saving ASCI')
        for i,t in enumerate(tmp_data):
            savetxt('converted/CFrame%04d.dat'%i,c_[data.calibration.cx,t],fmt='%.4f %d')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
