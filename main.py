#!/usr/bin/env python
from src.xrd_data import DataXRD 
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

from scipy import signal

from matplotlib.image import imsave

from argparse import ArgumentParser

def main():
    """
    ArgParse
    """
    parser = ArgumentParser()

    parser.add_argument('path')
    parser.add_argument('--parameters',default='Scanning_Parameters.txt',help='scanning parameters file')
    parser.add_argument('-c','--calibration',default='calibration.ini',help='calibration file')
    parser.add_argument('-s','--shift',default=0,help='shift correction',type=int)
    parser.add_argument('-l','--load',action='store_true')
    parser.add_argument('-z','--shift-z',default = 555,type=int)

    args = parser.parse_args()
    kwargs = vars(args)

    print(args)
    print('Source data directory:',args.path)
    """
    Reading data
    """

    load = kwargs.pop('load')
    shift = kwargs.pop('shift')
    shift_z = kwargs.pop('shift_z')

    if load is False:
        data = DataXRD(**kwargs).from_source()
        data.save_h5()
        data.shift(shift)

        if shift_z != 0:
            data.shift_z()

    else:
        data = DataXRD(**kwargs).load_h5()
        data.shift(shift)

        if shift_z != 0:
            data.shift_z()

    data.calibrate()
    #data.read_calibration_file()
    #data.calibrate_channels()

    print(data)
    """
    Open window
    """
    app = mkQApp()
    window = MainWindow(data)

    exec_()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
