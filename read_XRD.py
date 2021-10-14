#!/usr/bin/env python
from src.xrd_data import DataXRD,Calibration
from glob import glob
from matplotlib.pyplot import show,imshow,plot,figure,xlim

from argparse import ArgumentParser

def main():
    """
    Read source data and save data.h5 file
    """
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--parameters',default='Scanning_Parameters.txt',help='scanning parameters file')
    parser.add_argument('-l','--load',action='store_true')
    parser.add_argument('-n','--name',default=None)

    args = parser.parse_args()
    kwargs = vars(args)

    print(args)
    print('Source data directory:',args.path)

    load = kwargs.pop('load')
    name = kwargs.pop('name')
    print('name:',name)

    if load is False:
        d = DataXRD(**kwargs).from_source()
        d.save_h5(name=name)

    else:
        d = DataXRD(**kwargs).load_h5()

    name = args.path + '/' + 'Calibration.ini'
    calibration = Calibration(name)
    calibration.calibrate_channels()

    figure()
    imshow(d.inverted[:,:,725])

    figure()
    imshow(d.integrated_spectra)

    figure()
    plot(d.all_spectra)
    xlim(0,1280)

    show()

if __name__ == "__main__":
    main()
