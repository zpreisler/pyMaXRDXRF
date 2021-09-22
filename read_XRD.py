#!/usr/bin/env python
from dataxrd import DataXRD 
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

    args = parser.parse_args()
    kwargs = vars(args)

    print(args)
    print('Source data directory:',args.path)

    load = kwargs.pop('load')

    if load is False:
        d = DataXRD(**kwargs).from_source()
        d.save_h5()

    else:
        d = DataXRD(**kwargs).load_h5()

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
