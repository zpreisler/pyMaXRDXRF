# pyMaXRDXRF

usage:

```
python main.py data_XRD
```

where data_XRD is a folder with the source data.

after first use you can load '.h5' file with  `-l` option 

`-s $n` option can be used to set shift at start of the program

`-z $n` option can be used to set z-shift at start of the program

```
python main.py data_XRD -l -s 2
```

Scanning parameters `'Scanning_parameters.txt'` and calibration file `'calibration.ini'` are by default located in the data folder.

Keyboard:

You can print ROIS by pressing `'p'`

Press `'m'` or `space` to change mode

Press `'n'` to change ROI normalization

Press `'c'` to turn on/off calibration

pres `'1'`, `'2'` or `'3'` to change the selection region in the RGB mode.

![Snapshot](doc/snapshot.png)
