from src.xrd_data import DataXRD,Preprocessing
from src.roi import MyROI
from src.viewbox import MyGLW,MyViewBox,MySpectraViewBox

from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout,ColorBarItem,HistogramLUTItem
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from matplotlib.pyplot import imsave

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt,argmin,round
from numpy.random import random,randint
from itertools import cycle

class MainWindow(QtWidgets.QMainWindow):
    """
    Modified Main Window class
    """
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    def setLayout(self):
        self.layout = MyGLW(border=True)
        self.setCentralWidget(self.layout)

        self.layout.ci.layout.setRowStretchFactor(0,3)
        self.layout.ci.layout.setRowStretchFactor(1,2)
        self.layout.ci.layout.setRowStretchFactor(2,2)
        self.layout.ci.layout.setVerticalSpacing(24)

    def setImagePlot(self):
        """
        Image
        """

        self.image_plot = self.layout.addFancyPlot(viewBox = MyViewBox(border=[255,255,255],name='image_plot'),enableMenu=False)
        self.img = ImageItem()

        #bar = ColorBarItem(values=(0,255))

        self.hist = HistogramLUTItem(levelMode='mono')
        self.hist_rgba = HistogramLUTItem(levelMode='rgba')

        self.hist.setImageItem(self.img)
        self.hist_rgba.setImageItem(self.img)

        self.image_plot.addItem(self.img)

        self.histogram_plot = self.layout.addItem(self.hist)
        self.histogram_rgba_plot = self.layout.addItem(self.hist_rgba)
        #self.histogram_plot.addItem(hist)

        self.image = self.data.normalized_spectra
        self.img.setImage(self.image)

        #bar.setImageItem(self.img,insert_in=self.image_plot)

        self.data.image = self.image

        self.image_plot.setYRange(0,self.image.shape[0])
        self.image_plot.setXRange(0,self.image.shape[1])
        self.image_plot.hideAxis('bottom')
        self.image_plot.hideAxis('left')
        self.image_plot.setTitle('Image')

    def setIntensityPlot(self):
        """
        Spectra 1
        """

        self.layout.nextRow()
        self.intensity_plot = self.layout.addSpectraPlot(name='intensity',viewBox = MySpectraViewBox(name='intensity_plot',title='Full Intensity'))
        self.intensity_plot.setDefaultPadding(padding=0)

        self.intensity_plot.setLabel('bottom',text='Channel')
        self.intensity_plot.setLabel('left',text='Total count per pixel')
        self.intensity_plot.setTitle('Full intensity')

        self.redrawIntensity()

    def redrawIntensity(self):
        self.intensity_plot.clearPlots()

        if self.calibration is True:
            #self.intensity_plot.plot(self.data.calibration.cx,self.data.avg_spectra,pen=fn.mkPen((255,166,166), width=1.666))
            self.intensity_plot.plot(self.data.calibration.cx,self.data.spectra255,pen=fn.mkPen((255,166,166), width=1.666))
            self.intensity_plot.setLabel('bottom',text='Angle')

        else:
            self.intensity_plot.plot(self.data.spectra255,pen=fn.mkPen((255,166,166), width=1.666))
            print(self.data.spectra255)
            self.intensity_plot.setLabel('bottom',text='Channel')

    def setSpectraPlot(self):
        """
        Spectra 2
        """
        self.layout.nextRow()
        self.spectra_plot = self.layout.addSpectraPlot(name='roi',viewBox = MySpectraViewBox(name='spectra_plot',title='ROI Spectras'))
        self.spectra_plot.setDefaultPadding(padding=0)

        self.spectra_plot.snip_m = 24
        self.spectra_plot.normalized_roi = False
        self.spectra_plot.calibration = False
        self.spectra_plot.subtract_snip = False

        #self.spectra_plot.setLabel('bottom',text='Angle')
        self.spectra_plot.setLabel('bottom',text='Channel')
        self.spectra_plot.setLabel('left',text='Count per pixel')
        self.spectra_plot.setTitle('ROI spectras')

    def setFirstRoi(self):
        """
        Init ROI
        """

        roi = MyROI([32,32],[12,12],translateSnap = True, scaleSnap = True,maxBounds = QtCore.QRectF(0,0,self.image.shape[1],self.image.shape[0]))

        roi.setMain(self)

        self.image_plot.addItem(roi)
        self.image_plot.roi_list += [roi]

        roi.roiUpdate()
        roi.sigRegionChanged.connect(roi.roiUpdate)

    def setMonoRegion(self):
        """
        Mono Region
        """
        self.mono_region = LinearRegionItem(brush = [222,222,222,122], hoverBrush = [222,222,222,168], bounds = (0,1279))
        self.mono_region.setZValue(10)
        self.mono_region.setRegion([521,575])

        self.intensity_plot.addItem(self.mono_region)
        self.mono_region.hide()

        self.mono_region.sigRegionChanged.connect(self.monoUpdate)

    def setRGBRegion(self):
        """
        Mono Region
        """

        self.rgb_region = []
        colors = [[232,0,0],[0,255,0],[151,203,255]]
        pos = [445,720,1134]

        for i,x in enumerate(colors):
            region = LinearRegionItem(brush = x + [112], hoverBrush = x + [162], bounds = (0,1279)) 
            region.setZValue(10)
            region.setRegion((pos[i],pos[i]+12))
            region.hide()

            self.rgb_region += [region]
            self.intensity_plot.addItem(region)

            region.sigRegionChanged.connect(self.rgbUpdate)

    def __init__(self,data):
        super().__init__()

        self.keyPressed.connect(self.onKey)
        self.data = data

        self.data._inverted = self.data.inverted
        self.data._convoluted = self.data.convoluted

        self.mode = 0 
        self.selected = None
        self.speed_cycle = cycle([1,2,8,16]) 
        self.speed = 8
        self.shift = 0

        self.calibration = False

        self.data.inverted  = Preprocessing.shift_y(self.data._inverted,self.shift)
        self.data.convoluted  = Preprocessing.shift_y(self.data._convoluted,self.shift)

        setConfigOptions(background='w',antialias=True,leftButtonPan=False,imageAxisOrder='row-major')

        self.resize(900,900)
        self.setLayout()

        self.setImagePlot()
        self.setIntensityPlot()
        self.setSpectraPlot()

        self.setFirstRoi()

        self.image_plot.vb.spectra_plot = self.spectra_plot
        self.image_plot.vb.intensity_plot = self.intensity_plot
        self.image_plot.vb.main = self

        self.spectra_plot.setXLink(self.intensity_plot)

        self.setMonoRegion()
        self.setRGBRegion()

        self.show()

    def redrawROI(self):
        if self.image_plot.roi_list:
            for roi in self.image_plot.roi_list:
                roi.calculate()
            roi.redraw()

    def monoUpdate(self):
        """
        Mono update
        """
        x = asarray(self.mono_region.getRegion()).astype(float)

        if self.calibration:
            x  = self.data.calibration.ic(x)

        self.data.image = self.data.crop_spectra(*x)
        if self.mode == 1:
            self.img.setImage(self.data.image)

    def rgbUpdate(self):
        """
        RGB update
        """
        image = []
        for region in self.rgb_region:
            x = asarray(region.getRegion()).astype(float)
            if self.calibration is True:
                x  = self.data.calibration.ic(x)

            if self.spectra_plot.subtract_snip == True:
                image += [self.data.crop_snip_spectra(*x)]
            else:
                image += [self.data.crop_spectra(*x)]

        rgb_image = stack(image,-1)
        rgb_image = rgb_image.astype(uint8)

        self.data.image = rgb_image 

        if self.mode == 2:
            self.img.setImage(rgb_image)

    def intensityUpdate(self):

        self.image = self.data.normalized_spectra

        self.data.image = self.image 
        self.img.setImage(self.image)

    def keyPressEvent(self,event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event)

    def moveRegion(self,event):

        if event.key() == QtCore.Qt.Key.Key_X:
            self.speed = next(self.speed_cycle)
            print('Shift speed:',self.speed)

        if event.key() == QtCore.Qt.Key.Key_Left:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x -= self.speed
                self.selected.setRegion(x)

        if event.key() == QtCore.Qt.Key.Key_Right:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x += self.speed 
                self.selected.setRegion(x)

        if event.key() == QtCore.Qt.Key.Key_Up:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x[1] += self.speed
                self.selected.setRegion(x)

        if event.key() == QtCore.Qt.Key.Key_Down:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x[1] -= self.speed
                self.selected.setRegion(x)

    def printRoi(self,event):
        if event.key() == QtCore.Qt.Key.Key_P:
            for i,roi in enumerate(self.image_plot.roi_list):
                name = self.data.path + '/' + 'roi_%d.dat'%i
                print('Saving ROI spectras',name)
                savetxt(name,c_[roi.data.calibration.cx,roi.z],fmt='%0.3f %0.3f')

                name = self.data.path + '/' + 'roi_%d_raw.dat'%i
                print('Saving raw ROI spectras',name)
                savetxt(name,roi.z,fmt='%0.3f')

                print('Saving ROI images',name)
                name = self.data.path + '/' + 'roi_%d.tiff'%i
                imsave(name,roi.data.image[::-1])

                name = self.data.path + '/' + 'roi_crop_%d.tiff'%i
                print('Saving ROI crop images',name)
                imsave(name,roi.crop())

    def adjustSnip(self,event):
        if event.key() == QtCore.Qt.Key.Key_W:
            self.spectra_plot.snip_m += 1
            print('Snip:',self.spectra_plot.snip_m)
            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_E:
            self.spectra_plot.snip_m -= 1
            print('Snip:',self.spectra_plot.snip_m)
            self.redrawROI()

    def switchCalibration(self,event):

        if event.key() == QtCore.Qt.Key.Key_C:
            if self.calibration == True:
                print('Calibration off')
                self.calibration = False

                regions =  [self.mono_region] + self.rgb_region
                for region in regions:

                    x = asarray(region.getRegion()).astype(float)
                    x = self.data.calibration.ic(x)

                    region.setRegion(x)

            else:
                print('Calibration on')
                self.calibration = True

                regions = [self.mono_region] + self.rgb_region 
                for region in regions:

                    x = asarray(region.getRegion()).astype(float)
                    x = self.data.calibration.c(x)

                    region.setRegion(x)

            self.redrawIntensity()
            self.redrawROI()

    def normalizeSpectra(self,event):
        if event.key() == QtCore.Qt.Key.Key_N:

            if self.spectra_plot.normalized_roi == True:
                self.spectra_plot.normalized_roi = False
                print('Roi spectra not normalized')
            else:
                self.spectra_plot.normalized_roi = True
                print('ROI spectra normalized to 1000')

            self.redrawROI()

    def switchModes(self,event):
        if event.key() == QtCore.Qt.Key.Key_M or event.key() == QtCore.Qt.Key.Key_Space:
            self.mode = (self.mode + 1) % 3
            modes = ['Intesity','Mono','RGB']
            print('Mode selected:',self.mode,modes[self.mode])

            if self.mode == 1:
                self.selected = self.mono_region

                self.mono_region.show()

                for region in self.rgb_region:
                    region.hide()

                self.monoUpdate()

            elif self.mode == 2:
                self.selected = self.rgb_region[0]
                self.mono_region.hide()
                for region in self.rgb_region:
                    region.show()

                self.rgbUpdate()

            else:
                self.selected = None
                self.mono_region.hide()
                for region in self.rgb_region:
                    region.hide()

                self.intensityUpdate()

    def switchRegion(self,event):

        if event.key() == QtCore.Qt.Key.Key_1:
            self.selected = self.rgb_region[0]
            print('selected: red')

        if event.key() == QtCore.Qt.Key.Key_2:
            self.selected = self.rgb_region[1]
            print('selected: green')

        if event.key() == QtCore.Qt.Key.Key_3:
            self.selected = self.rgb_region[2]
            print('selected: blue')

    def applySnip(self,event):
        if event.key() == QtCore.Qt.Key.Key_S:
            if self.spectra_plot.subtract_snip == True:
                print('Subtract snip off')
                self.spectra_plot.subtract_snip = False
            else:
                print('Subtract snip on')
                self.spectra_plot.subtract_snip = True

            self.redrawROI()

    def shiftY(self,event):
        if event.key() == QtCore.Qt.Key.Key_U:
            self.shift += 1
            print('Shift:',self.shift)
            self.data.inverted  = Preprocessing.shift_y(self.data._inverted,self.shift)
            self.data.convoluted  = Preprocessing.shift_y(self.data._convoluted,self.shift)

            self.intensityUpdate()

        if event.key() == QtCore.Qt.Key.Key_I:
            self.shift -= 1
            print('Shift:',self.shift)
            self.data.inverted  = Preprocessing.shift_y(self.data._inverted,self.shift)
            self.data.convoluted  = Preprocessing.shift_y(self.data._convoluted,self.shift)

            self.intensityUpdate()
    
    def setLogPlots(self,event):

        if event.key() == QtCore.Qt.Key.Key_L:

            if self.spectra_plot.ctrl.logYCheck.isChecked():

                self.spectra_plot.setLogMode(False,False)
                self.intensity_plot.setLogMode(False,False)

            else:

                self.spectra_plot.setLogMode(False,True)
                self.intensity_plot.setLogMode(False,True)

    def onKey(self,event):
        """
        Keyboard inputs
        """

        self.moveRegion(event)
        self.switchCalibration(event)
        self.normalizeSpectra(event)
        self.switchModes(event)
        self.switchRegion(event)
        self.adjustSnip(event)
        self.applySnip(event)
        self.printRoi(event)
        self.shiftY(event)
        self.setLogPlots(event)

