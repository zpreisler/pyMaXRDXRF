from src.xrd_data import DataXRD,Preprocessing
from src.roi import MyROI
from src.viewbox import MyGLW,MyViewBox,MySpectraViewBox

from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt
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

        self.image_plot.addItem(self.img)

        self.image = self.data.normalized_spectra
        self.img.setImage(self.image)

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

        self.intensity_plot.setXRange(0,1280,padding=0)
        self.intensity_plot.setLabel('bottom',text='Channel')
        self.intensity_plot.setLabel('left',text='Total count per pixel')
        self.intensity_plot.setTitle('Full intensity')

        self.intensity_plot.plot(self.data.calibration.cx,self.data.avg_spectra,pen=fn.mkPen((255,166,166), width=1.666))

    def setSpectraPlot(self):
        """
        Spectra 2
        """
        self.layout.nextRow()
        self.spectra_plot = self.layout.addSpectraPlot(name='roi',viewBox = MySpectraViewBox(name='spectra_plot',title='ROI Spectras'))

        self.spectra_plot.snip_m = 24
        self.spectra_plot.normalized_roi = False
        self.spectra_plot.calibration = False
        self.spectra_plot.subtract_snip = False

        #self.spectra_plot.setLabel('bottom',text='Angle')
        self.spectra_plot.setLabel('bottom',text='Channel')
        self.spectra_plot.setLabel('left',text='Count per pixel')
        self.spectra_plot.setTitle('ROI spectras')

    def __init__(self,data):
        super().__init__()

        self.keyPressed.connect(self.onKey)
        self.data = data

        self.data._inverted = self.data.inverted
        self.data._convoluted = self.data.convoluted

        self.mode = 0 
        self.selected = None
        self.speed_cycle = cycle([1,2,8,16]) 
        self.speed = 2
        self.shift = 0

        self.data.inverted  = Preprocessing.shift_y(self.data._inverted,self.shift)
        self.data.convoluted  = Preprocessing.shift_y(self.data._convoluted,self.shift)

        setConfigOptions(background='w',antialias=True,leftButtonPan=False,imageAxisOrder='row-major')

        self.resize(800,800)
        self.setLayout()

        self.setImagePlot()
        self.setIntensityPlot()
        self.setSpectraPlot()

        self.image_plot.vb.spectra_plot = self.spectra_plot
        self.image_plot.vb.intensity_plot = self.intensity_plot
        self.image_plot.vb.main = self

        self.spectra_plot.setXLink(self.intensity_plot)

        """
        Init ROI
        """

        roi = MyROI([32,32],[12,12],translateSnap = True, scaleSnap = True,maxBounds = QtCore.QRectF(0,0,self.image.shape[1],self.image.shape[0]))

        roi.image_plot = self.image_plot
        roi.spectra_plot = self.spectra_plot
        roi.intensity_plot = self.intensity_plot
        roi.data = self.data
        roi.img = self.img

        self.image_plot.addItem(roi)
        self.image_plot.roi_list += [roi]

        roi.roi_update()
        roi.sigRegionChanged.connect(roi.roi_update)

        self.show()

        """
        Regions
        """
        self.mono_region = LinearRegionItem(brush = [222,222,222,122], hoverBrush = [222,222,222,168])
        self.mono_region.setZValue(10)
        self.mono_region.setRegion([718,730])
        self.intensity_plot.addItem(self.mono_region)
        self.mono_region.hide()

        self.mono_region.sigRegionChanged.connect(self.mono_update)

        self.rgb_region = []
        colors = [[232,0,0],[0,255,0],[151,203,255]]
        pos = [445,720,1134]

        for i,x in enumerate(colors):
            region = LinearRegionItem(brush = x + [112], hoverBrush = x + [162]) 
            region.setZValue(10)
            region.setRegion((pos[i],pos[i]+12))
            region.hide()

            region.sigRegionChanged.connect(self.rgbUpdate)

            self.rgb_region += [region]
            self.intensity_plot.addItem(region)

    def redrawROI(self):
        for roi in self.image_plot.roi_list:
            roi.calculate()
        roi.redraw()

    def mono_update(self):
        """
        Mono update
        """
        left,right = asarray(self.mono_region.getRegion()).astype(int)

        integrated = self.data.inverted[:,:,left:right].sum(axis=2)
        image = (integrated / integrated.max() * 255).astype(uint8)

        self.data.image = image 
        self.img.setImage(image)

    def rgbUpdate(self):
        """
        RGB update
        """
        image = []
        for region in self.rgb_region:
            left,right = asarray(region.getRegion()).astype(int)

            integrated = self.data.inverted[:,:,left:right].sum(axis=2)
            image += [(integrated / integrated.max() * 255).astype(uint8)]

        rgb_image = stack(image,-1).astype(uint8)

        self.data.image = rgb_image 
        self.img.setImage(rgb_image)

    def intensityUpdate(self):

        self.image = self.data.normalized_spectra
        self.data.image = self.image 
        self.img.setImage(self.image)

    def keyPressEvent(self,event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event)

    def onKey(self,event):
        """
        Keyboard inputs
        """
        #if event.key() == QtCore.Qt.Key.Key_U:
        #    self.shift += 1
        #    print('Shift:',self.shift)
        #    self.data.inverted  = Preprocessing.shift_y(self.data.inverted_org,self.shift)
        #    self.data.convoluted  = Preprocessing.shift_y(self.data.convoluted_org,self.shift)

        #    self.intensityUpdate()

        #if event.key() == QtCore.Qt.Key.Key_I:
        #    self.shift -= 1
        #    print('Shift:',self.shift)
        #    self.data.inverted  = Preprocessing.shift_y(self.data.inverted_org,self.shift)
        #    self.data.convoluted  = Preprocessing.shift_y(self.data.convoluted_org,self.shift)

        #    self.intensityUpdate()

        if event.key() == QtCore.Qt.Key.Key_Right:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x += self.speed 
                self.selected.setRegion(x)

        if event.key() == QtCore.Qt.Key.Key_Left:
            if self.selected:
                x = asarray(self.selected.getRegion()).astype(int)
                x -= self.speed
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

        if event.key() == QtCore.Qt.Key.Key_P:
            for i,roi in enumerate(self.image_plot.roi_list):
                name = self.data.path + '/' + 'roi_%d.dat'%i
                print('Saving ROI spectras',name)
                savetxt(name,c_[roi.data.calibration.cx,roi.z],fmt='%0.3f %d')

                print('Saving ROI images',name)
                name = self.data.path + '/' + 'roi_%d.tiff'%i
                imsave(name,roi.data.image)

                name = self.data.path + '/' + 'roi_crop_%d.tiff'%i
                print('Saving ROI crop images',name)
                imsave(name,roi.crop())

        if event.key() == QtCore.Qt.Key.Key_X:
            self.speed = next(self.speed_cycle)
            print('Shift speed:',self.speed)

        if event.key() == QtCore.Qt.Key.Key_W:
            self.spectra_plot.snip_m += 1
            print('Snip:',self.spectra_plot.snip_m)
            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_E:
            self.spectra_plot.snip_m -= 1
            print('Snip:',self.spectra_plot.snip_m)
            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_S:
            if self.spectra_plot.subtract_snip == True:
                print('Subtract snip off')
                self.spectra_plot.subtract_snip = False
                #self.spectra_plot.setXRange(0,1280,padding=0)
            else:
                print('Subtract snip on')
                self.spectra_plot.subtract_snip = True
                #self.spectra_plot.setXRange(0,1280,padding=0)
                #self.spectra_plot.setYRange(0,1280,padding=0)

            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_C:
            if self.spectra_plot.calibration == True:
                print('Calibration off')
                self.spectra_plot.calibration = False
                self.spectra_plot.setXRange(0,1280,padding=0)

            else:
                print('Calibration on')
                self.spectra_plot.calibration = True
                self.spectra_plot.setXRange(self.data.calibration.cx[0],self.data.calibration.cx[-1],padding=0)

            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_N:

            if self.spectra_plot.normalized_roi == True:
                self.spectra_plot.normalized_roi = False
                print('ROI spectra not normalized')
            else:
                self.spectra_plot.normalized_roi = True
                print('ROI spectra normalized to 1000')

            self.redrawROI()

        if event.key() == QtCore.Qt.Key.Key_M or event.key() == QtCore.Qt.Key.Key_Space:
            self.mode = (self.mode + 1) % 3
            modes = ['Intesity','Mono','RGB']
            print('Mode selected:',self.mode,modes[self.mode])

            if self.mode == 1:
                self.selected = self.mono_region

                self.mono_region.show()

                for region in self.rgb_region:
                    region.hide()

                self.mono_update()

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

        if event.key() == QtCore.Qt.Key.Key_1:
            self.selected = self.rgb_region[0]
            print('selected: red')

        if event.key() == QtCore.Qt.Key.Key_2:
            self.selected = self.rgb_region[1]
            print('selected: green')

        if event.key() == QtCore.Qt.Key.Key_3:
            self.selected = self.rgb_region[2]
            print('selected: blue')
