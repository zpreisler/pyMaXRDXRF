#!/usr/bin/env python
from dataxrd import DataXRD 
from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack
from numpy.random import random,randint
from itertools import cycle

class MyGLW(GraphicsView):
    """
    Enhanced
    """
    def __init__(self,parent=None,**kwargs):
        super().__init__(parent)
        self.ci = GraphicsLayout(**kwargs)

        for n in ['nextRow', 'nextCol', 'nextColumn', 'addPlot', 'addViewBox', 'addItem', 'getItem', 'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear']:
            setattr(self, n, getattr(self.ci, n))
        self.setCentralItem(self.ci)

    def addFancyPlot(self,row=None,col=None,rowspan=1,colspan=1,**kwargs):
        plot = FancyPlotItem(**kwargs)
        self.addItem(plot,row,col,rowspan,colspan)

        return plot 

class FancyPlotItem(PlotItem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.vb.image_plot = self 
        self.roi_list = []

        self.setMenuEnabled(False)

class MyViewBox(ViewBox):
    """
    MyViewBox
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.setMouseMode(self.RectMode)
        self.setAspectLocked(True)
        self.setDefaultPadding(0.0)

        #ROI box
        self.roiBox = QtGui.QGraphicsRectItem(0, 0, 1, 1)
        self.roiBox.setPen(fn.mkPen((255,100,255), width=1))
        self.roiBox.setBrush(fn.mkBrush(255,0,255,100))
        self.roiBox.setZValue(1e9)
        self.roiBox.hide()
        self.addItem(self.roiBox, ignoreBounds=True)

    def hoverEvent(self,event):

        if event.isExit():
            self.image_plot.setTitle('')
            return 

        p = self.mapToView(event.pos())
        x,y = p.x(),p.y()

        self.image_plot.setTitle('x: %d  y: %d'%(x,y))

    def updateRoiBox(self,p1,p2):

        r = QtCore.QRectF(p1,p2)
        r = self.childGroup.mapRectFromParent(r)
        self.roiBox.setPos(r.topLeft())
        transform = QtGui.QTransform.fromScale(r.width(), r.height())
        self.roiBox.setTransform(transform)
        self.roiBox.show()

    def mouseDragEvent(self, event):

        event.accept()

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.isFinish():
                self.roiBox.hide()

                m = self.mapToView(event.buttonDownPos())
                n = self.mapToView(event.pos())

                x = [int(m.x()),int(n.x())]
                y = [int(m.y()),int(n.y())]

                x0 = min(x)
                y0 = min(y)

                w = abs(x[0] - x[1])
                h = abs(y[0] - y[1])

                roi = MyROI([x0,y0],[w,h])

                self.image_plot.addItem(roi)
                self.image_plot.roi_list += [roi]

                roi.image_plot = self.main.image_plot
                roi.spectra_plot = self.main.spectra_plot
                roi.data = self.main.data
                roi.img = self.main.img

                roi.roi_update()
                roi.sigRegionChanged.connect(roi.roi_update)

            else:
                self.updateRoiBox(event.buttonDownPos(), event.pos())



class MyROI(ROI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.addScaleHandle([1,0],[0,1])
        color = randint(0,256,3)
        self.pen = fn.mkPen(color, width=1.66)
        self.setPen(self.pen)

    def calc(self):

        y = self.getArraySlice(self.data.integrated_spectra,self.img)

        s1,s2 = y[0][0],y[0][1]
        z = self.data.inverted[s1,s2]
        res = z.shape[0] * z.shape[1]

        z = z.sum(axis=0).sum(axis=0)

        self.z = z / res

        return self.z

    def roi_update(self):
        self.calc()
        self.redraw()

    def redraw(self):
        self.spectra_plot.clear()
        for roi in self.image_plot.roi_list:
            self.spectra_plot.plot(roi.z,pen=roi.pen)

    def mouseClickEvent(self,event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self in self.image_plot.roi_list:
                self.image_plot.roi_list.remove(self)
                self.image_plot.removeItem(self)

            self.redraw()


class MainWindow(QtWidgets.QMainWindow):
    """
    Modified Main Window class
    """
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    def __init__(self,data):
        super().__init__()

        self.keyPressed.connect(self.onKey)
        self.data = data

        self.mode = 0 
        self.speed_cycle = cycle([1,2,8,16]) 
        self.speed = 2
        self.selected = None

        setConfigOptions(background='w',antialias=True,leftButtonPan=False,imageAxisOrder='row-major')

        self.resize(800,800)

        self.layout = MyGLW(border=True)
        self.setCentralWidget(self.layout)

        """
        Image
        """

        self.image_plot = self.layout.addFancyPlot(viewBox = MyViewBox(border=[255,255,255],name='image_plot'))
        self.layout.ci.layout.setRowStretchFactor(0,3)
        self.layout.ci.layout.setRowStretchFactor(1,2)
        self.layout.ci.layout.setRowStretchFactor(2,2)
        self.layout.ci.layout.setVerticalSpacing(24)

        self.img = ImageItem()

        self.image_plot.addItem(self.img)
        self.image_plot.setTitle('Image')

        self.image = (self.data.integrated_spectra / self.data.integrated_spectra.max() * 255).astype(uint8)
        self.img.setImage(self.image)

        self.image_plot.setYRange(0,self.image.shape[0])
        self.image_plot.setXRange(0,self.image.shape[1])
        self.image_plot.hideAxis('bottom')
        self.image_plot.hideAxis('left')

        """
        Spectra 1
        """
        self.layout.nextRow()
        self.intensity_plot = self.layout.addPlot(enableMenu=False)

        self.intensity_plot.setXRange(0,1280,padding=0)
        self.intensity_plot.setLabel('bottom',text='Channel')
        self.intensity_plot.setLabel('left',text='Total count per pixel')
        self.intensity_plot.setTitle('Full intensity')


        z = self.data.inverted
        res = z.shape[0] * z.shape[1]
        z = z.sum(axis=0).sum(axis=0) / res

        self.intensity_plot.plot(z,pen=fn.mkPen((255,166,166), width=1.666))

        """
        Spectra 2
        """
        self.layout.nextRow()

        self.spectra_plot = self.layout.addPlot(enableMenu=False)
        self.image_plot.vb.spectra_plot = self.spectra_plot
        self.image_plot.vb.main = self

        self.spectra_plot.setXRange(0,1280,padding=0)

        self.spectra_plot.setLabel('bottom',text='Channel')
        self.spectra_plot.setLabel('left',text='Count per pixel')
        self.spectra_plot.setTitle('ROI spectras')

        """
        Init ROI
        """
        roi = MyROI([32,32],[12,12])

        roi.image_plot = self.image_plot
        roi.spectra_plot = self.spectra_plot
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

    def mono_update(self):
        """
        Mono update
        """
        left,right = asarray(self.mono_region.getRegion()).astype(int)

        integrated = self.data.inverted[:,:,left:right].sum(axis=2)
        image = (integrated / integrated.max() * 255).astype(uint8)

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
        self.img.setImage(rgb_image)

    def intensityUpdate(self):
        self.img.setImage(self.image)

    def keyPressEvent(self,event):
        super().keyPressEvent(event)
        self.keyPressed.emit(event)

    def onKey(self,event):
        """
        Keyboard inputs
        """

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

        if event.key() == QtCore.Qt.Key.Key_X:
            self.speed = next(self.speed_cycle)
            print(self.speed)

        if event.key() == QtCore.Qt.Key.Key_M or event.key() == QtCore.Qt.Key.Key_Space:
            self.mode = (self.mode + 1) % 3
            print(self.mode)

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

def main():

    """
    Readind data
    """
    data = DataXRD('data_XRD2/').load_h5()
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
