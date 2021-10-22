#from src.xrd_data import DataXRD 
from src.roi import MyROI
#from src.mainwindow import MainWindow

from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt
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

    def addSpectraPlot(self,row=None,col=None,rowspan=1,colspan=1,**kwargs):
        plot = SpectraItem(**kwargs)
        self.addItem(plot,row,col,rowspan,colspan)

        return plot 

class FancyPlotItem(PlotItem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.vb.image_plot = self 
        self.roi_list = []

        self.setMenuEnabled(False)

class SpectraItem(PlotItem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.vb.spectra_plot = self 

        self.setMenuEnabled(False)

class MySpectraViewBox(ViewBox):
    def __init__(self,*args,**kwargs):

        self.title = kwargs.pop('title','')

        super().__init__(*args,**kwargs)

        self.setDefaultPadding(0.0)
        self.setMouseMode(self.RectMode)


    def hoverEvent(self,event):

        if event.isExit():
            self.spectra_plot.setTitle(self.title)
            return 

        p = self.mapToView(event.pos())
        x,y = p.x(),p.y()

        self.spectra_plot.setTitle('x: %d  y: %d'%(x,y))

    def mouseClickEvent(self, event):
        p = self.mapToView(event.pos())
        print('%d'%(p.x()))

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
            self.image_plot.setTitle('Image')
            return 

        p = self.mapToView(event.pos())
        x,y = p.x(),p.y()

        self.image_plot.setTitle('x: %d  y: %d'%(x,y))

    def mouseClickEvent(self, event):

        p = self.mapToView(event.pos())
        x,y = p.x(),p.y()
        print('x: %d y: %d'%(x,y))

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

                w = abs(x[0] - x[1]) + 1
                h = abs(y[0] - y[1]) + 1

                roi = MyROI([x0,y0],[w,h],translateSnap = True,scaleSnap = True, maxBounds = QtCore.QRectF(0,0,self.main.data.inverted.shape[1],self.main.data.inverted.shape[0]))
                
                print('new roi:',[x0,y0],[w,h])

                self.image_plot.addItem(roi)
                self.image_plot.roi_list += [roi]

                roi.setMain(self.main)

                roi.roiUpdate()
                roi.sigRegionChanged.connect(roi.roiUpdate)

            else:
                self.updateRoiBox(event.buttonDownPos(), event.pos())
