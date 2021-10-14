from src.xrd_data import DataXRD 
from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt
from numpy.random import random,randint

class MyROI(ROI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.addScaleHandle([1,0],[0,1])
        color = randint(0,256,3)
        self.pen = fn.mkPen(color, width=1.66)
        self.snip_pen = fn.mkPen(color, width=.66)
        self.conv_pen = fn.mkPen((0,0,0), width=1)
        self.setPen(self.pen)

    def calc(self):

        y = self.getArraySlice(self.data.integrated_spectra,self.img)

        s1,s2 = y[0][0],y[0][1]
        z = self.data.inverted[s1,s2]
        print('[x:%d y:%d]'%(s2.start,s1.stop),'[x:%d y:%d]'%(s2.stop,s1.start))


        res = 1 / (z.shape[0] * z.shape[1])

        z = z.sum(axis=0).sum(axis=0)

        conv = self.data.conv[s1,s2]
        self.conv = conv.sum(axis=0).sum(axis=0) * res

        if self.spectra_plot.normalized_roi == True:
            res = 1000.0 / z.max()

        self.z = z * res

        self.snip_z = self.snip()

        return self.z

    def crop(self):
        y = self.getArraySlice(self.data.image,self.img)
        s1,s2 = y[0][0],y[0][1]
        z = self.data.image[s1,s2]

        return z[::-1]

    def roi_update(self):
        self.calc()
        self.redraw()

    def snip(self):
        self.conv_z = self.conv.copy()
        x = self.conv.copy()

        for p in range(1,self.spectra_plot.snip_m)[::-1]:
            a1 = x[p:-p]
            a2 = (x[:(-2 * p)] + x[(2 * p):]) * 0.5
            x[p:-p] = minimum(a2,a1)

        return x

    def redraw(self):
        self.spectra_plot.clear()
        for roi in self.image_plot.roi_list:
            #self.spectra_plot.plot(roi.z,pen=roi.pen)
            if self.spectra_plot.calibration == True:
                self.spectra_plot.plot(roi.data.cx,roi.z,pen=roi.pen)
            elif self.spectra_plot.subtract_snip == True:
                self.spectra_plot.plot(roi.z - roi.snip_z,pen=roi.pen)
            else:
                self.spectra_plot.plot(roi.z,pen=roi.pen)
                self.spectra_plot.plot(roi.snip_z,pen=roi.snip_pen)
                self.spectra_plot.plot(roi.conv_z,pen=roi.conv_pen)

        if self.spectra_plot.calibration == True:
            self.spectra_plot.setLabel('bottom',text='Angle')
        else:
            self.spectra_plot.setLabel('bottom',text='Channel')

    def mouseClickEvent(self,event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self in self.image_plot.roi_list:
                self.image_plot.roi_list.remove(self)
                self.image_plot.removeItem(self)

            self.redraw()
    
        else:
            p = self.mapToView(event.pos())
            x,y = p.x(),p.y()
            print('x: %d y: %d'%(x,y))
