from src.xrd_data import DataXRD 
from pyqtgraph import exec as exec_
from pyqtgraph import functions as fn
from pyqtgraph.Point import Point
from pyqtgraph import mkQApp,GraphicsLayoutWidget,setConfigOptions
from pyqtgraph import GraphicsView,ViewBox,Point,PlotItem,ImageItem,AxisItem,ROI,LinearRegionItem,GraphicsLayout
from pyqtgraph.Qt import QtCore,QtWidgets,QtGui

from numpy import uint8,array,asarray,stack,savetxt,c_,pad,where,minimum,sqrt
from numpy.random import random,randint

class MouseDragHandler(object):

    def __init__(self, roi):
        self.roi = roi
        self.dragMode = None
        self.startState = None
        self.snapModifier = QtCore.Qt.KeyboardModifier.ControlModifier
        self.translateModifier = QtCore.Qt.KeyboardModifier.NoModifier
        self.rotateModifier = QtCore.Qt.KeyboardModifier.AltModifier
        self.scaleModifier = QtCore.Qt.KeyboardModifier.ShiftModifier
        self.rotateSpeed = 0.5
        self.scaleSpeed = 1.01

    def mouseDragEvent(self, ev):
        roi = self.roi

        if ev.isStart():
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                roi.setSelected(True)
                mods = ev.modifiers() & ~self.snapModifier
                if roi.translatable and mods == self.translateModifier:
                    self.dragMode = 'translate'
                elif roi.rotatable and mods == self.rotateModifier:
                    self.dragMode = 'rotate'
                elif roi.resizable and mods == self.scaleModifier:
                    self.dragMode = 'scale'
                else:
                    self.dragMode = None

                if self.dragMode is not None:
                    roi._moveStarted()
                    self.startPos = roi.mapToParent(ev.buttonDownPos())
                    self.startState = roi.saveState()
                    self.cursorOffset = roi.pos() - self.startPos
                    ev.accept()
                else:
                    ev.ignore()
            else:
                self.dragMode = None
                ev.ignore()


        if ev.isFinish() and self.dragMode is not None:
            roi._moveFinished()
            return

        # roi.isMoving becomes False if the move was cancelled by right-click
        if not roi.isMoving or self.dragMode is None:
            return

        snap = True if (ev.modifiers() & self.snapModifier) else None
        pos = roi.mapToParent(ev.pos())
        if self.dragMode == 'translate':
            newPos = pos + self.cursorOffset
            roi.translate(newPos - roi.pos(), snap=snap, finish=False)
        elif self.dragMode == 'scale':
            print('SCALE!')
            diff = self.scaleSpeed ** -(ev.scenePos() - ev.buttonDownScenePos()).y()
            #roi.setSize(Point(self.startState['size']) * diff, centerLocal=ev.buttonDownPos(), snap=snap, finish=False)
            roi.setSize(Point(self.startState['size']) * diff, snap=snap, finish=False)

class MyROI(ROI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mouseDragHandler = MouseDragHandler(self)

        self.addScaleHandle([1,0],[0,1])
        color = randint(0,256,3)

        self.pen = fn.mkPen(color, width=1.66)
        self.setPen(self.pen)

        self.snip_pen = fn.mkPen(color, width=.66)
        self.conv_pen = fn.mkPen((0,0,0), width=1)

    def setMain(self,main):

        self.main = main
        self.image_plot = main.image_plot
        self.spectra_plot = main.spectra_plot
        self.intensity_plot = main.intensity_plot
        self.data = main.data
        self.img = main.img

    def setSize(self, size, center=None, centerLocal=None, snap=False, update=True, finish=True):
        """
        Set the ROI's size.
        """
        if update not in (True, False):
            raise TypeError("update argument must be bool")
        size = Point(size)
        if snap:
            size[0] = round(size[0] / self.scaleSnapSize) * self.scaleSnapSize
            size[1] = round(size[1] / self.scaleSnapSize) * self.scaleSnapSize

        if centerLocal is not None:
            oldSize = Point(self.state['size'])
            oldSize[0] = 1 if oldSize[0] == 0 else oldSize[0]
            oldSize[1] = 1 if oldSize[1] == 0 else oldSize[1]
            center = Point(centerLocal) / oldSize

        if center is not None:
            center = Point(center)
            c = self.mapToParent(Point(center) * self.state['size'])
            c1 = self.mapToParent(Point(center) * size)
            newPos = self.state['pos'] + c - c1
            self.setPos(newPos, update=False, finish=False)

        self.prepareGeometryChange()
    
        # I added this
        size[0] = max(size[0],1)
        size[1] = max(size[1],1)

        self.state['size'] = size

        if update:
            self.stateChanged(finish=finish)


    def zgetArraySlice(self, data, img, axes=(0,1)):
        """
        Get ROI positions.
        """
        dShape = (data.shape[axes[0]], data.shape[axes[1]])
        try:
            tr = self.sceneTransform() * fn.invertQTransform(img.sceneTransform())
        except np.linalg.linalg.LinAlgError:
            return None
        axisOrder = img.axisOrder
        if axisOrder == 'row-major':
            tr.scale(float(dShape[1]) / img.width(), float(dShape[0]) / img.height())
        else:
            tr.scale(float(dShape[0]) / img.width(), float(dShape[1]) / img.height())

        dataBounds = tr.mapRect(self.boundingRect())

        x = [int(round(dataBounds.left())),int(round(dataBounds.right()))]
        y = [int(round(dataBounds.top())),int(round(dataBounds.bottom()))]

        return [x,y]

    def calculate(self):
        """
        Roi
        """
        _y,_x = self.zgetArraySlice(self.data.integrated_spectra,self.img)
        s1 = slice(*_x)
        s2 = slice(*_y)

        z = self.data.inverted[s1,s2]
        conv = self.data.convoluted[s1,s2]

        #print('roi shape:',z.shape)

        res = 1.0 / (z.shape[0] * z.shape[1])

        self.z = z.sum(axis=0).sum(axis=0).astype(float)
        self.conv = conv.sum(axis=0).sum(axis=0).astype(float)

        if self.spectra_plot.normalized_roi == True:
            res = 255.0 / self.z.max()

        self.z *= res
        self.conv *= res

        self.snip_z = self.snip(self.conv)

        return self.z

    def crop(self):
        _y,_x = self.zgetArraySlice(self.data.integrated_spectra,self.img)
        s1 = slice(*_x)
        s2 = slice(*_y)
        z = self.data.image[s1,s2]

        return z[::-1]

    def roiUpdate(self):
        self.calculate()
        self.redraw()

    def snip(self,data):
        x = data.copy()

        for p in range(1,self.spectra_plot.snip_m)[::-1]:
            a1 = x[p:-p]
            a2 = (x[:(-2 * p)] + x[(2 * p):]) * 0.5
            x[p:-p] = minimum(a2,a1)

        return x

    def redraw(self):
        self.spectra_plot.clear()

        for roi in self.image_plot.roi_list:

            if self.main.calibration == True:

                if self.spectra_plot.subtract_snip == True:
                    roi.item = self.spectra_plot.plot(roi.data.calibration.cx,roi.z - roi.snip_z,pen=roi.pen)
                    roi.item_conv = self.spectra_plot.plot(roi.data.calibration.cx,roi.conv - roi.snip_z,pen=roi.conv_pen)

                else:
                    roi.item = self.spectra_plot.plot(roi.data.calibration.cx,roi.z,pen=roi.pen)
                    roi.item_conv = self.spectra_plot.plot(roi.data.calibration.cx,roi.conv,pen=roi.conv_pen)
                    roi.item_snip = self.spectra_plot.plot(roi.data.calibration.cx,roi.snip_z,pen=roi.snip_pen)

                self.spectra_plot.setLabel('bottom',text='Angle')


            else:
                if self.spectra_plot.subtract_snip == True:
                    roi.item = self.spectra_plot.plot(roi.z - roi.snip_z,pen=roi.pen)
                    roi.item_conv = self.spectra_plot.plot(roi.conv - roi.snip_z,pen=roi.conv_pen)
                else:
                    roi.item = self.spectra_plot.plot(roi.z,pen=roi.pen)
                    roi.item_conv = self.spectra_plot.plot(roi.conv,pen=roi.conv_pen)
                    roi.item_snip = self.spectra_plot.plot(roi.snip_z,pen=roi.snip_pen)

                self.spectra_plot.setLabel('bottom',text='Channel')


    def mouseClickEvent(self,event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self in self.image_plot.roi_list:
                self.image_plot.roi_list.remove(self)
                self.image_plot.removeItem(self)

                self.spectra_plot.removeItem(self.item)
                self.spectra_plot.removeItem(self.item_conv)

                if hasattr(self,'item_snip'):
                    self.spectra_plot.removeItem(self.item_snip)

            #self.redraw()
    
        else:
            p = self.mapToView(event.pos())
            x,y = p.x(),p.y()
            print('x: %d y: %d'%(x,y))
