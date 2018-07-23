#!/Users/fisher/anaconda2/envs/py35/bin/python
# -*- coding: utf-8 -*-

__author__ = 'sjyu1988'

import sys
import os
import getopt
import numpy as np
from os.path import abspath, join, dirname
import collections
import matplotlib
import pylab
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from PyQt5.QtWidgets import QMainWindow, QWidget, QTextEdit, QSizePolicy, QAction, QFileDialog, QMessageBox, QApplication, QSplitter
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, uic, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from suncasa.utils import DButil

myFolder = os.path.split(os.path.realpath(__file__))[0]
sys.path = [myFolder, os.path.join(myFolder, 'widgets')] + sys.path

os.chdir(myFolder)

from utils import module_manager
from utils.module_class import Viewer_Module
from utils.layer_class import Layer

# --- Warnigs setup

import warnings

warnings.filterwarnings("ignore")

# --- Initial setup

pylab.style.use('dark_background')
matplotlib.rcParams.update({'font.size': 6})

# --- Global variables -------------
global avaible_modules, loaded_modules, verbose, clickpointDF, cutslitplt
avaible_modules = collections.defaultdict(list)
prog_name = os.path.basename(sys.argv[0])
prog_ver = 0.1
verbose = True
clickpointDF = pd.DataFrame({'xx': [], 'yy': []})
cutslitplt = {}

ascii_logo = open('./resources/viewer_logo.ascii', 'r').read()
help_message = '''
   :::::::::::::::::::::::::::::::::::::::::::::::::::
   ::     Python based quick look tool app Solar    ::
   :::::::::::::::::::::::::::::::::::::::::::::::::::

    You can setup some parameters from command line:

    -i / --input  - point to the snap file
    -h / --help   - print this help msg

   ............................................................
'''


# class layers(layer):
#     def __init__(self, lys=None):
#         super(layers, self).__init__()
#         pass


def FitSlit(xx, yy, cutwidth, cutang, cutlength, s=None, method='Polyfit'):
    # if len(xx) <= 3 or method == 'Polyfit':
    #     '''polynomial fit'''
    out = DButil.polyfit(xx, yy, cutlength, len(xx) - 1 if len(xx) <= 3 else 2)
    xs, ys, posangs = out['xs'], out['ys'], out['posangs']
    # else:
    #     if method == 'Param_Spline':
    #         '''parametic spline fit'''
    #         out = DButil.paramspline(xx, yy, cutlength, s=s)
    #         xs, ys, posangs = out['xs'], out['ys'], out['posangs']
    #     else:
    #         '''spline fit'''
    #         out = DButil.spline(xx, yy, cutlength, s=s)
    #         xs, ys, posangs = out['xs'], out['ys'], out['posangs']
    # if not ascending and (fitmethod != 'Param_Spline' or len(xx) <= 3):
    #     xs, ys = xs[::-1], ys[::-1]
    #     posangs = posangs[::-1]
    dist = DButil.findDist(xs, ys)
    dists = np.cumsum(dist)
    posangs2 = posangs + np.pi / 2
    cutwidths = dists * np.tan(cutang) + cutwidth
    xs0 = xs - cutwidths / 2. * np.cos(posangs2)
    ys0 = ys - cutwidths / 2. * np.sin(posangs2)
    xs1 = xs + cutwidths / 2. * np.cos(posangs2)
    ys1 = ys + cutwidths / 2. * np.sin(posangs2)
    return {'xcen': xs, 'ycen': ys, 'xs0': xs0, 'ys0': ys0, 'xs1': xs1, 'ys1': ys1, 'cutwidth': cutwidths, 'posangs': posangs, 'posangs2': posangs2,
            'dist': dists}


def MakeSlit(pointDF):
    pointDFtmp = pointDF
    xx = pointDFtmp.loc[:, 'xx'].values
    yy = pointDFtmp.loc[:, 'yy'].values
    if len(pointDFtmp.index) <= 1:
        cutslitplt = {'xcen': [], 'ycen': [], 'xs0': [], 'ys0': [], 'xs1': [], 'ys1': [], 'cutwidth': [], 'posangs': [], 'posangs2': [], 'dist': []}
    else:
        # if len(pointDFtmp.index) <= 3:
        cutslitplt = FitSlit(xx, yy, 10, 0.0, 200, method='Polyfit')
    return cutslitplt


def getimprofile(data, cutslit, xrange=None, yrange=None):
    num = len(cutslit['xcen'])
    if num > 1:
        intens = np.zeros(num)
        ndy, ndx = data.shape
        if xrange is not None and yrange is not None:
            xs0 = (cutslit['xs0'] - xrange[0]) / (xrange[1] - xrange[0]) * ndx
            xs1 = (cutslit['xs1'] - xrange[0]) / (xrange[1] - xrange[0]) * ndx
            ys0 = (cutslit['ys0'] - yrange[0]) / (yrange[1] - yrange[0]) * ndy
            ys1 = (cutslit['ys1'] - yrange[0]) / (yrange[1] - yrange[0]) * ndy
        else:
            xs0 = cutslit['xs0']
            xs1 = cutslit['xs1']
            ys0 = cutslit['ys0']
            ys1 = cutslit['ys1']
        for ll in range(num):
            inten = DButil.improfile(data, [xs0[ll], xs1[ll]], [ys0[ll], ys1[ll]], interp='nearest')
            intens[ll] = np.mean(inten)
        intensdist = {'x': cutslit['dist'], 'y': intens}
        return intensdist


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=7, dpi=150):
        self.fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # self.fig.tight_layout()

        # self.axes.hold(False) # We want the axes cleared every time plot() is
        # called

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.setParent(parent)

        # load boot screen picture
        self.figure_data = self.axes.imshow(matplotlib.image.imread('./resources/init.jpg'))
        self.axes.set_axis_off()


class MainWindow(QMainWindow):
    # [file_format] -> [List of modules names capable to deal wit it]
    file_extension_to_module_name = collections.defaultdict(list)
    loaded_modules = collections.defaultdict(Viewer_Module)
    active_module = None

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.main_ui = uic.loadUi('ui/viewer.ui', self)
        self.initUI()

    def initUI(self):
        self.setAcceptDrops(True)
        # embed()
        # self.plotWidget.size
        # plotsz = self.plotWidget.size()
        self.figure = MyMplCanvas(self.plotWidget, width=7, height=7, dpi=150)
        # self.figure.setFocusPolicy(QtCore.Qt.ClickFocus)
        # self.figure.setFocus()
        self.navi_toolbar = NavigationToolbar(self.figure, self)
        self.navi_toolbar.setFloatable(True)
        self.navi_toolbar.setMovable(True)
        self.navi_toolbar.setIconSize(QtCore.QSize(18, 18))
        self.navi_toolbar.setMaximumSize(QtCore.QSize(16777215, 25))
        # self.navi_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.plotLayout = QVBoxLayout(self.plotWidget)
        self.plotLayout.addWidget(self.navi_toolbar)
        self.plotLayout.addWidget(self.figure)
        self.plotLayout.setContentsMargins(0, 0, 2, 2)
        self.actionOpen.triggered.connect(self.showDialog)
        self.actionExit.triggered.connect(self.close)
        self.setWindowTitle("Viewer")
        # self.actionTrack.triggered.connect(self.testaction)
        self.actiontestaction.triggered.connect(self.testaction)
        self.btn_next.clicked.connect(self.btn_next_click)
        self.btn_prev.clicked.connect(self.btn_prev_click)
        cid = self.figure.fig.canvas.mpl_connect('button_press_event', self.onclick_handler)
        self.actionslit.triggered[bool].connect(self.slit_handler)
        self.clickedpoints, = self.figure.axes.plot([], [], 'o', color='white')
        self.slitline, = self.figure.axes.plot([], [], color='white', ls='solid')
        self.slitline0, = self.figure.axes.plot([], [], color='white', ls='dotted')
        self.slitline1, = self.figure.axes.plot([], [], color='white', ls='dotted')
        self.image_layers = collections.defaultdict(list)
        self.curr_layer = None

        # --- check which modules are avaible
        for module_name, module_handle in avaible_modules.items():
            self.registerModule(module_name, module_handle)
            # also add to module list
            self.main_ui.menuModules.addAction(QAction(module_name, self))

        self.show()

    def registerModule(self, module_name, module_handle):
        module_class = getattr(module_handle, module_name)
        module_instance = module_class(self)
        for file_extension in module_instance.file_extensions:
            self.file_extension_to_module_name[file_extension].append(module_name)

        self.loaded_modules[module_name] = module_instance

    def plot(self):
        if self.active_module is not None:
            self.active_module.plot()

    def update_plot(self):
        if self.active_module is not None:
            self.active_module.update_plot()

    def update_data(self):
        if self.active_module is not None:
            self.active_module.update_data()
            self.active_module.update_plot()

    def openByFileName(self, path, fast=False):
        self.statusbar.showMessage("Opening [{}] file!".format(path))
        if fast:
            self.active_module.open_file(path)
            self.update_plot()
        else:
            extension = os.path.splitext(path)[1]
            # embed()
            if extension in self.file_extension_to_module_name:
                module_names = self.file_extension_to_module_name[extension]
                if len(module_names) > 1:
                    self.statusbar.showMessage("Wow! More then one module can Open this file ..choose which one")
                else:
                    if (self.active_module):
                        self.active_module.hide_gui()
                        self.active_module.clear_gui()
                    self.setActiveModule(self.loaded_modules[module_names[0]])
                    self.active_module.show_gui()
                    self.active_module.open_file(path)
                    self.plot()
            else:
                self.statusbar.showMessage("No module to deal with [{}] file!".format(path))

    def setActiveModule(self, module):
        self.active_module = module
        self.statusbar.showMessage("Module [{}] activated!".format(module.name))

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '~')
        if fname[0]:
            print(fname[0])  # f = open(fname[0], 'r')

            # with f:  #     data = f.read()  #     self.textEdit.setText(data)

    # comment this part when test
    # def closeEvent(self, event):
    #     reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                paths.append(path)
        print(path)
        if len(paths) > 0:
            nrow = len(self.image_layers.keys())
            self.curr_layer = nrow
            self.image_layers[nrow] = Layer(nrow, paths, 0)
            self.ImageLayer_Table.setRowCount(nrow + 1)
            self.ImageLayer_Table.setItem(nrow, 0, QtWidgets.QTableWidgetItem(str(nrow)))
            self.ImageLayer_Table.resizeColumnsToContents()
            self.ImageLayer_Table.resizeRowsToContents()
            self.openByFileName(self.image_layers[self.curr_layer].url[0])
            self.signalDisconnect(self.Slider_frame.valueChanged[int], self.Slider_frame_handler)
            self.Slider_frame.setMinimum(1)
            self.Slider_frame.setMaximum(self.image_layers[nrow].nidx)
            self.Slider_frame.setValue(1)
            self.Slider_frame.valueChanged[int].connect(self.Slider_frame_handler)
            self.label_frame.setText('{}/{}'.format(self.image_layers[nrow].idx + 1, self.image_layers[nrow].nidx))
            self.image_layers[nrow].telescope = self.active_module.fits_data[self.active_module.fits_current_field_id].header['TELESCOP']
            self.image_layers[nrow].drange = [np.nanmin(self.active_module.img_data), np.nanmax(self.active_module.img_data)]
            qc = self.active_module.image_control.control_colors_list
            allitems = []
            if self.image_layers[nrow].telescope == 'SDO/AIA':
                for ll in range(qc.count()):
                    allitems.append(qc.itemText(ll))
                allitems = ["sdoaia{}".format(self.active_module.fits_data[self.active_module.fits_current_field_id].header['WAVELNTH'])] + allitems
            else:
                for ll in range(qc.count()):
                    if 'sdoaia' not in qc.itemText(ll):
                        allitems.append(qc.itemText(ll))
            qc.clear()
            qc.addItems(allitems)
            qc.setCurrentIndex(0)
            dmin = self.image_layers[nrow].drange[0]
            dmax = self.image_layers[nrow].drange[1]
            dstep = (dmax-dmin)/100.
            if dmax>1e5:
                self.active_module.image_control.Slider_dmin_GBox.setTitle('dmin [{0:.2e}~{1:.2e}] --> {0:.2e}'.format(dmin, dmax))
                self.active_module.image_control.Slider_dmin.setMinimum(dmin/1e5)
                self.active_module.image_control.Slider_dmin.setMaximum(dmax/1e5)
                self.active_module.image_control.Slider_dmin.setSingleStep(dstep/1e5)
                self.active_module.image_control.Slider_dmin.setValue(dmin/1e5)
            else:
                self.active_module.image_control.Slider_dmin_GBox.setTitle('dmin [{0:.0f}~{1:.0f}] --> {0:.0f}'.format(dmin, dmax))
                self.active_module.image_control.Slider_dmin.setMinimum(dmin)
                self.active_module.image_control.Slider_dmin.setMaximum(dmax)
                self.active_module.image_control.Slider_dmin.setSingleStep(dstep)
                self.active_module.image_control.Slider_dmin.setValue(dmin)

            if dmax > 1e5:
                self.active_module.image_control.Slider_dmax_GBox.setTitle('dmax [{0:.2e}~{1:.2e}] --> {0:.2e}'.format(dmin, dmax))
                self.active_module.image_control.Slider_dmax.setMinimum(dmin/1e5)
                self.active_module.image_control.Slider_dmax.setMaximum(dmax/1e5)
                self.active_module.image_control.Slider_dmax.setSingleStep(dstep/1e5)
                self.active_module.image_control.Slider_dmax.setValue(dmax/1e5)
            else:
                self.active_module.image_control.Slider_dmax_GBox.setTitle('dmax [{0:.0f}~{1:.0f}] --> {0:.0f}'.format(dmin, dmax))
                self.active_module.image_control.Slider_dmax.setMinimum(dmin)
                self.active_module.image_control.Slider_dmax.setMaximum(dmax)
                self.active_module.image_control.Slider_dmax.setSingleStep(dstep)
                self.active_module.image_control.Slider_dmax.setValue(dmax)

            self.signalDisconnect(self.active_module.image_control.Slider_dmin.valueChanged, self.active_module.update_plot)
            self.signalDisconnect(self.active_module.image_control.Slider_dmax.valueChanged, self.active_module.update_plot)
            if self.image_layers[nrow].telescope == 'SDO/AIA':
                self.signalDisconnect(self.active_module.image_control.control_log_check.stateChanged, self.active_module.update_plot)
                self.active_module.image_control.control_log_check.setChecked(True)
                self.signalConnect(self.active_module.image_control.control_log_check.stateChanged, self.active_module.update_plot)
                clrange = DButil.sdo_aia_scale_dict(
                    wavelength=self.active_module.fits_data[self.active_module.fits_current_field_id].header['WAVELNTH'])
                print('default color normalization:', clrange)
                # embed()
                self.active_module.image_control.Slider_dmin.setValue(clrange['low'])
                self.active_module.image_control.Slider_dmax.setValue(clrange['high'])
                self.active_module.image_control.Slider_dmin.parent().setTitle(
                    '{0} [{1[0]:.0f}~{1[1]:.0f}] --> {2:.0f}'.format('dmin', self.image_layers[nrow].drange, clrange['low']))
                self.active_module.image_control.Slider_dmax.parent().setTitle(
                    '{0} [{1[0]:.0f}~{1[1]:.0f}] --> {2:.0f}'.format('dmax', self.image_layers[nrow].drange, clrange['high']))
            self.signalConnect(self.active_module.image_control.Slider_dmin.valueChanged, self.active_module.update_plot)
            self.signalConnect(self.active_module.image_control.Slider_dmax.valueChanged, self.active_module.update_plot)
            self.actionLoadChunk.triggered.connect(self.active_module.LoadChunk_handler)
            self.active_module.update_plot()
        elif os.path.isdir(path):
            self.statusbar.showMessage("File type error (unknown extension)")  # print(paths)

    def signalDisconnect(self, signal, slot):
        try:
            signal.disconnect(slot)
        except:
            pass

    def signalConnect(self, signal, slot):
        try:
            signal.connect(slot)
        except:
            pass

    def testaction(self):
        stackplt = []
        for idx, ll in enumerate(self.image_layers[self.curr_layer].url):
            self.image_layers[self.curr_layer].idx = idx
            self.Slider_frame.setValue(idx + 1)
            intens = getimprofile(self.active_module.img_data, cutslitplt, xrange=list(self.active_module.spmap.xrange.value),
                                  yrange=list(self.active_module.spmap.yrange.value))
            stackplt.append(intens['y'])
        if len(stackplt) > 1:
            stackplt = np.vstack(stackplt)
            stackplt = stackplt.transpose()
        imshow_args = dict((k, self.active_module.imshow_args[k]) for k in ('cmap', 'norm'))
        plt.pcolormesh(stackplt, **imshow_args)
        plt.show()

    def btn_next_click(self):
        if self.curr_layer is not None:
            idx = self.image_layers[self.curr_layer].idx
            idx += 1
            idx = idx % self.image_layers[self.curr_layer].nidx
            self.image_layers[self.curr_layer].idx = idx
            self.Slider_frame.setValue(idx + 1)

    def btn_prev_click(self):
        if self.curr_layer is not None:
            idx = self.image_layers[self.curr_layer].idx
            idx -= 1
            idx = idx % self.image_layers[self.curr_layer].nidx
            self.image_layers[self.curr_layer].idx = idx
            self.Slider_frame.setValue(idx + 1)

    def Slider_frame_handler(self, value):
        if self.curr_layer is not None:
            self.image_layers[self.curr_layer].idx = value - 1
            self.label_frame.setText('{}/{}'.format(value, self.image_layers[self.curr_layer].nidx))
            if 'mapdict' in dir(self.active_module):
                self.openByFileName(self.image_layers[self.curr_layer].url[value - 1], fast=True)
            else:
                self.openByFileName(self.image_layers[self.curr_layer].url[value - 1])
            self.slit_handler(checked=self.actionslit.isChecked())

    def onclick_handler(self, event=None):
        global clickpointDF, cutslitplt
        print(event.xdata, event.ydata)
        if self.actionslit.isChecked():
            if event.xdata != None:
                if event.button == 1:
                    clickpointDF = clickpointDF.append(pd.Series({'xx': event.xdata, 'yy': event.ydata}), ignore_index=True)
                elif event.button == 3:
                    if len(clickpointDF.index) > 0:
                        clickpointDF.drop(len(clickpointDF.index) - 1, inplace=True)
                cutslitplt = MakeSlit(clickpointDF)
                self.pushslit2plt(cutslitplt, clickpointDF)

    def slit_handler(self, checked=False):
        if checked:
            self.pushslit2plt(cutslitplt, clickpointDF)
        else:
            self.clearslitplt()

    def pushslit2plt(self, cutslit, clkptDF):
        self.clickedpoints.set_data(clkptDF['xx'].as_matrix(), clkptDF['yy'].as_matrix())
        self.clickedpoints.figure.canvas.draw()
        if cutslit:
            self.slitline.set_data(cutslit['xcen'], cutslit['ycen'])
            self.slitline0.set_data(cutslit['xs0'], cutslit['ys0'])
            self.slitline1.set_data(cutslit['xs1'], cutslit['ys1'])
            self.slitline.figure.canvas.draw()
            self.slitline0.figure.canvas.draw()
            self.slitline1.figure.canvas.draw()

    def clearslitplt(self):
        self.clickedpoints.set_data([], [])
        self.slitline.set_data([], [])
        self.slitline0.set_data([], [])
        self.slitline1.set_data([], [])
        self.clickedpoints.figure.canvas.draw()
        self.slitline.figure.canvas.draw()
        self.slitline0.figure.canvas.draw()
        self.slitline1.figure.canvas.draw()


def main(argv=None):
    global help_message, avaible_modules

    print(ascii_logo % (prog_name, prog_ver))

    if argv is None:
        argv = sys.argv

        opts, args = getopt.getopt(argv[1:], "hi:", ["help", "input="])

        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                print(help_message)
                return 0
            if option in ("-i", "--input"):
                input_file = value
        pass

        print(help_message)

    avaible_modules = module_manager.import_modules('core_modules/', verbose)
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show
    try:
        mw.openByFileName(input_file)
    except:
        pass
    sys.exit(app.exec_())


if __name__ == '__main__':
    sys.exit(main())
