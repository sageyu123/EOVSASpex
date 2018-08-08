from collections import OrderedDict, defaultdict
from astropy.time import Time
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QListView
from PyQt5 import QtWidgets, QtCore
from IPython import embed
import copy


# from enum import Enum
# class LayerType(Enum):

class LayerList(QListView):

    def __init__(self):
        super(LayerList, self).__init__()
        # self.parent = parent
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragEnabled(True)
        self.setTabKeyNavigation(True)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.setMaximumSize(QtCore.QSize(1000, 1000))
        self.model = QStandardItemModel(self)
        self.setModel(self.model)

        # embed()

    def dragLeaveEvent(self, event):
        selectedIdx = self.selectedIndexes()[0].row()
        print('remove item ', self.model.item(selectedIdx).text())
        self.model.removeRow(selectedIdx)

        # def dropEvent(self, event):
        #     embed()


class Layer:
    def __init__(self, layer_id: 'layer_id', url: 'layer_url', frame_idx: 'frame_idx') -> 'layer_class':
        self.alpha = 1.0
        self.layer_id = layer_id
        self.layer_name = 'layer{}'.format(layer_id)
        self.url = url
        self.frame_idx = frame_idx
        self.nidx = len(url)
        self.bound = [0, -1]
        self.log = False
        self.abs = False
        self.colorNorm = False
        self.colorBar = True
        self.color = 0
        self.cmap = None
        self.cmaps = []
        self.dataField = 0
        self.dataFields = []
        self.img_rotation = 0
        self.wcs = True
        self.solargrid = True
        self.solarlimb = True
        self.dmode = None
        self.dmodes = None
        self.dmode_name = None
        self.slicers_axisinfo = None
        self.dim_idx = None
        self.combos_idx = None
        self.slicers_val = None
        self.fits_data = None
        self.fits_fields = None
        self.fits_fields_to_id = None

    @property
    def info(self):
        return vars(self)

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)


class LyManager:
    def __init__(self, parent=None):
        super(LyManager, self).__init__()
        self.parent = parent
        self.List = LayerList()
        self.image_layers = defaultdict(list)
        self.parent.parent_widget.LayerLayout.addWidget(self.List)
        self.parent.parent_widget.LayerLayout.setContentsMargins(0, 0, 2, 1)
        self.curr_layer = 0
        self.List.model.itemChanged.connect(self.testevent)
        self.List.clicked.connect(self.select)

    def testevent(self, item):
        print(item.text(), item.checkState())
        # embed()

    def header2title(self, header):
        telescope = header['TELESCOP']
        if telescope == 'SDO/AIA':
            h = {'INSTRUME': 'AIA', 'WAVELNTH': '{:.0f}A'.format(header['WAVELNTH']),
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        elif telescope == 'EOVSA':
            h = {'INSTRUME': 'EOVSA', 'WAVELNTH': '{:.2f}GHz'.format(header['RESTFRQ'] / 1e9),
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        else:
            h = {'INSTRUME': header['TELESCOP'], 'WAVELNTH': '',
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        layertitle = '{h[INSTRUME]}{h[WAVELNTH]}    {h[DATE-OBS]}'.format(h=h)
        return layertitle

    def layer2fitsdata(self, layer=None):
        self.parent.fits_data = layer.fits_data.copy()
        self.parent.fits_fields = layer.fits_fields.copy()
        self.parent.fits_fields_to_id = layer.fits_fields_to_id.copy()

    def register(self, url=None):
        if url is str:
            url = [url]
        self.parent.readfits(url[0])
        nrow = self.List.model.rowCount()
        self.curr_layer = nrow
        self.image_layers[nrow] = Layer(nrow, url, 0)
        self.parent.image_control.control_dataField.clear()  # always remove items first
        self.parent.image_control.control_dataField.addItems(self.parent.fits_fields)
        self.parent.parent_widget.actionDisplay_Header.setEnabled(True)
        self.parent.parent_widget.actionDisplay_Header.triggered.connect(self.parent.DisplayHeader)
        self.parent.update_data()
        self.parent.image_control.control_displayMode.clear()
        control_displaymode = ['Images']
        for cidx, c_dm in enumerate(self.parent.img_slicer.slicers_axisinfo['CTYPE'][:-2]):
            control_displaymode.append('Transparent Contours - Dim {}'.format(c_dm))
        self.parent.image_control.control_displayMode.addItems(control_displaymode)
        self.parent.set_plot_data()
        print(' regieter layer: ', self.curr_layer)
        item = QStandardItem(self.header2title(self.parent.img_slicer.header))
        item.setCheckable(True)
        item.setDragEnabled(True)
        item.setDropEnabled(False)
        item.setCheckState(Qt.Checked)
        self.List.model.appendRow(item)
        # embed()

    def select(self, item):
        print(item.row(), item.data())
        layer_id = item.row()
        layer = self.image_layers[layer_id]
        self.layer2fitsdata(layer)
        self.curr_layer = layer_id
        ic = self.parent.image_control
        ic.control_dataField.clear()  # always remove items first
        ic.control_dataField.addItems(self.parent.fits_fields)
        # self.parent.parent_widget.actionDisplay_Header.setEnabled(True)
        # self.parent.parent_widget.actionDisplay_Header.triggered.connect(self.parent.DisplayHeader)
        self.parent.update_data()
        ic.control_displayMode.clear()
        ic.control_displayMode.addItems(layer.dmodes)
        ic.control_displayMode.setCurrentIndex(layer.dmodes.index(layer.dmode_name))
        ic.control_colorBar_check.setChecked(True)
        ic.control_log_check.setChecked(False)
        ic.control_abs_check.setChecked(False)
        ic.control_colors_list.clear()
        print(layer)
        # embed()
        ic.control_colors_list.addItems(layer.cmaps)
        try:
            ic.control_colors_list.setCurrentIndex(layer.cmaps.index(layer.cmap))
        except:
            pass
        ic.control_log_check.setChecked(layer.log)
        ic.control_abs_check.setChecked(layer.abs)
        ic.control_colorNorm_check.setChecked(layer.colorNorm)
        ic.control_colorBar_check.setChecked(layer.colorBar)
        ic.control_wcs_check.setChecked(layer.wcs)
        # ic.Slider_dmin.valueChanged.connect(self.update_plot)
        # ic.Slider_dmax.valueChanged.connect(self.update_plot)
        ic.control_SolarGrid_check.setChecked(layer.solargrid)
        ic.control_SolarLimb_check.setChecked(layer.solarlimb)
        self.solargrid = {'grid': [], 'limb': []}

        self.parent.plot()

    def delete(self, layer_id=None):
        pass

    def widget2layer(self, layer_id=None):
        if layer_id is None:
            layer_id = self.curr_layer

        layer = self.image_layers[layer_id]
        layer.log = self.parent.image_control.control_log_check.isChecked()
        layer.abs = self.parent.image_control.control_abs_check.isChecked()
        layer.colorNorm = self.parent.image_control.control_colorNorm_check.isChecked()
        layer.colorBar = self.parent.image_control.control_colorBar_check.isChecked()
        ccl = self.parent.image_control.control_colors_list
        layer.color = ccl.currentIndex()
        layer.cmap = ccl.currentText()
        if not layer.cmaps:
            layer.cmaps = [ccl.itemText(i) for i in range(ccl.count())]
        cdf = self.parent.image_control.control_dataField
        layer.dataField = cdf.currentIndex()
        if not layer.dataFields:
            layer.dataFields = [cdf.itemText(i) for i in range(cdf.count())]
        layer.img_rotation = self.parent.image_control.rotate_dial.value()
        layer.wcs = self.parent.image_control.control_wcs_check.isChecked()
        layer.solargrid = self.parent.image_control.control_SolarGrid_check.isChecked()
        layer.solarlimb = self.parent.image_control.control_SolarLimb_check.isChecked()
        cdm = self.parent.image_control.control_displayMode
        if cdm.currentText() == 'Images':
            layer.dmode = 'imgs'
        elif cdm.currentText().startswith('Transparent Contours'):
            layer.dmode = 'tconts'
        else:
            layer.dmode = 'imgs'
        layer.dmode_name = cdm.currentText()
        if not layer.dmodes:
            layer.dmodes = [cdm.itemText(i) for i in range(cdm.count())]
        # embed()
        layer.slicers_axisinfo = self.parent.img_slicer.slicers_axisinfo
        if hasattr(self.parent.img_slicer, 'combos'):
            dim_idx = copy.copy(self.parent.img_slicer.slicers_axisinfo['DIM_IDX'])
            for ll in self.parent.img_slicer.combos:
                dim_idx.pop(dim_idx.index(ll.currentIndex()))
            layer.dim_idx = dim_idx[0]
        else:
            layer.dim_idx = 0
        try:
            layer.combos_idx = [ll.currentIndex() for ll in self.parent.img_slicer.combos]
            layer.slicers_val = [ll.value() for ll in self.parent.img_slicer.slicers]
        except Exception as err:
            self.parent.parent_widget.statusbar.showMessage("[!] 2 Dimensions data, no slicer widget created.")
        if not layer.fits_data:
            layer.fits_data = self.parent.fits_data.copy()
            layer.fits_fields = self.parent.fits_fields.copy()
            layer.fits_fields_to_id = self.parent.fits_fields_to_id.copy()
        return layer.info

    def layer2widget(self):
        pass
