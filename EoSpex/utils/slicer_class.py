from PyQt5 import QtWidgets
from PyQt5 import QtCore
import numpy as np
from IPython import embed


class Slicer(object):
    """Slicer. Takes varius object and define "slice" which can be visualise.

    Attributes:
        name: Object name
        slice: An array.. basis for visualisation
        slice_dimension: Slice dimension must be in [1,2,3]
        parent_object:
        parent_ui: one which we can modify to add or remove different slicers
    """

    name = "data slicer"
    ui_file = "empty.ui"
    parent_obj = None
    parent_layout = None
    data = None
    shape_raw = None
    dim_raw = None
    dim = -1
    data_slice = None
    slicers = None
    slicers_titles = None
    slicers_widgets = None
    slicers_idx = None
    slicers_axisinfo = None

    def __init__(self, parent_obj, squeeze=True):

        self.parent_obj = parent_obj
        self.parent_ui = parent_obj.image_control
        # embed()
        self.parent_layout = parent_obj.image_control.slicer_group.layout()
        self.parent_layout.setContentsMargins(0, 0, 0, 0)
        array = parent_obj.fits_data[parent_obj.fits_current_field_id].data
        header = parent_obj.fits_data[parent_obj.fits_current_field_id].header
        self.shape_raw = array.shape
        self.dim_raw = len(self.shape_raw)
        self.header = header

        if (squeeze):
            self.data = np.squeeze(array)
        else:
            self.data = array

        self.dim = len(self.data.shape)
        print('dimensions:', self.dim)
        self.slicers_axisinfo = {'DIM_RAW': self.dim_raw, 'NAXIS_RAW': [], 'CTYPE_RAW': [], 'CUNIT_RAW': [], 'CRVAL_RAW': [], 'CDELT_RAW': [],
                                 'CRPIX_RAW': [], 'DIM_RAW_IDX': [], 'DIM': self.dim, 'NAXIS': [], 'CTYPE': [], 'CUNIT': [], 'CRVAL': [], 'CDELT': [],
                                 'CRPIX': [], 'DIM_IDX': []}

        if (self.dim > 2):
            self.slicers_widgets = list()
            self.slicers_widgets_idx = {}
            self.slicers = list()
            self.slicers_titles = list()
            self.combos_widgets = list()
            self.combos = list()
            self.combos_titles = list()
            if header:
                for i, s in enumerate(self.shape_raw):
                    self.slicers_axisinfo['NAXIS_RAW'].append(header['NAXIS{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['CTYPE_RAW'].append(header['CTYPE{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['CUNIT_RAW'].append(header['CUNIT{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['CRVAL_RAW'].append(header['CRVAL{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['CDELT_RAW'].append(header['CDELT{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['CRPIX_RAW'].append(header['CRPIX{}'.format(self.dim_raw - i)])
                    self.slicers_axisinfo['DIM_RAW_IDX'].append(i)
                    if s > 1:
                        self.slicers_axisinfo['NAXIS'].append(header['NAXIS{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['CTYPE'].append(header['CTYPE{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['CUNIT'].append(header['CUNIT{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['CRVAL'].append(header['CRVAL{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['CDELT'].append(header['CDELT{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['CRPIX'].append(header['CRPIX{}'.format(self.dim_raw - i)])
                        self.slicers_axisinfo['DIM_IDX'].append(len(self.slicers_axisinfo['CRPIX']) - 1)
                self.slicers_names = ['Dim {} of size [{}]'.format(self.slicers_axisinfo['CTYPE'][i], limit) for i, limit in
                                      enumerate(self.data.shape)]
            else:
                self.slicers_axisinfo['NAXIS_RAW'] = list(self.shape_raw)
                self.slicers_axisinfo['DIM_RAW_IDX'] = [i for i in range(self.dim_raw)]
                self.slicers_axisinfo['NAXIS'] = list(self.shape)
                self.slicers_axisinfo['DIM_IDX'] = [i for i in range(self.dim)]
                self.slicers_names = ['Dim {} of size [{}]'.format(i, limit) for i, limit in enumerate(self.data.shape)]
            # clear the layout to avoid overlaying when new widget is added
            self.clearLayout(self.parent_layout)

            for i, lblname in enumerate(['X dim', 'Y dim']):
                w, c = self.create_dim_select(lblname)
                c.setCurrentIndex(self.dim - i - 1)
                self.combos_widgets.append(w)
                self.combos.append(c)
                self.combos_titles.append(lblname)

            for w in self.combos_widgets:
                self.parent_layout.addWidget(w)

            for c in self.combos:
                c.activated[str].connect(self.update_dim)

            for i, limit in enumerate(self.data.shape[:-2]):
                slicer_name = self.slicers_names[i]
                w, s, t = self.create_dim_slicer(i, slicer_name, limit)
                self.slicers_widgets.append(w)
                self.slicers_widgets_idx[i] = i
                self.slicers.append(s)
                self.slicers_titles.append(t)

            for w in self.slicers_widgets:
                self.parent_layout.addWidget(w)
            for i, s in enumerate(self.slicers):
                s.valueChanged[int].connect(self.update_slice)

            self.slicers_idx = [slice(None)] * self.dim
            for ll in range(self.dim - 2):
                self.slicers_idx[ll] = 0
        elif self.dim == 2:
            self.data_slice = self.data
        else:
            print('   [!] [Warning] Sorry no support for 1D data slices!')
            self.nope = np.load('./resources/CautionDirtyData.png')
            self.nope = np.array(self.nope, dtype=np.float)
            self.data_slice = np.flipud(self.nope)

        # self.parent_ui.control_dim3.setRange(0,self.data.shape[-1]-1)
        self.parent_ui.groupBox_dataField.setTitle('Data Field of size: ' + self.shape_to_str(self.shape_raw))

        pass

    def __del__(self):
        self.parent_ui.groupBox_dataField.setTitle('Data Field')

        if self.slicers:
            for s in self.slicers:
                s.disconnect()

        if self.slicers_widgets:
            for w in self.slicers_widgets:
                self.parent_layout.removeWidget(w)
                w.setParent(None)

    def update_dim(self, text):
        who = self.parent_ui.sender()
        # comboset = set(self.combos_titles)
        print('dim {} is activated in {}'.format(who.currentIndex(), who.objectName()))
        # who = self.parent_ui.sender()
        dim = str(who.objectName())
        print(text, dim)  # print('Value %d for dim %s' % (text, dim))
        ## currently not implemented

    def update_slice(self, value):
        who = self.parent_ui.sender()
        dim = int(who.objectName())
        print('Value %d for dim %d' % (value, dim))
        slice_idx = list(self.slicers_idx)
        slice_idx[dim] = value
        slice_idx = tuple(slice_idx)
        print(slice_idx)
        dim_the_same = ([type(x) for x in slice_idx] == [type(x) for x in self.slicers_idx])
        self.slicers_idx = slice_idx
        if (dim_the_same):
            self.parent_obj.plot_update()  # just update
            print('update_plot')
        else:
            # embed()
            print('plot')
            self.parent_obj.plot_initi()
        pass
        # Set title on bar
        self.slicers_widgets[self.slicers_widgets_idx[dim]].setTitle(self.slicers_titles[self.slicers_widgets_idx[dim]] % value)

    def get_slice(self, nans_free=False):
        if (self.dim > 2):
            self.data_slice = self.data[self.slicers_idx]

        if (nans_free):
            self.data_slice = self.data_slice[np.isnan(self.data_slice)] = np.nanmin(self.data_slice)

        return self.data_slice

    def module_info(self):
        """Returning a string representing short info about module"""
        pass

    def create_dim_select(self, labelname):
        combo = QtWidgets.QComboBox()
        combo.addItems(self.slicers_names)
        gbox_title = labelname
        gbox = QtWidgets.QGroupBox(gbox_title)
        gbox.setMaximumSize(QtCore.QSize(16777215, 50))
        vlay = QtWidgets.QVBoxLayout(gbox)
        combo.setObjectName(labelname)  # store information about dimension
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.addWidget(combo)
        return gbox, combo

    def create_dim_slicer(self, dim_id, name, limit):
        gbox_title = name + " (indx = %d)"
        gbox = QtWidgets.QGroupBox(gbox_title % 0)
        gbox.setMaximumSize(QtCore.QSize(16777215, 50))
        vlay = QtWidgets.QVBoxLayout(gbox)
        vlay.setContentsMargins(0, 0, 0, 0)
        slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slide.setMaximumSize(QtCore.QSize(16777215, 22))
        slide.setMinimum(0)
        slide.setMaximum(limit - 1)
        slide.setObjectName(str(dim_id))  # store information about dimension
        slide.setTickInterval(1)
        slide.setTickPosition(0)
        vlay.addWidget(slide)
        return gbox, slide, gbox_title

    def shape_to_str(self, shape):
        return '[' + ','.join([str(x) for x in shape]) + ']'

    # def make_index(self, slice_pos):
    #     zeros = [0] * self.dim
    #     zeros[slice_pos] = np.slice(None)
    #     return tuple(zeros)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
