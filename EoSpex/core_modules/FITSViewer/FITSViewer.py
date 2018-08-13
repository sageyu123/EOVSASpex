from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets, QtGui

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from IPython import embed
# import astropy.coordinates as cd
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout
import os
import numpy as np
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import sunpy.cm.cm as cm  ## to bootstrap sdoaia color map
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.module_class import Viewer_Module
from utils.slicer_class import Slicer
import sunpy.map as smap
import copy
from sunpy.visualization import wcsaxes_compat
from utils.layer_class import Layer, LayerManager, LayerList
import collections

from utils.utils import timetest
from utils import utils


class FITSHeaderViewer(QtWidgets.QWidget):
    ui_file = "FITSHeader.ui"

    def __init__(self, parent=None):
        super(FITSHeaderViewer, self).__init__(parent)
        self.parent_widget = parent
        self.ui = uic.loadUi(os.path.split(__file__)[0] + '/' + self.ui_file, self)

    def setTableData(self, header_data):
        self.ui.fits_headerTable.setRowCount(1)  # small trick to clear table
        self.ui.fits_headerTable.setRowCount(len(header_data.cards))
        for n, row_data in enumerate(header_data.cards):
            key = QtWidgets.QTableWidgetItem(str(row_data[0]))
            val = QtWidgets.QTableWidgetItem(str(row_data[1]))
            com = QtWidgets.QTableWidgetItem(str(row_data[2]))

            self.ui.fits_headerTable.setItem(n, 0, key)
            self.ui.fits_headerTable.setItem(n, 1, val)
            self.ui.fits_headerTable.setItem(n, 2, com)

        self.ui.fits_headerTable.resizeColumnsToContents()
        self.ui.fits_headerTable.resizeRowsToContents()
        self.ui.fits_headerTable.show()


def readfits_chunk(file_path, curr_field_id=0):
    from astropy.io import fits
    fits_data = fits.open(file_path)
    fits_data.verify('silentfix+warn')
    fits_fields = [str(key[0]) + ' ' + key[1] for key in fits_data.info(False)]
    fits_fields_to_id = dict(zip(fits_fields, range(len(fits_fields))))

    for idx, field in enumerate(fits_fields):
        if not type(fits_data[idx].data) == np.ndarray:
            fits_fields.remove(field)
            del fits_fields_to_id[field]
    return fits_data[curr_field_id].header, fits_data[curr_field_id].data


class FITSViewer(Viewer_Module):
    name = "FITSViewer"
    info_msg = "Basic FITS reader/visualiser"
    file_extensions = ['.fits']

    parent_widget = None

    fits_data = None
    fits_fields = None
    fits_fields_to_id = None
    fits_current_field = None
    fits_current_field_id = None
    fits_current_header = None

    img_slicer = None
    img_data = None
    img_rotation = None

    slider_dmax_value = None
    slider_dmin_value = None

    def __init__(self, parent):
        self.parent_widget = parent
        # self.image_control = uic.loadUi(os.path.split(__file__)[0] + '/../../ui/' + self.ui_file, QtWidgets.QWidget())
        # self.parent_widget.Control_tab.addTab(self.image_control, 'Image')
        self.header_viewer = FITSHeaderViewer()
        self.mapdict = {}

        self.solargrid = {'grid': [], 'limb': []}

        self.parent_widget.LayerLayout = QVBoxLayout(self.parent_widget.LayerlistBox)
        self.layermanger = LayerManager(parent=self)

    def module_info(self):
        """Returning a string representing short info about module"""
        return self.info_msg

    # def readfits(self, url=None):
    #     self.fits_data = fits.open(url)
    #     self.fits_data.verify('silentfix+warn')
    #     self.fits_fields = [str(key[0]) + ' ' + key[1] for key in self.fits_data.info(False)]
    #     self.fits_fields_to_id = dict(zip(self.fits_fields, range(len(self.fits_fields))))
    #
    #     for idx, field in enumerate(self.fits_fields):
    #         if not type(self.fits_data[idx].data) == np.ndarray:
    #             self.fits_fields.remove(field)
    #             del self.fits_fields_to_id[field]

    def open_file(self, file_path):
        """Opens data file"""
        # embed()
        # try:
        # if self.mapdict:
        #     self.fits_data[self.fits_current_field_id].header = self.mapdict['header'][
        #         self.layermanger.image_layers[self.layermanger.curr_layer].idx]
        #     # self.updatedata_from_widget()
        # else:
        self.layermanger.register(url=file_path)
        # except Exception as err:
        #     self.parent_widget.statusbar.showMessage("[Error] Have some troubles with file: %s " % file_path[0])
        #     print('')
        #     print('   [!] Failed to open FITS file: %s' % file_path[0])
        #     print('   [!] Error msg: ')
        #     print('   [!]', err)
        #     print('')

    def DisplayHeader(self):
        # update header viewer
        self.header_viewer.setTableData(self.fits_data[self.fits_current_field_id].header)
        self.header_viewer.show()




    # def reset_plot(self, who=None):
    #     print('reset plot')
    #     self.parent_widget.figure.axes.cla()
    #     self.clear_solargrid()
    #     self.clear_solarlimb()


    def LoadChunk(self):
        maplist = []
        timestamps = []
        header = []
        # try:
        x0, x1 = self.parent_widget.figure.axes.get_xlim()
        y0, y1 = self.parent_widget.figure.axes.get_ylim()
        for sidx, sfile in enumerate(self.layermanger.image_layers[self.layermanger.curr_layer].url):
            # embed()
            h, d = readfits_chunk(sfile, curr_field_id=self.fits_current_field_id)
            maptmp = smap.Map((d, h))
            if h['TELESCOP'] == 'SDO/AIA':
                maptmp = utils.normalize_aiamap(maptmp)
            submaptmp = maptmp.submap(u.Quantity([x0 * u.arcsec, x1 * u.arcsec]),
                                      u.Quantity([y0 * u.arcsec, y1 * u.arcsec]))
            maplist.append(submaptmp)
            timestamps.append(Time(submaptmp.meta['date-obs'].replace('T', ' '), format='iso', scale='utc').jd)
            header.append(h)
        mc = smap.Map(maplist, cube=True)
        self.mapdict = {'FOV': [x0, y0, x1, y1], 'subFOV': [x0, y0, x1, y1], 'mc': mc, 'time': np.array(timestamps),
                        'header': header}
        self.mapdict['submc'] = self.LoadSubChunk(self.mapdict, x0, x1, y0, y1)
        self.parent_widget.statusbar.showMessage(
            "Chunk Loaded!")  # embed()  # self.figure.  # except Exception as err:  #     print('')  #     print('   [!] Filed to load chunk')  #     print('   [!] Error msg: ')  #     print('   [!]', err)  #     print('')  #     pass

    def LoadSubChunk(self, mapdict, x0, x1, y0, y1):
        from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate
        submaplist = []
        for sidx, spmap in enumerate(mapdict['mc'].maps):
            submaptmp = spmap.submap(u.Quantity([x0 * u.arcsec, x1 * u.arcsec]),
                                     u.Quantity([y0 * u.arcsec, y1 * u.arcsec]))
            submaplist.append(submaptmp)
        try:
            submc = mapcube_solar_derotate(smap.Map(submaplist, cube=True))
        except:
            submc = smap.Map(submaplist, cube=True)
        return submc

    def UpdateSubChunk(self):
        x0, x1 = self.parent_widget.figure.axes.get_xlim()
        y0, y1 = self.parent_widget.figure.axes.get_ylim()
        self.mapdict['submc'] = self.LoadSubChunk(self.mapdict, x0, x1, y0, y1)
        self.mapdict['subFOV'] = [x0, y0, x1, y1]

    def LoadChunk_handler(self):
        x0, x1 = self.parent_widget.figure.axes.get_xlim()
        y0, y1 = self.parent_widget.figure.axes.get_ylim()
        if self.mapdict:
            if x0 >= self.mapdict['FOV'][0] and x1 <= self.mapdict['FOV'][2] and y0 >= self.mapdict['FOV'][1] and y1 <= \
                    self.mapdict['FOV'][3]:
                self.UpdateSubChunk()
            else:
                self.LoadChunk()
        else:
            self.LoadChunk()  # update_sdosubmp_image(Slider_sdoidx.value - 1)

    @QtCore.pyqtSlot()
    def on_pushShowHeaderButton_clicked(self):
        self.header_viewer.exec_()

    def show_gui(self):
        # self.side_bar.show()
        pass

    def hide_gui(self):
        # self.side_bar.hide()
        pass

    def clear_gui(self):
        # self.parent_widget.control_dataField.clear()
        try:
            self.img_slicer = None
            self.img_data = None
            self.parent_widget.figure.fig.delaxes(self.ax_cb)
            self.image_control.destroy()
        except:
            pass
        pass

    def __del__(self):
        self.clear_gui()
