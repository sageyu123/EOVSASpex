from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from IPython import embed
# import astropy.coordinates as cd

import os
import numpy as np
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sunpy.cm.cm as cm  ## to bootstrap sdoaia color map
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.module_class import Viewer_Module
from utils.slicer_class import Slicer
from suncasa.utils import DButil
import sunpy.map as smap
from sunpy.visualization import wcsaxes_compat


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


def readfits(file_path, curr_field_id=0):
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
    # ui_file = "FITSViewer'.ui"
    ui_file = "ImageControl.ui"

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

    def __init__(self, parent):
        self.parent_widget = parent
        self.image_control = uic.loadUi(os.path.split(__file__)[0] + '/../../ui/' + self.ui_file, QtWidgets.QWidget())
        self.parent_widget.Control_tab.addTab(self.image_control, 'Image')
        self.header_viewer = FITSHeaderViewer()
        self.mapdict = {}

        # --- set default states
        self.image_control.control_colorBar_check.setChecked(True)
        self.image_control.control_log_check.setChecked(False)
        self.image_control.control_abs_check.setChecked(False)

        # --- add colors to the list

        self.image_control.control_colors_list.addItems(['jet', 'viridis', 'gray', 'Greys', 'hot', 'OrRd', 'rainbow'])

        # --- assign control actions/signals

        self.image_control.control_colors_list.activated.connect(self.update_plot)

        self.image_control.control_log_check.stateChanged.connect(self.update_plot)
        self.image_control.control_abs_check.stateChanged.connect(self.update_plot)
        self.image_control.control_colorNorm_check.stateChanged.connect(self.update_plot)
        self.image_control.control_colorBar_check.stateChanged.connect(self.plot)
        self.image_control.control_wcs_check.stateChanged.connect(self.plot)

        # --- Data field selector
        self.image_control.control_dataField.activated.connect(self.update_data)

        # --- Display mode selector
        self.image_control.control_displayMode.activated.connect(self.plot)

        # --- Rotate dial
        self.image_control.rotate_dial.valueChanged.connect(self.update_plot)
        # self.side_bar.fits_showHeader_button.clicked.connect(self.on_pushShowHeaderButton_clicked)

        # --- Data drange sliders
        self.image_control.Slider_dmin.valueChanged.connect(self.update_plot)
        self.image_control.Slider_dmax.valueChanged.connect(self.update_plot)

        # --- Draw solar grid and limb
        self.image_control.control_SolarGrid_check.stateChanged.connect(self.update_plot)
        self.image_control.control_SolarLimb_check.stateChanged.connect(self.update_plot)
        self.solargrid = {'grid': [], 'limb': []}
        pass

    def module_info(self):
        """Returning a string representing short info about module"""
        return self.info_msg

    def open_file(self, file_path):
        """Opens data file"""
        # embed()
        try:
            if self.mapdict:
                self.fits_data[self.fits_current_field_id].header = self.mapdict['header'][
                    self.parent_widget.image_layers[self.parent_widget.curr_layer].idx]
                self.update_data()
            else:
                self.fits_data = fits.open(file_path)
                self.fits_data.verify('silentfix+warn')
                self.fits_fields = [str(key[0]) + ' ' + key[1] for key in self.fits_data.info(False)]
                self.fits_fields_to_id = dict(zip(self.fits_fields, range(len(self.fits_fields))))

                for idx, field in enumerate(self.fits_fields):
                    if not type(self.fits_data[idx].data) == np.ndarray:
                        self.fits_fields.remove(field)
                        del self.fits_fields_to_id[field]
                # embed()
                self.image_control.control_dataField.clear()
                self.image_control.control_dataField.addItems(self.fits_fields)  # should remove first
                self.image_control.control_displayMode.clear()
                self.image_control.control_displayMode.addItems(['Images', 'Transparent Contours'])  # should remove first
                self.parent_widget.actionDisplay_Header.setEnabled(True)
                self.parent_widget.actionDisplay_Header.triggered.connect(self.DisplayHeader)
                self.update_data()
                self.set_plot_data()
        except Exception as err:
            self.parent_widget.statusbar.showMessage("[Error] Have some troubles with file: %s " % file_path)
            print('')
            print('   [!] Failed to open FITS file: %s' % file_path)
            print('   [!] Error msg: ')
            print('   [!]', err)
            print('')

    def DisplayHeader(self):
        # update header viewer
        self.header_viewer.setTableData(self.fits_data[self.fits_current_field_id].header)
        self.header_viewer.show()

    def update_layerinfo(self):
        self.parent_widget.image_layers[self.parent_widget.curr_layer].log = self.image_control.control_log_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].abs = self.image_control.control_abs_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].colorNorm = self.image_control.control_colorNorm_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].colorBar = self.image_control.control_colorBar_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].color = self.image_control.control_colors_list.currentIndex()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].cmap = self.image_control.control_colors_list.currentText()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].dataField = self.image_control.control_dataField.currentIndex()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].img_rotation = self.image_control.rotate_dial.value()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].wcs = self.image_control.control_wcs_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].solargrid = self.image_control.control_SolarGrid_check.isChecked()
        self.parent_widget.image_layers[self.parent_widget.curr_layer].solarlimb = self.image_control.control_SolarLimb_check.isChecked()
        if self.image_control.control_displayMode.currentText() == 'Images':
            self.parent_widget.image_layers[self.parent_widget.curr_layer].dmode = 'imgs'
        elif self.image_control.control_displayMode.currentText() == 'Transparent Contours':
            self.parent_widget.image_layers[self.parent_widget.curr_layer].dmode = 'tconts'
        else:
            self.parent_widget.image_layers[self.parent_widget.curr_layer].dmode = 'imgs'
        # embed()
        try:
            self.parent_widget.image_layers[self.parent_widget.curr_layer].combos_idx = [ll.currentIndex() for ll in self.img_slicer.combos]
            self.parent_widget.image_layers[self.parent_widget.curr_layer].slicers_val = [ll.value() for ll in self.img_slicer.slicers]
        except Exception as err:
            self.parent_widget.statusbar.showMessage("[!] 2 Dimensions data, no slicer widget created.")
        # embed()
        return self.parent_widget.image_layers[self.parent_widget.curr_layer].info

    def update_data(self):
        self.fits_current_field = self.image_control.control_dataField.currentText()
        self.fits_current_field_id = self.fits_fields_to_id[self.fits_current_field]
        if self.mapdict == {}:
            # self.img_slicer = Slicer(self, self.fits_data[self.fits_current_field_id].data)
            self.img_slicer = Slicer(self, self.fits_data[self.fits_current_field_id].data, self.fits_data[self.fits_current_field_id].header)
        else:
            # self.img_slicer = Slicer(self, self.mapdict['submc'][self.parent_widget.image_layers[self.parent_widget.curr_layer].idx].data)
            self.img_slicer = Slicer(self, self.mapdict['submc'][self.parent_widget.image_layers[self.parent_widget.curr_layer].idx].data,
                                     self.mapdict['submc'][self.parent_widget.image_layers[self.parent_widget.curr_layer].idx].header)

    def set_plot_data(self):
        print('set_plot_data')
        # embed()
        self.plot_settings = self.update_layerinfo()

        if self.plot_settings['dmode'] == 'tconts':
            self.img_data = self.img_slicer.data.copy()
            if self.mapdict == {}:
                img_data_2D = self.img_slicer.get_slice().copy()
                self.spmap = smap.Map((img_data_2D, self.fits_data[self.fits_current_field_id].header))
            else:
                self.spmap = self.mapdict['submc'][self.parent_widget.image_layers[self.parent_widget.curr_layer].idx]
        else:
            self.img_data = self.img_slicer.get_slice().copy()
            if self.mapdict == {}:
                if self.fits_data[self.fits_current_field_id].header['TELESCOP'] == 'SDO/AIA':
                    self.spmap = smap.Map((self.img_data, self.fits_data[self.fits_current_field_id].header))
                    self.spmap = DButil.normalize_aiamap(self.spmap)
                    self.img_data = self.spmap.data
                else:
                    self.spmap = smap.Map((self.img_data, self.fits_data[self.fits_current_field_id].header))
            else:
                self.spmap = self.mapdict['submc'][self.parent_widget.image_layers[self.parent_widget.curr_layer].idx]

        if self.plot_settings['abs']:
            self.img_data = np.abs(self.img_data)

        try:
            drange = self.parent_widget.image_layers[self.parent_widget.curr_layer].drange
            if drange[1] > 1e5:
                vmin = self.image_control.Slider_dmin.value() * 1e5
                vmax = self.image_control.Slider_dmax.value() * 1e5
            else:
                vmin = self.image_control.Slider_dmin.value()
                vmax = self.image_control.Slider_dmax.value()
        except:
            vmin = self.image_control.Slider_dmin.value()
            vmax = self.image_control.Slider_dmax.value()
        if self.plot_settings['log']:
            lowcut = 1.0
            if vmin < lowcut:
                vmin = lowcut
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if self.plot_settings['colorNorm']:
                norm = None
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if self.plot_settings['dmode'] == 'imgs':
            img_rotation = self.plot_settings['img_rotation']
            if (img_rotation > 0):
                self.img_data = np.rot90(self.img_data, img_rotation)

        self.imshow_args = {'cmap': cm.get_cmap(self.plot_settings['cmap']), 'norm': norm, 'interpolation': 'nearest',
                            'origin': 'lower'}

        if self.plot_settings['wcs']:
            try:
                xaxis_type = self.spmap.coordinate_system.axis1
                yaxis_type = self.spmap.coordinate_system.axis2
                xaxis_unit = self.spmap.spatial_units.axis1
                yaxis_unit = self.spmap.spatial_units.axis2
            except:
                xaxis_type = self.spmap.coordinate_system.x
                yaxis_type = self.spmap.coordinate_system.y
                xaxis_unit = self.spmap.spatial_units.x
                yaxis_unit = self.spmap.spatial_units.y

            if xaxis_type == 'HG':
                xlabel = 'Longitude ({lon})'.format(lon=xaxis_unit)
            else:
                xlabel = 'Solar X-position ({xpos})'.format(xpos=xaxis_unit)
            if yaxis_type == 'HG':
                ylabel = 'Latitude ({lat})'.format(lat=yaxis_unit)
            else:
                ylabel = 'Solar Y-position ({ypos})'.format(ypos=yaxis_unit)
            self.parent_widget.figure.axes.set_xlabel(xlabel)
            self.parent_widget.figure.axes.set_ylabel(ylabel)
            self.draw_limb()
            self.draw_grid()
            self.imshow_args.update({'extent': list(self.spmap.xrange.to(u.arcsec).value) + list(self.spmap.yrange.to(u.arcsec).value)})

    def draw_limb(self):
        if self.plot_settings['solarlimb']:
            if len(self.solargrid['limb']) == 0:
                # self.solargrid['limb'] += self.spmap.draw_limb(axes=self.parent_widget.figure.axes, linewidth=0.5, alpha=1.0, color='w')  # embed()
                rsun = self.spmap.rsun_obs
                phi = np.linspace(-180, 180, num=181) * u.deg
                x = np.cos(phi) * rsun
                y = np.sin(phi) * rsun
                self.solargrid['limb'] += self.parent_widget.figure.axes.plot(x, y, linewidth=0.5, alpha=1.0, color='w', linestyle='-')
        else:
            self.clear_solarlimb()

    def draw_grid(self, grid_spacing=15. * u.deg):
        if self.plot_settings['solargrid']:
            if len(self.solargrid['grid']) == 0:
                def hgs2hcc(rsun, lon, lat, B0, L0):
                    lon_L0 = lon - L0
                    x = rsun * np.cos(lat) * np.sin(lon)
                    y = rsun * (np.sin(lat) * np.cos(B0) - np.cos(lat) * np.cos(lon_L0) * np.sin(B0))
                    z = rsun * (np.sin(lat) * np.sin(B0) + np.cos(lat) * np.cos(lon_L0) * np.cos(B0))
                    return x, y, z

                def hcc2hpc(x, y, z, dsun):
                    d = np.sqrt(x ** 2 + y ** 2 + (dsun - z) ** 2)
                    Tx = np.arctan2(x, dsun - z)
                    Ty = np.arcsin(y / d)
                    return Tx, Ty

                dsun = self.spmap.dsun
                rsun = self.spmap.rsun_meters

                b0 = self.spmap.heliographic_latitude.to(u.deg)
                l0 = self.spmap.heliographic_longitude.to(u.deg)
                hg_longitude_deg = np.linspace(-90, 90, num=91) * u.deg
                hg_latitude_deg = np.arange(-90, 90, grid_spacing.to(u.deg).value) * u.deg

                for lat in hg_latitude_deg:
                    c = hgs2hcc(rsun, hg_longitude_deg, lat * np.ones(91), b0, l0)
                    coords = hcc2hpc(c[0], c[1], c[2], dsun)
                    self.solargrid['grid'] += self.parent_widget.figure.axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), linewidth=0.5,
                                                                                  alpha=1.0, color='w', linestyle='--')

                hg_longitude_deg = np.arange(-90, 90, grid_spacing.to(u.deg).value) * u.deg
                hg_latitude_deg = np.linspace(-90, 90, num=91) * u.deg

                for lon in hg_longitude_deg:
                    c = hgs2hcc(rsun, lon * np.ones(91), hg_latitude_deg, b0, l0)
                    coords = hcc2hpc(c[0], c[1], c[2], dsun)
                    self.solargrid['grid'] += self.parent_widget.figure.axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), linewidth=0.5,
                                                                                  alpha=1.0, color='w', linestyle='--')
        else:
            self.clear_solargrid()

    def clear_solargrid(self):
        if self.solargrid['grid']:
            for col in self.solargrid['grid']:
                try:
                    col.remove()
                except:
                    pass
            self.solargrid['grid'] = []

    def clear_solarlimb(self):
        if self.solargrid['limb']:
            for col in self.solargrid['limb']:
                try:
                    col.remove()
                except:
                    pass
            self.solargrid['limb'] = []

    def reset_plot(self):
        self.parent_widget.figure.axes.cla()
        self.clear_solargrid()
        self.clear_solarlimb()

    def plot_colorbar(self, custom_cb=None):
        self.divider = make_axes_locatable(self.parent_widget.figure.axes)
        self.ax_cb = self.divider.new_horizontal(size="5%", pad=0.05)
        self.parent_widget.figure.fig.add_axes(self.ax_cb)
        if custom_cb:
            self.cbar = colorbar.ColorbarBase(self.ax_cb, norm=colors.Normalize(vmax=custom_cb['vmax'], vmin=custom_cb['vmin']),
                                              cmap=custom_cb['cmap'])
            if self.spmap.meta['telescop'] == 'SDO/AIA':
                self.ax_cb.set_ylabel('Wavelength [A]')
            else:
                try:
                    self.ax_cb.set_ylabel('{} [{}]'.format(self.img_slicer.header['CTYPE3'], self.spmap.meta['CUNIT3']))
                except:
                    print('Telescope not recognized.')
        else:
            self.cbar = self.parent_widget.figure.fig.colorbar(self.plot_data, cax=self.ax_cb)
            if self.spmap.meta['telescop'] == 'SDO/AIA':
                self.ax_cb.set_ylabel('Intensity [DN/s]')
            else:
                try:
                    self.ax_cb.set_ylabel('{} [{}]'.format(self.spmap.meta['btype'], self.spmap.meta['bunit']))
                except:
                    print('Telescope not recognized.')

    def plot(self):
        print('plot')
        self.set_plot_data()
        # self.parent_widget.figure.axes.cla()
        self.reset_plot()
        self.parent_widget.clickedpoints, = self.parent_widget.figure.axes.plot([], [], 'o', color='red')
        self.parent_widget.slitline, = self.parent_widget.figure.axes.plot([], [], color='white', ls='solid')
        self.parent_widget.slitline0, = self.parent_widget.figure.axes.plot([], [], color='white', ls='dotted')
        self.parent_widget.slitline1, = self.parent_widget.figure.axes.plot([], [], color='white', ls='dotted')
        # embed()
        if self.plot_settings['dmode'] == 'tconts':
            #todo reset the plot when update levels
            #todo no reset when update the cmap,clim,norm
            self.plot_data = []
            # embed()

            # nspw = self.img_slicer.slicers[-1].maximum() - self.img_slicer.slicers[-1].minimum()
            nspw = self.img_slicer.header['NAXIS3']
            #todo change the slicers [index] for the case of dim >3
            XX, YY = np.meshgrid(np.arange(self.spmap.data.shape[1]), np.arange(self.spmap.data.shape[0]))
            mesh = self.spmap.pixel_to_world(XX * u.pix, YY * u.pix)
            mapx, mapy = mesh.Tx.value, mesh.Ty.value
            levels = np.array([0.2, 1.0])
            cmap = cm.get_cmap(self.plot_settings['cmap'])
            for s in range(nspw):
                self.plot_data.append(
                    self.parent_widget.figure.axes.contourf(mapx, mapy, self.img_data[s, ...], levels=levels * np.nanmax(self.img_data[s, ...]),
                                                            colors=[cmap(float(s + 1) / nspw)] * len(levels), alpha=0.25, antialiased=True))
                # This is the fix for the white lines between contour levels
                for cnt in self.plot_data[-1].collections:
                    cnt.set_edgecolor("face")
                    cnt.set_linewidth(0.000000000000000)  # ax.set_title(' ')

            # allways try to remove color bar first to avoid overplots when new Fits file is added
            try:
                self.parent_widget.figure.fig.delaxes(self.ax_cb)
            except:
                pass
            vmin = self.img_slicer.header['CRVAL3']
            vmax = vmin + nspw*self.img_slicer.header['CDELT3']
            if self.image_control.control_colorBar_check.isChecked():
                self.plot_colorbar(custom_cb={'cmap': cm.get_cmap(self.plot_settings['cmap']), 'vmax': vmax, 'vmin': vmin})

        else:
            self.plot_data = self.parent_widget.figure.axes.imshow(self.img_data, **self.imshow_args)  # import pdb  # pdb.set_trace()
            # allways try to remove color bar first to avoid overplots when new Fits file is added
            try:
                self.parent_widget.figure.fig.delaxes(self.ax_cb)
            except:
                pass

            if self.image_control.control_colorBar_check.isChecked():
                self.plot_colorbar()

        self.parent_widget.figure.axes.format_coord = lambda x, y: '[ x = %.2f , y = %.2f ]' % (x, y)
        self.parent_widget.figure.axes.format_cursor_data = "x"
        self.parent_widget.figure.axes.format_zdata = "nope"
        if self.plot_settings['wcs']:
            self.set_plot_data()
            try:
                self.plot_data.set_extent(self.imshow_args['extent'])
            except:
                pass
        self.parent_widget.figure.draw()
        self.parent_widget.statusbar.showMessage("Plot refeshed!")

    def update_plot(self):
        """Updates plot widget"""
        print('update_plot')
        who = self.image_control.sender()
        try:
            if who.objectName() == 'Slider_dmin' or who.objectName() == 'Slider_dmax':
                drange = self.parent_widget.image_layers[self.parent_widget.curr_layer].drange
                if drange[1] > 1e5:
                    who.parent().setTitle('{} [{:.2e}~{:.2e}] --> {:.2e}'.format(who.objectName()[-4:], drange[0], drange[1], who.value() * 1e5))
                else:
                    who.parent().setTitle('{} [{:.0f}~{:.0f}] --> {:.0f}'.format(who.objectName()[-4:], drange[0], drange[1], who.value()))
                print(who.objectName())
        except:
            pass
        self.parent_widget.statusbar.showMessage("Update Plot!")
        self.set_plot_data()
        if self.plot_settings['dmode'] == 'tconts':
            pass  #todo
        else:
            try:
                self.plot_data.set_extent(self.imshow_args['extent'])
            except:
                pass
            self.plot_data.set_data(self.img_data)
            self.plot_data.axes.autoscale(False, 'both', True)
            self.plot_data.axes.autoscale_view(True, True, True)
            self.plot_data.axes.relim(visible_only=True)
            self.plot_data.set_norm(self.imshow_args['norm'])
            self.plot_data.set_cmap(self.imshow_args['cmap'])
        self.parent_widget.figure.draw()  # embed()

    def LoadChunk(self):
        maplist = []
        timestamps = []
        header = []
        # try:
        x0, x1 = self.parent_widget.figure.axes.get_xlim()
        y0, y1 = self.parent_widget.figure.axes.get_ylim()
        for sidx, sfile in enumerate(self.parent_widget.image_layers[self.parent_widget.curr_layer].url):
            # embed()
            h, d = readfits(sfile, curr_field_id=self.fits_current_field_id)
            maptmp = smap.Map((d, h))
            if h['TELESCOP'] == 'SDO/AIA':
                maptmp = DButil.normalize_aiamap(maptmp)
            submaptmp = maptmp.submap(u.Quantity([x0 * u.arcsec, x1 * u.arcsec]), u.Quantity([y0 * u.arcsec, y1 * u.arcsec]))
            maplist.append(submaptmp)
            timestamps.append(Time(submaptmp.meta['date-obs'].replace('T', ' '), format='iso', scale='utc').jd)
            header.append(h)
        mc = smap.Map(maplist, cube=True)
        self.mapdict = {'FOV': [x0, y0, x1, y1], 'subFOV': [x0, y0, x1, y1], 'mc': mc, 'time': np.array(timestamps), 'header': header}
        self.mapdict['submc'] = self.LoadSubChunk(self.mapdict, x0, x1, y0, y1)
        self.parent_widget.statusbar.showMessage(
            "Chunk Loaded!")  # embed()  # self.figure.  # except Exception as err:  #     print('')  #     print('   [!] Filed to load chunk')  #     print('   [!] Error msg: ')  #     print('   [!]', err)  #     print('')  #     pass

    def LoadSubChunk(self, mapdict, x0, x1, y0, y1):
        from sunpy.physics.transforms.solar_rotation import mapcube_solar_derotate
        submaplist = []
        for sidx, spmap in enumerate(mapdict['mc'].maps):
            submaptmp = spmap.submap(u.Quantity([x0 * u.arcsec, x1 * u.arcsec]), u.Quantity([y0 * u.arcsec, y1 * u.arcsec]))
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
            if x0 >= self.mapdict['FOV'][0] and x1 <= self.mapdict['FOV'][2] and y0 >= self.mapdict['FOV'][1] and y1 <= self.mapdict['FOV'][3]:
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
        except:
            pass
        pass

    def __del__(self):
        self.clear_gui()
