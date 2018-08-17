import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
from matplotlib.contour import QuadContourSet
import sunpy.cm.cm as cm  ## to bootstrap sdoaia color map
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from collections import defaultdict
from astropy.time import Time
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QListView, QColorDialog
from PyQt5 import QtWidgets, QtCore
import copy
import numpy as np
from utils import utils
from utils.utils import timetest
import astropy.units as u
from utils.ImageControl_class import ImageControl, ImageControl_Limb, ImageControl_Grid
from utils.slicer_class import Slicer
from utils.layer_base_class import Layer_Line_Base, Layer_Image_Base
import sunpy.map as smap
import time
from IPython import embed


def signalDisconnect(signal, slot):
    try:
        signal.disconnect(slot)
    except:
        pass


def signalConnect(signal, slot):
    try:
        signal.connect(slot)
    except:
        pass


class LayerList(QListView):

    def __init__(self):
        super(LayerList, self).__init__()
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragEnabled(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setTabKeyNavigation(True)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.setMaximumSize(QtCore.QSize(1000, 1000))
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self.image_layers = None
        # embed()

    def dragLeaveEvent(self, event):
        selectedIdx = self.selectedIndexes()[0].row()
        print('remove item ', self.model.item(selectedIdx).whatsThis())
        self.image_layers[self.model.item(selectedIdx).whatsThis()].destroy()
        del self.image_layers[self.model.item(selectedIdx).whatsThis()]
        self.model.removeRow(selectedIdx)

    # def dropEvent(self, event):
    #     selectedIdx = self.selectedIndexes()[0].row()
    #     print('move item ', self.model.item(selectedIdx).whatsThis())
    #     self.dropIndicatorPosition()
    #     # embed()
    #     self.model.moveRow(self.model.item(selectedIdx).index(),1,self.model.item(self.dropIndicatorPosition()-1).index(),1)


class Layer_Image(Layer_Image_Base):
    def __init__(self, layer_id: 'layer_id', url: 'layer_url', frame_idx: 'frame_idx', parent_obj: 'parent_obj',
                 axes: 'axes') -> 'Layer_class':
        super(Layer_Image, self).__init__(layer_id, url, frame_idx, parent_obj, axes)
        self.parent = parent_obj
        self.axes = axes
        self.mapdict = {}
        self.alpha = 1.0
        self.layer_id = layer_id
        self.layer_name = 'layer{}'.format(layer_id)
        self.layer_title = None
        self.url = url
        self.frame_idx = frame_idx
        self.nidx = len(url)
        self.bound = [0, -1]
        self.slider_dmax_value = None
        self.slider_dmin_value = None
        self.log = False
        self.abs = False
        self.colorNorm = False
        self.colorBar = True
        self.cmap = None
        self.norm = None
        self.dataField = 0
        self.dataFields = []
        self.img_rotation = 0
        self.dmode = None
        self.slicers_axisinfo = None
        self.dim_idx = None
        self.fits_data = None
        self.fits_fields = None
        self.fits_fields_to_id = None
        self.fits_fields_title = ''
        self.fits_current_field_id = None
        self.telescope = None
        self.im = []
        self.im_cbar = None
        self.image_control = ImageControl()
        self.divider = None
        self.ax_cb = None
        self.img_data = None
        self.zorder = 0
        self.dataxy_image = []
        self.dataxy_conts = []
        self.plot_settings = {}

    def get_zorder(self):
        if self.im:
            if type(self.im) is list:
                if isinstance(self.im[0], QuadContourSet):
                    self.zorder = self.im[0].collections[0].get_zorder()
                else:
                    self.zorder = self.im[0].get_zorder()
            else:
                self.zorder = self.im.get_zorder()
        return self.zorder

    def set_zorder(self, zorder=1):
        if self.im:
            if type(self.im) is list:
                for im in self.im:
                    if isinstance(im, QuadContourSet):
                        for col in im.collections:
                            col.set_zorder(zorder)
                    else:
                        im.set_zorder(zorder)
            else:
                self.zorder = self.im.set_zorder(zorder)
        return self.zorder

    def readfits(self, url=None):
        self.fits_data = fits.open(url)
        self.fits_data.verify('silentfix+warn')
        self.fits_fields = [str(key[0]) + ' ' + key[1] for key in self.fits_data.info(False)]
        self.fits_fields_to_id = dict(zip(self.fits_fields, range(len(self.fits_fields))))

        for idx, field in enumerate(self.fits_fields):
            if not type(self.fits_data[idx].data) == np.ndarray:
                self.fits_fields.remove(field)
                del self.fits_fields_to_id[field]
            else:
                self.fits_data[idx].data[np.isnan(self.fits_data[idx].data)] = 0.0

    def setData(self):
        self.fits_current_field_id = self.fits_fields_to_id[self.image_control.control_dataField.currentText()]
        # embed()
        if self.mapdict == {}:
            self.img_slicer = Slicer(self)
        else:
            pass
            # self.img_slicer = Slicer(layer, self.mapdict['submc'][
            #     self.layermanger.image_layers[self.layermanger.curr_layer].idx].data,
            #                          self.mapdict['submc'][
            #                              self.layermanger.image_layers[self.layermanger.curr_layer].idx].header)

    def setColorMapList(self):
        ccl = self.image_control.control_colors_list
        cmaps = []
        if self.telescope == 'SDO/AIA':
            for ll in range(ccl.count()):
                cmaps.append(ccl.itemText(ll))
            aia_cmap = "sdoaia{}".format(self.fits_data[self.fits_current_field_id].header['WAVELNTH'])
            cmaps = [aia_cmap] + cmaps
        else:
            for ll in range(ccl.count()):
                if 'sdoaia' not in ccl.itemText(ll):
                    cmaps.append(ccl.itemText(ll))
        cmaps = sorted(cmaps)
        ccl.clear()
        ccl.addItems(cmaps)
        if self.telescope == 'SDO/AIA':
            ccl.setCurrentText(aia_cmap)
        else:
            ccl.setCurrentText('jet')

    def plot_colorbar(self, custom_cb=None):
        self.divider = make_axes_locatable(self.axes)
        self.ax_cb = self.divider.new_horizontal(size="5%", pad=0.05)
        self.axes.figure.add_axes(self.ax_cb)
        if custom_cb:
            self.im_cbar = colorbar.ColorbarBase(self.ax_cb,
                                                 norm=colors.Normalize(vmax=custom_cb['vmax'], vmin=custom_cb['vmin']),
                                                 cmap=custom_cb['cmap'], alpha=custom_cb['alpha'])
            if self.spmap.meta['telescop'] == 'SDO/AIA':
                self.ax_cb.set_ylabel('Wavelength [A]')
            else:
                try:
                    self.ax_cb.set_ylabel(
                        '{} [{}]'.format(self.plot_settings['slicers_axisinfo']['CTYPE'][
                                             self.plot_settings['dim_idx']],
                                         self.plot_settings['slicers_axisinfo']['CUNIT'][
                                             self.plot_settings['dim_idx']]))
                except:
                    print('Telescope not recognized.')
        else:
            self.im_cbar = self.axes.figure.colorbar(self.im, cax=self.ax_cb)
            if self.spmap.meta['telescop'] == 'SDO/AIA':
                self.ax_cb.set_ylabel('Intensity [DN/s]')
            else:
                try:
                    self.ax_cb.set_ylabel(
                        '{} [{}]'.format(self.spmap.meta['btype'], self.spmap.meta['bunit']))
                except:
                    print('Telescope not recognized.')

    def toggle_colorbar(self, s):
        self.im_cbar.ax.set_visible(bool(s))
        self.axes.figure.canvas.draw_idle()

    def plot_image(self):
        # print('plot_image')
        # embed()
        t0 = time.time()
        mapx, mapy = self.dataxy_image
        print('plot_image mark 1 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        # self.im = self.axes.pcolormesh(mapx, mapy, self.img_data, antialiased=True, rasterized=True,# linewidth=0.01,
        #                                **self.imshow_args)
        self.im = self.axes.imshow(self.img_data, **self.imshow_args)
        print('plot_image mark 2 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        self.axes.set_xlim(self.spmap.xrange.to(u.arcsec).value)
        self.axes.set_ylim(self.spmap.yrange.to(u.arcsec).value)
        try:
            # allways try to remove color bar first to avoid overplots when new Fits file is added
            self.axes.figure.delaxes(self.ax_cb)
        except:
            pass

        if self.image_control.control_colorBar_check.isChecked():
            self.plot_colorbar()
        print('plot_image mark 3 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        if self.img_slicer.slicers_widgets is not None:
            for s in self.img_slicer.slicers_widgets:
                s.setEnabled(True)
        print('plot_image mark 4 {:.4f}'.format(time.time() - t0))

    def plot_conts(self, dmax=0.0, dmin=0.0):
        t0 = time.time()
        levels = [dmin, dmax]
        naxis = self.plot_settings['slicers_axisinfo']['NAXIS'][self.plot_settings['dim_idx']]
        cmap = cm.get_cmap(self.plot_settings['cmap'])
        alpha = self.plot_settings['alpha']
        self.im = []
        print('plot_conts mark 1 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        mapx, mapy = self.dataxy_conts
        print('plot_conts mark 2 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        for s in range(naxis):
            self.im.append(
                self.axes.contourf(mapx, mapy, self.img_data[s, ...],
                                   levels=levels,
                                   colors=[cmap(float(s + 1) / naxis)] * len(levels),
                                   alpha=self.plot_settings['alpha'],
                                   antialiased=True))
            for cnt in self.im[-1].collections:
                # This is the fix for the white lines between contour levels
                cnt.set_edgecolor("face")
                cnt.set_linewidth(0.000000000000000)
        print('plot_conts mark 3 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        self.img_slicer.slicers_widgets[self.plot_settings['dim_idx']].setDisabled(True)
        self.axes.set_xlim(self.spmap.xrange.to(u.arcsec).value)
        self.axes.set_ylim(self.spmap.yrange.to(u.arcsec).value)
        print('plot_conts mark 4 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        try:
            # allways try to remove color bar first to avoid overplots when new Fits file is added
            self.axes.figure.delaxes(self.ax_cb)
        except:
            pass
        vmin = self.plot_settings['slicers_axisinfo']['CRVAL'][self.plot_settings['dim_idx']]
        vmax = vmin + naxis * self.plot_settings['slicers_axisinfo']['CDELT'][
            self.plot_settings['dim_idx']]
        if self.image_control.control_colorBar_check.isChecked():
            self.plot_colorbar(
                custom_cb={'cmap': cm.get_cmap(self.plot_settings['cmap']), 'vmax': vmax, 'vmin': vmin, 'alpha': alpha})
        print('plot_conts mark 5 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        self.axes.figure.canvas.draw_idle()
        print('plot_conts mark 6 {:.4f}'.format(time.time() - t0))

    def plot(self):
        t0 = time.time()
        print('plot-------------')
        # fov = None
        who = self.image_control.sender()
        print('plot mark 1 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        if who is not None:
            if who.objectName() == 'control_displayMode':
                if type(self.im) is list:
                    for col in self.im:
                        if isinstance(col, QuadContourSet):
                            for c in col.collections:
                                c.remove()
                        else:
                            col.remove()
                else:
                    self.im.remove()
                self.im = []
        print('plot mark 2 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        # self.reset_plot(who=who)
        # self.axes.cla()
        parent_widget = self.parent.parent_widget
        parent_widget.clickedpoints, = self.axes.plot([], [], 'o', color='red')
        parent_widget.slitline, = self.axes.plot([], [], color='white', ls='solid')
        parent_widget.slitline0, = self.axes.plot([], [], color='white', ls='dotted')
        parent_widget.slitline1, = self.axes.plot([], [], color='white', ls='dotted')
        print('plot mark 3 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        print('plot mark 4 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        if self.dmode == 'tconts':
            dmin_new, dmax_new = self.reset_drange_sliders()
            if self.slider_dmax_value == dmax_new and self.slider_dmin_value == dmin_new:
                # Note: when change the display mode from the images mode to the contours mode,
                # the values of the sliders do not change.
                # Force to update the plot in this case
                self.plot_update()
        else:
            self.plot_image()
            if who is not None:
                if who.objectName() == 'control_displayMode':
                    self.reset_drange_sliders()
        print('plot mark 5 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        self.axes.format_coord = lambda x, y: '[ x = %.2f , y = %.2f ]' % (x, y)
        self.axes.format_cursor_data = "x"
        self.axes.format_zdata = "nope"
        self.axes.figure.canvas.draw_idle()
        print('plot mark 6 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        zoom_fov = self.parent.parent_widget.navi_toolbar.zoom_fov
        if zoom_fov:
            self.axes.set_xlim(zoom_fov[:2])
            self.axes.set_ylim(zoom_fov[2:])

        self.parent.parent_widget.statusbar.showMessage("Plot refeshed!")
        print('plot mark 7 {:.4f}'.format(time.time() - t0))

    # @timetest
    def plot_update(self):
        """Updates plot widget"""
        print('update_plot')
        t0 = time.time()
        updateplotflag = True
        ic = self.image_control
        who = ic.sender()
        if who is not None:
            if who.objectName() == 'Slider_dmin' or who.objectName() == 'Slider_dmax':
                if ic.Slider_dmax.value() > ic.Slider_dmin.value():
                    if self.slider_dmax_value == ic.Slider_dmax.value() and self.slider_dmin_value == ic.Slider_dmin.value():
                        updateplotflag = False
                    else:
                        self.slider_dmax_value = ic.Slider_dmax.value()
                        self.slider_dmin_value = ic.Slider_dmin.value()
                        self.update_drange_sliders(who)
                        if self.plot_settings['colorNorm']:
                            updateplotflag = False

                else:
                    updateplotflag = False
                    ic.Slider_dmax.setValue(self.slider_dmax_value)
                    ic.Slider_dmin.setValue(self.slider_dmin_value)
            elif who.objectName() == 'Slider_alpha':
                self.alpha = ic.Slider_alpha.value() / 100.
                ic.Slider_alpha_GBox.setTitle(
                    'alpha  --> {:.2f}'.format(self.alpha))
        print('plot_upgrade mark 1 {:.4f}'.format(time.time() - t0))
        t0 = time.time()
        self.parent.parent_widget.statusbar.showMessage("Update Plot!")
        if updateplotflag:
            # if who is not None:
            #     print(who.objectName())
            #     if who.objectName() not in ['control_displayMode']:
            #         self.setPlotData()
            if self.plot_settings['dmode'] == 'tconts':
                dmax, dmin = self.drange_normalize(
                    data_in=[ic.Slider_dmax.value(), ic.Slider_dmin.value()],
                    reverse=True)
                if who.objectName() in ['control_displayMode']:
                    self.plot_conts(dmax=dmax, dmin=dmin)
                elif who.objectName() in ['control_colors_list']:
                    levels = [dmin, dmax]
                    naxis = self.plot_settings['slicers_axisinfo']['NAXIS'][self.plot_settings['dim_idx']]
                    cmap = cm.get_cmap(self.plot_settings['cmap'])
                    for s in range(naxis):
                        self.im[s].set_cmap(colors.ListedColormap([cmap(float(s + 1) / naxis)] * len(levels)))
                elif who.objectName() in ['Slider_alpha']:
                    alpha = self.plot_settings['alpha']
                    naxis = self.plot_settings['slicers_axisinfo']['NAXIS'][self.plot_settings['dim_idx']]
                    for s in range(naxis):
                        for c in self.im[s].collections:
                            c.set_alpha(alpha)
                elif who.objectName() in ['Slider_dmin', 'Slider_dmax', 'control_colorBar_check', 'control_wcs_check']:
                    try:
                        for cnts in self.im:
                            for cnt in cnts.collections:
                                cnt.remove()
                    except:
                        pass
                    self.plot_conts(dmax=dmax, dmin=dmin)
                else:
                    pass
            else:
                # self.im.set_array(self.img_data.ravel())
                self.im.set_array(self.img_data)
                self.im.axes.autoscale(False, 'both', True)
                self.im.axes.autoscale_view(True, True, True)
                self.im.axes.relim(visible_only=True)
                self.im.set_norm(self.imshow_args['norm'])
                self.im.set_cmap(self.imshow_args['cmap'])
                self.im.set_alpha(self.imshow_args['alpha'])
            zoom_fov = self.parent.parent_widget.navi_toolbar.zoom_fov
            if zoom_fov:
                self.axes.set_xlim(zoom_fov[:2])
                self.axes.set_ylim(zoom_fov[2:])
            self.axes.figure.canvas.draw_idle()
        print('plot_upgrade mark 2 {:.4f}'.format(time.time() - t0))

    def drange_normalize(self, drange=None, data_in=None, reverse=False):
        ## normalize the data dynamic range to 1~100
        '''
        :param drange:
        :param data_in:
        :param reverse: Set False if the input is the real data, set True if the input is normalized value
        :return:
        '''
        if drange is None:
            drange = self.drange
        dmin, dmax = drange
        if data_in is None:
            data_in = np.array(drange)
        else:
            data_in = np.array(data_in)
        if reverse:
            atten_func = lambda x: x / 100. * (dmax - dmin) + dmin
        else:
            atten_func = lambda x: (x - dmin) / (dmax - dmin) * 100.
        data_out = atten_func(data_in)
        return data_out

    def setPlotData(self):
        t0 = time.time()
        if self.getDmode() == 'tconts':
            self.img_data = self.img_slicer.data.copy()
            if self.mapdict == {}:
                img_data_2D = self.img_slicer.get_slice().copy()
                self.spmap = smap.Map((img_data_2D, self.fits_data[self.fits_current_field_id].header))
            else:
                self.spmap = self.mapdict['submc'][self.idx]
        else:
            self.img_data = self.img_slicer.get_slice().copy()
            if self.mapdict == {}:
                if self.fits_data[self.fits_current_field_id].header['TELESCOP'] == 'SDO/AIA':
                    self.spmap = smap.Map((self.img_data, self.fits_data[self.fits_current_field_id].header))
                    self.spmap = utils.normalize_aiamap(self.spmap)
                    self.img_data = self.spmap.data
                else:
                    self.spmap = smap.Map((self.img_data, self.fits_data[self.fits_current_field_id].header))
            else:
                self.spmap = self.mapdict['submc'][self.idx]
        print('setPlotData mark 1 {:.4f}'.format(time.time() - t0))

    def setPlotDataXY(self):
        t0 = time.time()
        self.dataxy_image = utils.map2wcsgrids(self.spmap, cell=False)
        self.dataxy_conts = utils.map2wcsgrids(self.spmap)
        print('setPlotDataXY mark 1 {:.4f}'.format(time.time() - t0))

    def updatePlotSettings(self):
        # print('updatePlotData')
        t0 = time.time()

        ic = self.image_control

        if self.plot_settings['abs']:
            self.img_data = np.abs(self.img_data)

        if self.plot_settings['dmode'] == 'imgs':
            img_rotation = self.plot_settings['img_rotation']
            if (img_rotation > 0):
                self.img_data = np.rot90(self.img_data, img_rotation)

        try:
            vmax, vmin = self.drange_normalize(
                data_in=[ic.Slider_dmax.value(), ic.Slider_dmin.value()],
                reverse=True)
        except:
            vmin = ic.Slider_dmin.value()
            vmax = ic.Slider_dmax.value()
        if self.plot_settings['log']:
            lowcut = 1.0
            if vmin < lowcut:
                vmin = lowcut
            self.norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if self.plot_settings['colorNorm']:
                self.norm = None  # colors.Normalize(vmin=np.nanmin(self.img_data), vmax=np.nanmax(self.img_data))
            else:
                self.norm = colors.Normalize(vmin=vmin, vmax=vmax)

        self.imshow_args = {'cmap': cm.get_cmap(self.plot_settings['cmap']), 'norm': self.norm,
                            'alpha': ic.Slider_alpha.value() / 100., 'origin': 'lower', 'interpolation': 'nearest',
                            'extent': [self.dataxy_conts[0][0][0], self.dataxy_conts[0][0][-1],
                                       self.dataxy_conts[-1][0][0],
                                       self.dataxy_conts[-1][-1][0]]}

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
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        print('updatePlotData mark 3 {:.4f}'.format(time.time() - t0))

    def getDmode(self):
        cdm = self.image_control.control_displayMode
        if cdm.currentText() == 'Images':
            self.dmode = 'imgs'
        elif cdm.currentText().startswith('Transparent Contours'):
            self.dmode = 'tconts'
        else:
            self.dmode = 'imgs'
        return self.dmode

    def image_controlEventManager(self):
        ic = self.image_control
        who = ic.sender()
        if who is not None:
            print(who.objectName())
            if who.objectName() == 'control_colors_list':
                self.plot_settings['cmap'] = ic.control_colors_list.currentText()
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'control_log_check':
                self.plot_settings['log'] = ic.control_log_check.isChecked()
                self.setPlotData()
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'control_abs_check':
                self.plot_settings['abs'] = ic.control_abs_check.isChecked()
                self.setPlotData()
                self.updatePlotSettings()

                self.plot_update()
            elif who.objectName() == 'control_colorNorm_check':
                self.plot_settings['colorNorm'] = ic.control_colorNorm_check.isChecked()
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'control_colorBar_check':
                self.plot_settings['colorBar'] = ic.control_colorBar_check.isChecked()
                self.toggle_colorbar(self.plot_settings['colorBar'])
            elif who.objectName() == 'control_dataField':
                self.setData()
            elif who.objectName() == 'control_displayMode':
                self.setPlotData()
                self.getDmode()
                self.plot()
                self.parent.layermanger.setZorders()
            elif who.objectName() == 'rotate_dial':
                self.plot_settings['img_rotation'] = ic.rotate_dial.value()
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'Slider_dmin':
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'Slider_dmax':
                self.updatePlotSettings()
                self.plot_update()
            elif who.objectName() == 'Slider_alpha':
                self.updatePlotSettings()
                self.plot_update()
            else:
                pass

    def image_control_connect(self):
        ic = self.image_control

        # --- assign control actions/signals
        ic.control_colors_list.activated.connect(self.image_controlEventManager)
        ic.control_log_check.stateChanged.connect(self.image_controlEventManager)
        ic.control_abs_check.stateChanged.connect(self.image_controlEventManager)
        ic.control_colorNorm_check.stateChanged.connect(self.image_controlEventManager)
        ic.control_colorBar_check.stateChanged.connect(self.image_controlEventManager)

        # --- Data field selector
        ic.control_dataField.activated.connect(self.image_controlEventManager)

        # --- Display mode selector
        ic.control_displayMode.activated.connect(self.image_controlEventManager)

        # --- Rotate dial
        ic.rotate_dial.valueChanged.connect(self.image_controlEventManager)

        # --- Data drange sliders
        ic.Slider_dmin.valueChanged.connect(self.image_controlEventManager)
        ic.Slider_dmax.valueChanged.connect(self.image_controlEventManager)

        # --- layer alpha slider
        ic.Slider_alpha.valueChanged.connect(self.image_controlEventManager)

    def image_control_disconnect(self):
        ic = self.image_control

        ## disconnect all signals
        try:
            ic.control_colors_list.disconnect()
            ic.control_log_check.disconnect()
            ic.control_abs_check.disconnect()
            ic.control_colorNorm_check.disconnect()
            ic.control_colorBar_check.disconnect()
            ic.control_dataField.disconnect()
            ic.control_displayMode.disconnect()
            ic.rotate_dial.disconnect()
            ic.Slider_dmin.disconnect()
            ic.Slider_dmax.disconnect()
            ic.Slider_alpha.disconnect()
        except:
            pass

    # def image_control_show(self):
    #     # self.image_control_connect()
    #     self.image_control.show()

    def init_alpha_slider(self):
        ic = self.image_control
        ic.Slider_alpha_GBox.setTitle(
            'alpha  --> {:.2f}'.format(1.))
        ic.Slider_alpha.setMinimum(0)
        ic.Slider_alpha.setMaximum(100)
        ic.Slider_alpha.setSingleStep(1)
        ic.Slider_alpha.setValue(100)
        self.alpha = 1.

    def init_drange_sliders(self, drange=None, dmaxvalue=None, dminvalue=None):
        dmin, dmax = self.drange_normalize(drange=drange)
        dmaxvalue_org = copy.copy(dmaxvalue)
        dminvalue_org = copy.copy(dminvalue)
        if dmaxvalue is None:
            dmaxvalue = dmax
            dmaxvalue_org = dmax
        else:
            dmaxvalue = self.drange_normalize(drange, dmaxvalue)
        if dminvalue is None:
            dminvalue = dmin
            dminvalue_org = dmin
        else:
            dminvalue = self.drange_normalize(drange, dminvalue)
        dstep = (dmax - dmin)
        ic = self.image_control
        ic.Slider_dmin_GBox.setTitle(
            'dmin [{0:.2e}~{1:.2e}] --> {2:.2e}'.format(drange[0], drange[1], dminvalue_org))
        ic.Slider_dmin.setMinimum(dmin)
        ic.Slider_dmin.setMaximum(dmax)
        ic.Slider_dmin.setSingleStep(dstep)
        ic.Slider_dmin.setValue(dminvalue)
        self.slider_dmin_value = dminvalue

        ic.Slider_dmax_GBox.setTitle(
            'dmax [{0:.2e}~{1:.2e}] --> {2:.2e}'.format(drange[0], drange[1], dmaxvalue_org))
        ic.Slider_dmax.setMinimum(dmin)
        ic.Slider_dmax.setMaximum(dmax)
        ic.Slider_dmax.setSingleStep(dstep)
        ic.Slider_dmax.setValue(dmaxvalue)
        self.slider_dmax_value = dmaxvalue

    def reset_drange_sliders(self, drange=None):
        ic = self.image_control
        dmin, dmax = self.drange_normalize(drange=drange)
        if self.plot_settings['dmode'] == 'tconts':
            ddiff = dmax - dmin
            dmin = dmin + 0.1 * ddiff
        ic.Slider_dmax.disconnect()
        ic.Slider_dmin.disconnect()
        ic.Slider_dmax.setValue(dmax)
        ic.Slider_dmin.setValue(dmin)
        ic.Slider_dmin.valueChanged.connect(self.plot_update)
        ic.Slider_dmax.valueChanged.connect(self.plot_update)
        self.slider_dmax_value = dmax
        self.slider_dmin_value = dmin
        return dmin, dmax

    def update_drange_sliders(self, who):
        drange = self.drange
        who.parent().setTitle(
            '{} [{:.2e}~{:.2e}] --> {:.2e}'.format(who.objectName()[-4:], drange[0], drange[1],
                                                   self.drange_normalize(data_in=[who.value()][0], reverse=True)))

    def clear_plot(self):
        if self.im:
            if type(self.im) is list:
                for col in self.im:
                    if isinstance(col, QuadContourSet):
                        # col.collections.clear()
                        for c in col.collections:
                            c.remove()
                    else:
                        col.remove()
            else:
                self.im.remove()
        try:
            self.im_cbar.remove()
        except:
            pass
        self.axes.figure.canvas.draw_idle()

    def destroy(self):
        self.image_control.close()
        self.image_control.destroy()
        try:
            self.clear_plot()
        except:
            pass

    @property
    def info(self):

        ic = self.image_control
        ## image settings
        self.log = ic.control_log_check.isChecked()
        self.abs = ic.control_abs_check.isChecked()
        self.colorNorm = ic.control_colorNorm_check.isChecked()
        self.colorBar = ic.control_colorBar_check.isChecked()
        ccl = ic.control_colors_list
        self.cmap = ccl.currentText()
        self.img_rotation = ic.rotate_dial.value()
        self.alpha = ic.Slider_alpha.value() / 100.

        ## Display Mode
        self.getDmode()

        ## slicer
        self.slicers_axisinfo = self.img_slicer.slicers_axisinfo
        if hasattr(self.img_slicer, 'combos'):
            dim_idx = copy.copy(self.img_slicer.slicers_axisinfo['DIM_IDX'])
            for ll in self.img_slicer.combos:
                dim_idx.pop(dim_idx.index(ll.currentIndex()))
            self.dim_idx = dim_idx[0]
        else:
            self.dim_idx = 0
        return vars(self)

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)


class Layer_Limb(Layer_Line_Base):

    def __init__(self, layer_id: 'layer_id', parent_obj: 'parent_obj', axes: 'axes') -> 'Layer_Limb_class':
        super(Layer_Limb, self).__init__(layer_id, parent_obj, axes)
        self.linestyle = 'solid'
        self.image_control = ImageControl_Limb()
        ic = self.image_control
        ic.control_linestyleSelector.setCurrentText(self.linestyle)
        ic.Slider_alpha_GBox.setTitle('alpha  --> {:.2f}'.format(self.alpha))
        ic.Slider_linewidth_GBox.setTitle('line width  --> {:.2f}'.format(self.linewidth))
        ic.Slider_linewidth.setValue(self.linewidth * 20)
        # --- assign control actions/signals
        ic.control_colorSelector.clicked.connect(self.plot_update)
        # --- line style selector
        ic.control_linestyleSelector.activated.connect(self.plot_update)
        # --- line width sliders
        ic.Slider_linewidth.valueChanged.connect(self.plot_update)
        # --- layer alpha slider
        ic.Slider_alpha.valueChanged.connect(self.plot_update)

    def plot(self):
        if self.im:
            for col in self.im:
                for cl in col:
                    cl.set_visible(True)
        else:
            ic = self.image_control
            zorder = max([_.zorder for _ in self.axes.get_children()]) + 1000
            rsun = self.parent.spmap.rsun_obs
            phi = np.linspace(-180, 180, num=181) * u.deg
            x = np.cos(phi) * rsun
            y = np.sin(phi) * rsun
            linestyle = ic.linestyles[ic.control_linestyleSelector.currentText()]
            self.im += self.axes.plot(x, y, linewidth=self.linewidth, alpha=self.alpha, color=self.color,
                                      linestyle=linestyle, zorder=zorder)

    def plot_update(self):
        ic = self.image_control
        who = ic.sender()
        if who is not None:
            if who.objectName() == 'control_colorSelector':
                self.color = self.color_picker()
                for im in self.im:
                    im.set_color(self.color)
            elif who.objectName() == 'control_linestyleSelector':
                linestyle = ic.linestyles[ic.control_linestyleSelector.currentText()]
                for im in self.im:
                    im.set_linestyle(linestyle)
            elif who.objectName() == 'Slider_linewidth':
                self.linewidth = ic.Slider_linewidth.value() / 20.
                ic.Slider_linewidth_GBox.setTitle('line width  --> {:.2f}'.format(self.linewidth))
                for im in self.im:
                    im.set_linewidth(self.linewidth)
            elif who.objectName() == 'Slider_alpha':
                self.alpha = ic.Slider_alpha.value() / 100.
                ic.Slider_alpha_GBox.setTitle(
                    'alpha  --> {:.2f}'.format(self.alpha))
                for im in self.im:
                    im.set_alpha(self.alpha)
        self.axes.figure.canvas.draw_idle()

    def hide_plot(self):
        if self.im:
            for col in self.im:
                for cl in col:
                    cl.set_visible(False)

    def get_zorder(self):
        if self.im:
            if type(self.im) is list:
                if isinstance(self.im[0], QuadContourSet):
                    self.zorder = self.im[0].collections[0].get_zorder()
                else:
                    self.zorder = self.im[0].get_zorder()
            else:
                self.zorder = self.im.get_zorder()
        return self.zorder

    def color_picker(self):
        color = QColorDialog.getColor()
        self.image_control.control_colorSelector.setStyleSheet("QWidget { background-color: %s}" % color.name())
        print("QWidget { background-color: %s}" % color.name())
        return color.name()

    @property
    def info(self):
        return vars(self)

    def destroy(self):
        pass

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)


class Layer_Grid(Layer_Limb):
    def __init__(self, layer_id: 'layer_id', parent_obj: 'parent_obj', axes: 'axes') -> 'Layer_Grid_class':
        super(Layer_Grid, self).__init__(layer_id, parent_obj, axes)
        self.linestyle = 'dashed'
        self.gridspacing = 15.
        self.image_control = ImageControl_Grid()
        ic = self.image_control
        ic.control_linestyleSelector.setCurrentText(self.linestyle)
        ic.Slider_alpha_GBox.setTitle('alpha  --> {:.2f}'.format(self.alpha))
        ic.Slider_linewidth_GBox.setTitle('line width  --> {:.2f}'.format(self.linewidth))
        ic.Slider_linewidth.setValue(self.linewidth * 20)
        ic.Slider_gridspacing_GBox.setTitle('grid spacing  --> {:.0f} deg'.format(self.gridspacing))
        ic.Slider_gridspacing.setValue(self.gridspacing)
        # --- assign control actions/signals
        ic.control_colorSelector.clicked.connect(self.plot_update)
        # --- line style selector
        ic.control_linestyleSelector.activated.connect(self.plot_update)
        # --- line width sliders
        ic.Slider_linewidth.valueChanged.connect(self.plot_update)
        # --- layer alpha slider
        ic.Slider_alpha.valueChanged.connect(self.plot_update)
        # --- layer grid spacing slider
        ic.Slider_gridspacing.valueChanged.connect(self.plot_update)

    def plot(self, grid_spacing=None):
        if grid_spacing is None:
            grid_spacing = self.gridspacing * u.deg
        if self.im:
            for col in self.im:
                for cl in col:
                    cl.set_visible(True)
        else:
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

            dsun = self.parent.spmap.dsun
            rsun = self.parent.spmap.rsun_meters

            b0 = self.parent.spmap.heliographic_latitude.to(u.deg)
            l0 = self.parent.spmap.heliographic_longitude.to(u.deg)
            hg_longitude_deg = np.linspace(-90, 90, num=91) * u.deg
            hg_latitude_deg = np.arange(0, 90, grid_spacing.to(u.deg).value)
            hg_latitude_deg = np.hstack([-hg_latitude_deg[1:][::-1], hg_latitude_deg]) * u.deg
            zorder = max([_.zorder for _ in self.axes.get_children()]) + 1000
            for lat in hg_latitude_deg:
                c = hgs2hcc(rsun, hg_longitude_deg, lat * np.ones(91), b0, l0)
                coords = hcc2hpc(c[0], c[1], c[2], dsun)
                self.im += self.axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), linewidth=self.linewidth,
                                          alpha=self.alpha, color=self.color, linestyle=self.linestyle, zorder=zorder)

            hg_longitude_deg = np.arange(0, 90, grid_spacing.to(u.deg).value)
            hg_longitude_deg = np.hstack([-hg_longitude_deg[1:][::-1], hg_longitude_deg]) * u.deg
            hg_latitude_deg = np.linspace(-90, 90, num=91) * u.deg

            for lon in hg_longitude_deg:
                c = hgs2hcc(rsun, lon * np.ones(91), hg_latitude_deg, b0, l0)
                coords = hcc2hpc(c[0], c[1], c[2], dsun)
                self.im += self.axes.plot(coords[0].to(u.arcsec), coords[1].to(u.arcsec), linewidth=self.linewidth,
                                          alpha=self.alpha, color=self.color, linestyle=self.linestyle, zorder=zorder)

    def plot_update(self):
        ic = self.image_control
        who = ic.sender()
        if who is not None:
            if who.objectName() == 'control_colorSelector':
                self.color = self.color_picker()
                for im in self.im:
                    im.set_color(self.color)
            elif who.objectName() == 'control_linestyleSelector':
                linestyle = ic.linestyles[ic.control_linestyleSelector.currentText()]
                for im in self.im:
                    im.set_linestyle(linestyle)
            elif who.objectName() == 'Slider_linewidth':
                self.linewidth = ic.Slider_linewidth.value() / 20.
                ic.Slider_linewidth_GBox.setTitle('line width  --> {:.2f}'.format(self.linewidth))
                for im in self.im:
                    im.set_linewidth(self.linewidth)
            elif who.objectName() == 'Slider_alpha':
                self.alpha = ic.Slider_alpha.value() / 100.
                ic.Slider_alpha_GBox.setTitle(
                    'alpha  --> {:.2f}'.format(self.alpha))
                for im in self.im:
                    im.set_alpha(self.alpha)
            elif who.objectName() == 'Slider_gridspacing':
                self.gridspacing = ic.Slider_gridspacing.value()
                ic.Slider_gridspacing_GBox.setTitle(
                    'grid spacing  --> {:.0f} deg'.format(self.gridspacing))
                for im in self.im:
                    im.remove()
                self.im = []
                self.plot(grid_spacing=self.gridspacing * u.deg)

        self.axes.figure.canvas.draw_idle()


class LayerManager:
    def __init__(self, parent=None):
        super(LayerManager, self).__init__()
        self.parent = parent
        self.axes = self.parent.parent_widget.figure.axes
        self.fig = self.parent.parent_widget.figure.fig
        self.List = LayerList()
        self.image_layers = defaultdict(list)
        self.parent.parent_widget.LayerLayout.addWidget(self.List)
        self.parent.parent_widget.LayerLayout.setContentsMargins(0, 0, 2, 1)
        self.curr_layer = '0'
        self.zorders = []

        self.image_layers['solarlimb'] = Layer_Limb('solarlimb', None, self.axes)
        item = QStandardItem('Limb')
        item.setCheckable(True)
        item.setEditable(False)
        item.setDragEnabled(False)
        item.setDropEnabled(False)
        item.setCheckState(Qt.Unchecked)
        item.setWhatsThis('solarlimb')
        self.List.model.appendRow(item)

        self.image_layers['solargrid'] = Layer_Grid('solargrid', None, self.axes)
        item = QStandardItem('Grid')
        item.setCheckable(True)
        item.setEditable(False)
        item.setDragEnabled(False)
        item.setDropEnabled(False)
        item.setCheckState(Qt.Unchecked)
        item.setWhatsThis('solargrid')
        self.List.model.appendRow(item)

        self.List.model.itemChanged.connect(self.checkStateChange)
        self.List.doubleClicked.connect(self.doubleClicked)

    def getZorders(self):
        self.zorders = []
        for k, v in self.image_layers.items():
            if isinstance(v, Layer_Image):
                self.zorders.append(v.get_zorder())
        return self.zorders

    @timetest
    def setZorders(self):
        layers = []
        model = self.List.model
        for idx in range(model.rowCount()):
            layer_id = model.item(idx).whatsThis()
            if isinstance(self.image_layers[layer_id], Layer_Image):
                layers.append(self.image_layers[layer_id])
        self.zorders = np.arange(len(layers))[::-1] + 1
        for idx, ly in enumerate(layers):
            ly.set_zorder(self.zorders[idx])
        self.fig.canvas.draw_idle()

    def countImageLayer(self):
        count = 0
        for k, v in self.image_layers.items():
            if isinstance(v, Layer_Image):
                count += 1
        return count

    def layerTitle(self, header):
        telescope = header['TELESCOP']
        if telescope == 'SDO/AIA':
            h = {'DATATYPE': 'AIA{:.0f}A'.format(header['WAVELNTH']),
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        elif telescope == 'EOVSA':
            h = {'DATATYPE': 'EOVSA{:.2f}GHz'.format(header['RESTFRQ'] / 1e9),
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        else:
            h = {'DATATYPE': header['TELESCOP'],
                 'DATE-OBS': Time(header['DATE-OBS']).isot}
        return h

    def register(self, url=None):
        if url is str:
            url = [url]
        self.curr_layer = str(self.countImageLayer())
        ## initial the layer
        layer = self.image_layers[self.curr_layer] = Layer_Image(self.curr_layer, url, 0, self.parent, self.axes)
        ## read the data
        layer.readfits(url[0])
        print('register layer: {}'.format(self.curr_layer))
        ## initial the image_control widget settings
        ic = layer.image_control
        ## set the datafield
        ic.control_dataField.clear()  # always remove items first
        ic.control_dataField.addItems(layer.fits_fields)
        self.parent.parent_widget.actionDisplay_Header.setEnabled(True)
        self.parent.parent_widget.actionDisplay_Header.triggered.connect(self.parent.DisplayHeader)
        ## set the data
        layer.setData()
        ## set the display mode
        ic.control_displayMode.clear()
        control_displaymode = ['Images']
        for cidx, c_dm in enumerate(layer.img_slicer.slicers_axisinfo['CTYPE'][:-2]):
            control_displaymode.append('Transparent Contours - Dim {}'.format(c_dm))
        ic.control_displayMode.addItems(control_displaymode)
        ## set the telescope
        layer.telescope = layer.img_slicer.header['TELESCOP']
        ## set the enabled color maps
        layer.setColorMapList()
        ## set the plot data
        layer.setPlotData()
        ## set the plot dynamic range
        layer.drange = [np.nanmin(layer.img_data), np.nanmax(layer.img_data)]
        if layer.telescope == 'SDO/AIA':
            clrange = utils.sdo_aia_scale_dict(
                wavelength=layer.fits_data[layer.fits_current_field_id].header['WAVELNTH'])
        else:
            clrange = {'low': None, 'high': None, 'log': False}
        ## set the image control widget settings
        ic.control_log_check.setChecked(clrange['log'])
        layer.init_drange_sliders(layer.drange, dmaxvalue=clrange['high'],
                                  dminvalue=clrange['low'])
        layer.init_alpha_slider()

        ## add the layer to the QListView widget
        layer.layer_title = self.layerTitle(layer.img_slicer.header)
        ic.setWindowTitle(layer.layer_title['DATATYPE'])
        item = QStandardItem('    '.join(layer.layer_title.values()))
        item.setCheckable(True)
        item.setDragEnabled(True)
        item.setDropEnabled(False)
        item.setCheckState(Qt.Checked)
        item.setWhatsThis(self.curr_layer)
        self.List.model.insertRow(int(self.curr_layer), item)
        self.List.image_layers = self.image_layers

        ## update the layer
        layer.plot_settings = layer.info

        ## connect control signals
        layer.image_control_connect()

        ## plot the solar grid and limb
        if self.curr_layer == '0':
            self.axes.cla()
            # set the layer to be checked
            # embed()
            for itemname in ['Limb', 'Grid']:
                item = self.List.model.findItems(itemname)[0]
                self.image_layers[item.whatsThis()].parent = layer
                if not item.checkState():
                    item.setCheckState(Qt.Checked)
                else:
                    self.image_layers[item.whatsThis()].im = []
                    self.toggleIM(True, self.image_layers[item.whatsThis()])

        ## plot the layer
        layer.setPlotDataXY()
        layer.updatePlotSettings()
        layer.plot()
        self.setZorders()

    # @timetest
    def doubleClicked(self, item):
        layer_id = item.model().item(item.row()).whatsThis()
        print(item.row(), layer_id, item.data())
        # if layer_id not in ['solargrid', 'solarlimb']:
        layer = self.image_layers[layer_id]
        layer.image_control.show()

    def checkStateChange(self, item):
        print(item.text(), item.checkState())
        # embed()
        layer_id = item.model().item(item.row()).whatsThis()
        layer = self.image_layers[layer_id]
        self.toggleIM(item.checkState(), layer)
        if layer_id not in ['solargrid', 'solarlimb']:
            print('zorder,', layer.get_zorder())
            print('zorders', self.getZorders())
            self.setZorders()
        # embed()

    def toggleIM(self, state=False, layer=None):
        # print('toggle_im', layer.im, layer)
        if layer.im:
            s = bool(state)
            if type(layer.im) is list:
                for im in layer.im:
                    if isinstance(im, QuadContourSet):
                        for col in im.collections:
                            col.set_visible(s)
                    else:
                        im.set_visible(s)
            else:
                layer.im.set_visible(s)
            if layer.im_cbar:
                layer.im_cbar.ax.set_visible(s)
        else:
            layer.plot()
        self.fig.canvas.draw_idle()

    def delete(self, layer_id=None):
        print('move-----1')
        pass
