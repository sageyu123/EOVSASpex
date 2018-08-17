from abc import ABCMeta, abstractmethod

class Layer_Line_Base(object):

    __metaclass__ = ABCMeta

    def __init__(self, layer_id: 'layer_id', parent_obj: 'parent_obj', axes: 'axes') -> 'Layer_Line_class':
        self.parent = parent_obj
        self.axes = axes
        self.alpha = 1.0
        self.linewidth = 0.5
        self.linestyle = 'solid'
        self.layer_id = layer_id
        self.layer_name = 'layer{}'.format(layer_id)
        self.layer_title = None
        self.color = '#FFFFFF'
        self.im = []
        self.im_cbar = None
        self.image_control = None
        self.zorder = 0

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def hide_plot(self):
        pass

    @abstractmethod
    def get_zorder(self):
        pass

    @abstractmethod
    def color_picker(self):
        pass

    @abstractmethod
    def plot_update(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)

    @property
    def info(self):
        return vars(self)




class Layer_Image_Base:

    __metaclass__ = ABCMeta

    def __init__(self, layer_id: 'layer_id', url: 'layer_url', frame_idx: 'frame_idx', parent_obj: 'parent_obj',
                 axes: 'axes') -> 'Layer_Image_class':
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
        self.color = 0
        self.cmap = None
        self.cmaps = []
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
        self.image_control = None
        self.divider = None
        self.ax_cb = None
        self.img_data = None
        self.zorder = 0

    @abstractmethod
    def get_zorder(self):
        pass

    @abstractmethod
    def set_zorder(self):
        pass

    @abstractmethod
    def readfits(self):
        pass

    @abstractmethod
    def updatedata_from_widget(self):
        pass

    @abstractmethod
    def updateColormapList(self):
        pass

    @abstractmethod
    def plot_colorbar(self):
        pass

    @abstractmethod
    def toggle_colorbar(self):
        pass

    @abstractmethod
    def plot_image(self):
        pass

    @abstractmethod
    def plot_conts(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def plot_update(self):
        pass

    @abstractmethod
    def drange_normalize(self):
        pass

    @abstractmethod
    def set_plot_data(self):
        pass

    @abstractmethod
    def widget2layer(self):
        pass

    @abstractmethod
    def image_control_connect(self):
        pass

    @abstractmethod
    def image_control_disconnect(self):
        pass

    @abstractmethod
    def init_alpha_slider(self):
        pass

    @abstractmethod
    def init_drange_sliders(self):
        pass

    @abstractmethod
    def reset_drange_sliders(self):
        pass

    @abstractmethod
    def update_drange_sliders(self):
        pass

    @abstractmethod
    def clear_plot(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    @property
    def info(self):
        return vars(self)

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)