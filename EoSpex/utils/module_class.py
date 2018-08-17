from abc import ABCMeta, abstractmethod


class Viewer_Module(object):
    """ A prototype for Viewer module

    Attributes:
        file_formats: An array of string (file formats typical for this module)
        ui_file: A string with name of ui_file for this module
    """

    __metaclass__ = ABCMeta

    name = "No Name"
    file_extensions = ['empty']
    ui_file = "empty.ui"
    parent_widget = None

    def __init__(self):
        pass

    @abstractmethod
    def module_info(self):
        """Returning a string representing short info about module"""
        pass

    @abstractmethod
    def open_file(self):
        """Opens data file"""
        pass

    @abstractmethod
    def set_plot_data(self):
        """Sets plot data"""
        pass

    @abstractmethod
    def plot(self):
        """Sets plot widget"""
        pass

    @abstractmethod
    def update_plot(self):
        """Updates plot widget without changing axes"""
        pass

    @abstractmethod
    def update_data(self):
        """Update data object module"""
        pass

    @abstractmethod
    def get_data_slice(self):
        """Gets data slice"""
        pass

    @abstractmethod
    def show_gui(self):
        """Shows gui elements used by module"""
        pass

    @abstractmethod
    def hide_gui(self):
        """Hides gui elements used by module"""
        pass

    @abstractmethod
    def clean_gui(self):
        """Clean gui elements used by module"""
        pass
