from PyQt5 import QtWidgets
from PyQt5 import uic
import matplotlib.pyplot as plt
import os

class ImageControl(QtWidgets.QWidget):
    ui_file = "ImageControl.ui"

    def __init__(self, parent=None):
        super(ImageControl, self).__init__(parent)
        self.parent_widget = parent
        self.ui = uic.loadUi(os.path.split(__file__)[0] + '/../ui/' + self.ui_file, self)
        # --- set default states
        self.control_colorBar_check.setChecked(True)
        self.control_log_check.setChecked(False)
        self.control_abs_check.setChecked(False)

        # --- add colors to the list

        self.control_colors_list.clear()
        self.control_colors_list.addItems(plt.colormaps())
        # embed()

    # def closeEvent(self, event):
    #     ## disconnect all signals
    #     try:
    #         self.control_colors_list.disconnect()
    #         self.control_log_check.disconnect()
    #         self.control_abs_check.disconnect()
    #         self.control_colorNorm_check.disconnect()
    #         self.control_colorBar_check.disconnect()
    #         self.control_dataField.disconnect()
    #         self.control_displayMode.disconnect()
    #         self.rotate_dial.disconnect()
    #         self.Slider_dmin.disconnect()
    #         self.Slider_dmax.disconnect()
    #         self.control_SolarGrid_check.disconnect()
    #         self.control_SolarLimb_check.disconnect()
    #         self.destroy()
    #     except:
    #         pass