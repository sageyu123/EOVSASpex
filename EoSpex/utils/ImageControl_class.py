from PyQt5 import QtWidgets,QtCore,uic
from PyQt5.QtWidgets import QColorDialog
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from IPython import embed

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



class ImageControl_Line(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ImageControl_Line, self).__init__(parent)
        self.parent_widget = parent

        ## --- add linestype to the list
        self.linestyles = OrderedDict(
            [('solid', (0, ())),
             ('loosely dotted', (0, (1, 10))),
             ('dotted', (0, (1, 5))),
             ('densely dotted', (0, (1, 1))),

             ('loosely dashed', (0, (5, 10))),
             ('dashed', (0, (5, 5))),
             ('densely dashed', (0, (5, 1))),

             ('loosely dashdotted', (0, (3, 10, 1, 10))),
             ('dashdotted', (0, (3, 5, 1, 5))),
             ('densely dashdotted', (0, (3, 1, 1, 1))),

             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


class ImageControl_Limb(ImageControl_Line):
    ui_file = "ImageControl_Limb.ui"

    def __init__(self):
        super(ImageControl_Limb, self).__init__()
        self.ui = uic.loadUi(os.path.split(__file__)[0] + '/../ui/' + self.ui_file, self)

        self.setWindowTitle('Limb')
        self.control_linestyleSelector.addItems(list(self.linestyles.keys()))



class ImageControl_Grid(ImageControl_Line):
    ui_file = "ImageControl_Grid.ui"

    def __init__(self):
        super(ImageControl_Grid, self).__init__()
        self.ui = uic.loadUi(os.path.split(__file__)[0] + '/../ui/' + self.ui_file, self)

        self.setWindowTitle('Grid')
        self.control_linestyleSelector.addItems(list(self.linestyles.keys()))
        # self.control_linestyleSelector.setCurrentText('dashed')




