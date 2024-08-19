import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


class MplColorHelper:
    def __init__(self, cmap_name, start_val=0.0, stop_val=1.0):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)
