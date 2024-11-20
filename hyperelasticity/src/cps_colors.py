import numpy as np


CPS_COLORS = np.array([
    [246, 163, 21],
    [67, 83, 132],
    [194, 76, 76],
    [22, 164, 138],
    [104, 143, 198],
    [204, 204, 204],
]) / 255

CPS_COLORS = list(map(tuple, CPS_COLORS))

# CPS_COLORMAP = ListedColormap(CPS_COLORS, name="CPS_Colormap")