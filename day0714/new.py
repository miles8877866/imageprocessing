# -*- coding: utf-8 -*-
import os
import numpy as np
from matplotlib.font_manager import fontManager
fonts = [font.name for font in fontManager.ttflist if
         os.path.exists(font.fname) and
         os.stat(font.fname).st_size>1e6]
print("中文可用字型")
for font in np.unique(fonts):
    print(font)
    print("==="*10)