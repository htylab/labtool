import ctypes
lb = ctypes.CDLL('T1map.dll')
lib = ctypes.WinDLL(None, handle=lb._handle)
# Access the mandelbrot function.
T1map = lib.T1map

from numpy.ctypeslib import ndpointer
# Define the types of the output and arguments of this function.
T1map.restype = None
T1map.argtypes = [ndpointer(ctypes.c_double), ndpointer(ctypes.c_double),
                       ndpointer(ctypes.c_double)]


import numpy as np
# We initialize an empty array.

#TI= 120 220 370 1130 1168 1233 2115 2125 2145 3078 4035

#SI= 114 87 56 75 80 89 137 132 128 151 168
t=np.array([120.2, 220.3, 370.1, 1130.5, 1168, 1233, 2115, 2125, 2145, 3078, 4035],dtype=np.float64)
y=np.array([114.2, 87., 56., 75., 80., 89., 137., 132.4, 128., 151, 168.5],dtype=np.float64)
col = np.empty((1,11), dtype=np.double)
# We execute the C function, which will update the array.
T1map(t, y, col)
temp=col.copy()
from ctypes.wintypes import HMODULE
ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
ctypes.windll.kernel32.FreeLibrary(lb._handle)
print temp[0,0:9]