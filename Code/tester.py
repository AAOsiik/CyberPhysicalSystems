import time                         # Needed to set dynamic q-table filenames
import pickle                       # Save and load q table
import numpy as np                  # Array types of stuff
from PIL import Image               #
import cv2                          # OpenCV
import matplotlib.pyplot as plt     # Plotting
from matplotlib import style        # make graph pretty
from time import sleep

full_error_avg = [ [] for n in range(5)]
print(full_error_avg)
x = [1,2,3]

full_error_avg[0] = x
print(full_error_avg)