import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)

array = np.array([1, 2, 3])
print("NumPy array:", array)

plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Matplotlib Test")
plt.show()