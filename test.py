import numpy as np

hsv = np.arange(0, 27).reshape(3, 3, 3)
h = hsv[:, :, 0]
print(hsv)
print("h:{0}".format(h))
