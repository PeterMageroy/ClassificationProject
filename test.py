import numpy as np

classes = [7, 7, 7, 9, 7, 7, 9]
classes = np.array(classes)
i = 7
mask = (classes == i)
nums = np.array([1, 2, 3, 4, 5, 6, 7])
print(mask)
print(nums[mask])