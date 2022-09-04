import torch
import cv2
import torch.nn as nn
import numpy as np

# a = np.arange(255*20*20).reshape(255, 20, 20)
# # A = torch.tensor(a)
# A = torch.tensor(a).expand(1, 255, 20, 20)
# # print(A)
# print(A.shape)
# B = A.view(1, 3, 85, 20, 20)
# print(B.shape)
# C = B.permute(0, 1, 3, 4, 2)     #[1, 2, 4, 3]
# print(C.shape)
# c = B.permute(0, 1, 4, 3, 2)     #[1, 2, 4, 3]
# print(c.shape)
# D = C.contiguous()
# print(D.shape)
# # ——————————————————————
# # print(A)
# # print(A.shape)
# # b = torch.cat((A, A), 1)
# # print(b)
# # print(b.shape)
flag = torch.cuda.is_available()
print(flag)
path = r'D:\MingJiahui\MyProject\yolov5\yolov5\data\images\bus.jpg'
img = cv2.imread(path)
print(img.shape)
IMG = torch.from_numpy(img)
IMG.to("cuda:0") / 255
print(torch.version.cuda)
