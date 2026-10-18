import sys, os
f = open("d:/MajorProject-MeMOTR/data/IOTD/check/diag.txt", "w", encoding="utf-8")
f.write("A. python alive\n")
f.flush()

try:
    import numpy
    f.write("B. numpy ok\n")
    f.flush()
except Exception as e:
    f.write(f"B. numpy err: {e}\n")
    f.flush()

try:
    import cv2
    f.write("C. cv2 ok\n")
    f.flush()
except Exception as e:
    f.write(f"C. cv2 err: {e}\n")
    f.flush()

try:
    import torch
    f.write("D. torch ok\n")
    f.flush()
except Exception as e:
    f.write(f"D. torch err: {e}\n")
    f.flush()

try:
    import torchvision
    f.write("E. torchvision ok\n")
    f.flush()
except Exception as e:
    f.write(f"E. torchvision err: {e}\n")
    f.flush()

f.write("ALL OK\n")
f.close()
