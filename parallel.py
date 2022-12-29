# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:24:40 2022

@author: KASI VISWANATH
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def parallel_upper(x1,y1,x2,y2,d):
    s1 = (y2-y1)/(x2-x1)
    c1 = y2-s1*x1
    xm = (x1+x2)/2
    ym = (y1+y2)/2
    s2 = -1/s1
    c2 = ym - s2*xm
    
    A = s1
    B = -1
    
    x = (d*math.sqrt(A**2+B**2) + c1 - B*c2)/-(A+B*s2)
    y = s2*x + c2
    
    return x,y

def parallel_lower(x1,y1,x2,y2,d):
    s1 = (y2-y1)/(x2-x1)
    c1 = y2-s1*x1
    xm = (x1+x2)/2
    ym = (y1+y2)/2
    s2 = -1/s1
    c2 = ym - s2*xm
    
    A = s1
    B = -1
    
    x = (d*math.sqrt(A**2+B**2) - c1 - B*c2)/(A+B*s2)
    y = s2*x + c2
    
    return x,y

r = 5
fx = []
gx = []
fy = []
gy = []
for i in range(360):
    x1 = r*math.cos(i/360*math.pi)
    y1 = r*math.sin(i/360*math.pi)
    
    x2 = r*math.cos((i+1)/360*math.pi)
    y2 = r*math.sin((i+1)/360*math.pi)
    
    x,y = parallel_upper(x1,y1,x2,y2,1)
    fx.append(x1)
    gx.append(x)
    fy.append(y1)
    gy.append(y)

plt.scatter(fx,fy, marker = 'x')
plt.scatter(gx,gy,marker = 'x')
plt.show()
    