import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

r = 500
x3 = []
y3 = []
for i in range(0, 360):
    x3.append(r*math.cos(i/360*math.pi))
    y3.append(r*math.sin(i/360*math.pi)+500)

x4 = []
y4 = []
for i in range(360, 0, -1):
    x4.append(r*math.cos(-i/360*math.pi))
    y4.append(r*math.sin(-i/360*math.pi)-500)

y1 = np.arange(-500, 500, 0.01)
y2 = np.arange(-500, 500, 0.01)
# print(y1)
x1 = np.zeros_like(y1) + 500
x2 = np.zeros_like(y2) - 500

x = np.concatenate([x4,x1,x3,x2])
y = np.concatenate([y4, y1, y3,y2])
print(x)
def get_centroid(x,y):
    centroid = (sum(x)/len(x),sum(y)/len(y))
    return centroid
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def createLane(x, y,r_ = 1.01):
    cent = get_centroid(x, y)
    x_ = []
    y_ = []
    for i in range(len(x)):
        x_.append(x[i]-cent[0])
        y_.append(y[i]-cent[1])
    rad = []
    theta = []
    for i in range(len(x_)):
        r, phi = cart2pol(x_[i], y_[i])
        rad.append(r)
        theta.append(phi)
    # Scale the radius here
    rad_ = []
    for r in rad:
        rad_.append(r*r_)

    X = []
    Y = []
    for i in range(len(rad_)):
        X_, Y_ = pol2cart(rad_[i], theta[i])
        X.append(X_+cent[0])
        Y.append(Y_+cent[1])
    return np.array(X), np.array(Y)

def getWindow(mapp, x, y, size = 192):
    mapp = mapp[
    x-size//2:x+size//2,
    y-size//2:y+size//2]
    return mapp

# mapp = np.zeros((1500, 3000))
# X_ = x.astype(int)
# Y_ = y.astype(int)
# THICKNESS = 5
# for i in range(len(X_)):
#     mapp[X_[i]+750: X_[i]+750+THICKNESS, Y_[i]+1500: Y_[i]+1500+THICKNESS] = 1

# x_out, y_out = createLane(x, y, r_=1.05)
# X_out = x_out.astype(int)
# Y_out = y_out.astype(int)
# print(X_out-X_)
# for i in range(len(X_out)):
#     mapp[X_out[i]+750: X_out[i]+750+THICKNESS, Y_out[i]+1500: Y_out[i]+1500+THICKNESS] = 1

# x_in, y_in = createLane(x, y, r_=0.95)
# X_in = x_in.astype(int)
# Y_in = y_in.astype(int)
# print(X_in-X_)
# for i in range(len(X_in)):
#     mapp[X_in[i]+750: X_in[i]+750+THICKNESS, Y_in[i]+1500: Y_in[i]+1500+THICKNESS] = 1

# # for i in range(len(X_)):
# #     for j in range(THICKNESS):
# #         mapp[X_[i]+750+j, Y_[i]+1500+j] = 1
# # for i in range(len(X_)):
# #     for j in range(THICKNESS):
# #         mapp[X_[i]+750+j, Y_[i]+1500+j] = 1
# fig = plt.figure(figsize=(15, 30))
# # plt.imshow(mapp, cmap='gray')
# # np.save("lane.npy", mapp)

# # #road
# THICKNESS = 10
# for i in range(len(X_in)):
#     mapp[X_in[i]+750-THICKNESS: X_out[i]+750+THICKNESS, Y_in[i]+1500-THICKNESS: Y_out[i]+1500+THICKNESS] = 1
#     mapp[X_out[i]+750-THICKNESS: X_in[i]+750+THICKNESS, Y_in[i]+1500-THICKNESS: Y_out[i]+1500+THICKNESS] = 1
#     mapp[X_in[i]+750-THICKNESS: X_out[i]+750+THICKNESS, Y_out[i]+1500-THICKNESS: Y_in[i]+1500+THICKNESS] = 1
#     mapp[X_out[i]+750-THICKNESS: X_in[i]+750+THICKNESS, Y_out[i]+1500-THICKNESS: Y_in[i]+1500+THICKNESS] = 1
# plt.imshow(mapp, cmap='gray')
# np.save("road.npy", mapp)

lane = np.load("lane.npy")
road = np.load("road.npy")
img = np.array([lane, road, np.ones_like(lane)])
print(img.shape)
# plt.imshow(img.transpose())
# plt.show()

# fig = plt.figure(figsize=(15, 30))
# plt.imshow(img.transpose())
# plt.plot(x+750, y+1500)
# plt.show()
# print(x)
for i in range(1):
    # print((getWindow(img.transpose(), int(x[i]+750), int(y[i]+1500))).shape)
    # plt.imshow(getWindow(img.transpose(), int(x[0]+750), int(y[0]+1500)))
    # print(int(x[0])+ 192//2+750)
    # plt.imshow(img.transpose()[int(x[i])- 192//2+750:int(x[i])+192//2+750, int(x[1])-192//2+1500:int(x[1]) + 192//2+1500])
    # plt.show()
    # break
    print(int(x[i]+750), int(y[i])+1500)
    print(img.transpose().shape)
    img = getWindow(img.transpose(), int(x[i]+750), int(y[i])+1500)
    img = cv2.circle(img, (97, 97), 5, (255, 0, 0))
    cv2.imshow('image', img)
    
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # plt.imshow(getWindow(img.transpose(), int(x[i]), int(y[i])))
    # plt.show()