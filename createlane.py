import numpy as np
import matplotlib.pyplot as plt
import utm
import geopy.distance

# coords_1 = (19, 75)
# coords_2 = (19, 75.01)

# print(geopy.distance.geodesic(coords_1, coords_2).m)

def get_centroid(x,y):
    centroid = (sum(x)/len(x),sum(y)/len(y))
    return centroid

x=[0,15,10,0]
y=[0,0,15,10]
plt.plot(x,y)
cent = get_centroid(x, y)

x_ = []
y_ = []
for i in range(len(x)):
    x_.append(x[i]-cent[0])
    y_.append(y[i]-cent[1])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

rad = []
theta = []
for i in range(len(x_)):
    r, phi = cart2pol(x_[i], y_[i])
    rad.append(r)
    theta.append(phi)

rad_ = []
for r in rad:
    rad_.append(r*0.5)

X = []
Y = []
for i in range(len(rad_)):
    X_, Y_ = pol2cart(rad_[i], theta[i])
    X.append(X_+cent[0])
    Y.append(Y_+cent[1])

plt.plot(X,Y)
plt.plot(x,y)
plt.show()

