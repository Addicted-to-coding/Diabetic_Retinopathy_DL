import copy
import random
import numpy as np
import math

def get_shuffled_data(X, y):

	X2 = copy.deepcopy(X)
	y2 = copy.deepcopy(y)

	arr = list(zip(X2, y2))
	random.shuffle(arr)

	unzipped = list(zip(*arr))
	X2, y2 = np.array(unzipped[0]), np.array(unzipped[1])

	return X2, y2

def convert_y(y, C = 5):

	tmp = np.zeros((y.shape[0], C))
	for i in range(tmp.shape[0]):
		tmp[i][y[i]] = 1
	return tmp

def add_mixup_data(X, y):

	# X will be of the form (N, 3, 160, 160)
	# y will be of the form (N, 1)

	# Need to convert y into (N, C) where C = 5
	y = convert_y(y, C = 5)

	X2, y2 = get_shuffled_data(X, y)
	X_mix = []
	y_mix = []

	for i in range(X.shape[0]):
		Xmix, ymix = mix_up(X[i], y[i], X2[i], y2[i])
		X_mix.append(np.expand_dims(Xmix, axis = 0))
		y_mix.append(np.expand_dims(ymix, axis = 0))

	X_mix = np.concatenate(X_mix)
	y_mix = np.concatenate(y_mix)

	X_res = np.concatenate([X, X_mix])
	y_res = np.concatenate([y, y_mix])

	return X_res, y_res

def add_cutmix_data(X, y):

	y = convert_y(y, C = 5)
	X2, y2 = get_shuffled_data(X, y)
	
	X_cutmix = []
	y_cutmix = []

	for i in range(X.shape[0]):
		Xcutmix, ycutmix = cutmix(X[i], X2[i], y[i], y2[i])
		X_cutmix.append(np.expand_dims(Xcutmix, axis = 0))
		y_cutmix.append(np.expand_dims(ycutmix, axis = 0))

	X_cutmix = np.concatenate(X_cutmix)
	y_cutmix = np.concatenate(y_cutmix)

	X_res = np.concatenate([X, X_cutmix])
	y_res = np.concatenate([y, y_cutmix])

	return X_res, y_res


def cutmix(X1,X2,y1,y2, alpha = 1):

    C = X1.shape[0]
    W = X1.shape[2]
    H = X1.shape[1]

    lamb = np.random.beta(alpha, alpha)
    r_x = math.floor(np.random.uniform(0,W))
    r_y = math.floor(np.random.uniform(0,H))

    r_w = math.floor(math.sqrt(1-lamb) * W)
    r_h = math.floor(math.sqrt(1-lamb) * H)
    if r_x + r_w > W:
        r_w = W - r_x
    if r_y + r_h > H:
        r_h = H - r_y

    X = copy.deepcopy(X1)
    mask = X2[:, r_x:r_x+r_w, r_y:r_y+r_h]
    X[:, r_x:r_x+r_w, r_y:r_y+r_h] = mask

    y = lamb*y1 + (1-lamb)*y2
    
    return X, y

def mix_up(X1,y1,X2,y2, alpha=0.3):

    l = np.random.beta(alpha, alpha)

    images = (X1 * l) + (X2 * (1 - l))
    labels = (y1 * l) + (y2 * (1 - l))

    return images, labels