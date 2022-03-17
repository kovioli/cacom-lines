import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis,opening
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import os
import json



def curve_fit(path, pred, binary_threshold= 0.2, degree = 4):
    """Fits a polynomial of a defined degree the data.
    It is assumed that the data contains the upper aponeurosis at pred[:,:,0]
    and the lower aponeurosis respectively at pred[:,:,1].
    Else use curve_fit_2.

    This function creates a JSON file, containing the info dict.

    input:
        path to npy array of the prediction [pred]
        binary_threshold:   threshold of pixel value which are considered
                            for the binary image

    output:
        info [dict]:    containing the coefficients/intercept to the polynomial,
                        as well as the x and y values of the polynomial,
                        ready to plot


    Author: Maresa Fees
    """
    #filename = os.path.basename(path)[:-4]
    filename = path
    img_npy = pred
    img_npy_upper = img_npy[:,:,0]
    img_npy_lower = img_npy[:,:,1]

    img_bin_upper = np.where(img_npy_upper > binary_threshold, 1, 0)
    img_bin_lower = np.where(img_npy_lower > binary_threshold, 1, 0)

    # morphological operators (opening,medial_axis)
    # to "clean up" the data
    # distance can be used to keep track of the width
    opened_upper = opening(img_bin_upper)
    opened_lower = opening(img_bin_lower)
    skel_upper, distance_upper = medial_axis(opened_upper, return_distance=True)
    skel_lower, distance_lower = medial_axis(opened_lower, return_distance=True)

    # Get coordinates of pixels and sort
    X_u = np.argwhere(skel_upper)
    X_u = X_u[X_u[:, 1].argsort()]

    X_l = np.argwhere(skel_lower)
    X_l = X_l[X_l[:, 1].argsort()]

    x_u = X_u[:,1].reshape(-1, 1)
    y_u = X_u[:,0]

    x_l = X_l[:,1].reshape(-1, 1)
    y_l = X_l[:,0]

    fig, ax = plt.subplots()
    ax.set_ylim([0, img_npy.shape[1]])
    ax.set_xlim([0, img_npy.shape[0]])
    plt.gca().invert_yaxis()

    ax.scatter(x_u, y_u, s=1, c='orange', label= "upper_prediction")
    ax.scatter(x_l, y_l, s=1, c='blue', label= "lower_prediction")

    model1 = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model1.fit(x_u, y_u)
    y_plot1 = model1.predict(x_u)
    ax.plot(x_u, y_plot1, 'r',label = f"upper_{degree}th-poly")

    model2 = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model2.fit(x_l, y_l)
    y_plot2 = model2.predict(x_l)
    ax.plot(x_l, y_plot2, 'g', label = f"lower_{degree}th-poly")

    plt.legend()
    plt.title(f"{filename}")
    plt.savefig(f"{filename}.pdf")

    #plt.show()
    print(model1.named_steps.ridge.coef_)

    info = {
        "filename": filename,
        "size image": img_npy.shape,
        "upper": {
                "coefficients": model1.named_steps.ridge.coef_.tolist(),
                "intercept": model1.named_steps.ridge.intercept_,
                "curve_x_values": x_u.tolist(),
                "curve_y_values": y_plot1.tolist()
        },
        "lower": {
            "coefficients": model2.named_steps.ridge.coef_.tolist(),
            "intercept": model2.named_steps.ridge.intercept_,
            "curve_x_values": x_l.tolist(),
            "curve_y_values": y_plot2.tolist()
        }
    }

    with open(f'{filename}.json', 'w') as fp:
        json.dump(info, fp)

    return info




def curve_fit_2(path, binary_threshold = 0.2, degree = 4):
    """Function which fits a polynomial to the data
    input:
        path to numpy array
        binary_threshold:   threshold of pixel value which are considered
                            the binary image
        degree:             degree of polynomial to be fitted

    output:
            info [dict]:    containing the coefficients/intercept to the polynomial,
                            as well as the x and y values of the polynomial,
                            ready to plot

    Author: Maresa Fees
    """

    filename = os.path.basename(path)[:-4]
    img_npy = np.load(path)
    img_bin = np.where(img_npy > binary_threshold, 1, 0 )

    # morphological operators (opening,medial_axis)
    # to "clean up" the data
    # distance can be used to keep track of the width
    opened = opening(img_bin)
    skel, distance = medial_axis(opened, return_distance=True)

    # Get coordinates of pixels and sort
    X = np.argwhere(skel)
    X = X[X[:, 1].argsort()]


    # sort the data into upper aponeurosis and lower aponeurosis
    upp_y = np.array([])
    upp_x = np.array([])

    low_y = np.array([])
    low_x = np.array([])

    for i in range(len(X[:,1])):
        v = X[i,1]
        values = np.where(X[:,1] == v,X[:,0],np.nan)


        # two thresholds to improve the differentiation between upper and lower
        # if only one value exists --> upper ( assumption: upper longer then lower)
        # if the difference is smaller then 3 pixel --> it belongs to one --> upper
        if np.count_nonzero(~np.isnan(values)) >1 and np.nanmax(values)-np.nanmin(values) > 3:
            low = np.nanmax(values)
            upp = np.nanmin(values)

            low_y = np.append(low_y, low)
            low_x = np.append(low_x, v)
        else:
            upp = np.nanmin(values)
            low = np.nan
        # print(v,upp)
        upp_y = np.append(upp_y, upp)
        upp_x = np.append(upp_x, v)


    fig, ax = plt.subplots()
    ax.set_ylim([0, img_npy.shape[1]])
    ax.set_xlim([0, img_npy.shape[0]])
    plt.gca().invert_yaxis()

    ax.scatter(upp_x, upp_y, s=1, c='orange', label= "upper_prediction")
    ax.scatter(low_x, low_y, s=1, c='blue', label= "lower_prediction")



    upp_x = upp_x.reshape(-1, 1)
    low_x = low_x.reshape(-1, 1)

    model1 = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model1.fit(upp_x, upp_y)
    y_plot1 = model1.predict(upp_x)
    ax.plot(upp_x, y_plot1, 'r',label = f"upper_{degree}th-poly")

    model2 = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model2.fit(low_x, low_y)
    y_plot2 = model2.predict(low_x)
    ax.plot(low_x, y_plot2, 'g', label = f"lower_{degree}th-poly")

    plt.legend()
    plt.title(f"{filename}")
    plt.savefig(f"{filename}.pdf")

    plt.show()

    info = {
        "filename": filename,
        "size image": img_npy.shape,
        "upper": {
                "coefficients": model1.named_steps.ridge.coef_,
                "intercept": model1.named_steps.ridge.intercept_,
                "curve_x_values": upp_x,
                "curve_y_values": y_plot1
        },
        "lower": {
            "coefficients": model2.named_steps.ridge.coef_,
            "intercept": model2.named_steps.ridge.intercept_,
            "curve_x_values": low_x,
            "curve_y_values": y_plot2
        }
    }

    return info