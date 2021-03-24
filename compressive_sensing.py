import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.signal as sps
from cv2 import cv2
from sklearn.linear_model import Lasso
from numpy import r_
from numpy import pi, cos, sqrt
import warnings
import math
import time
from PIL import Image
import random

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
def img_read(file_name, filter):
    img = cv2.imread(file_name)
    if filter == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def dct2(x):
    return spfft.dct(spfft.dct(x, norm='ortho', axis = 0), norm='ortho', axis = 1)
def idct2(x):
    return spfft.idct(spfft.idct(x, norm='ortho', axis = 0), norm='ortho', axis = 1)
def plot_fig(var, color, title, xlabel, ylabel, file_name, colorbar):
    plt.figure()
    csfont = {'fontname': 'Times New Roman'}
    vmin = 0
    vmax = np.max(var)
    plt.axis('off')
    plt.imshow(var, cmap = color, vmin = vmin, vmax = vmax)
    if colorbar:
        plt.colorbar()
    plt.title(title, **csfont)
    if xlabel:
        xlabel(xlabel, **csfont)
    if ylabel:
        ylabel(ylabel, **csfont)
    plt.xticks(fontname = 'Times New Roman')
    plt.yticks(fontname = 'Times New Roman')
    if file_name:
        plt.savefig(str("{}.png".format(file_name)), dpi = 300)
    # plt.show()
def MSE_plot(MSEs, file_name, ylabel):
    plt.figure()
    markers = ['^', 'o']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    mpl.rc('font',family='Arial')
    patches = []
    rates = []
    for k, cv_method in enumerate(MSEs):
        for i, blur_m in enumerate(cv_method):
            for j, rate in enumerate(blur_m):
                plt.scatter([100 * s for s in sampling_rates], np.log10(rate), s = 100, marker = markers[i], color = colors[j], zorder = 2, alpha = .5, edgecolor = 'k')
                for r in rate:
                    rates.append(np.log10(r))
                if k == 0:
                    plt.plot([100 * s for s in sampling_rates], np.log10(rate), 'k-', zorder = 1, linewidth = 1)
                else:
                    plt.plot([100 * s for s in sampling_rates], np.log10(rate), 'k:', zorder = 1, linewidth = 1)
    
    for i, block_size in enumerate(block_sizes):
        patch = mpatches.Patch(facecolor = colors[i], edgecolor='k', label = str('{} x {} Block Size'.format(block_size, block_size)))
        patches.append(patch)
    patches.append(Line2D([0], [0], color='k', lw=2, label='K_folds', ls = '--'))
    patches.append(Line2D([0], [0], color='k', lw=2, label='Random', ls = ':'))
    star = plt.scatter([], [], color = 'k', marker = markers[0], label = 'Median Filter')
    plus = plt.scatter([], [], color = 'k', marker = markers[1], label = 'No Median Filter')
    plt.xlabel("Sampling Rate (%)")
    plt.ylabel(ylabel)
    patches.append(star)
    patches.append(plus)
    yint = range(math.floor(min(rates)), math.ceil(max(rates))+1)
    plt.yticks(yint)
    plt.legend(handles = patches, loc='upper left', bbox_to_anchor=(1.05, 1))
    # plt.show()
    plt.savefig(str("{}.png".format(file_name)), dpi = 300, bbox_inches='tight')
def DCT_val(x,y,u,v, P, Q):
    if u == 1:
        alpha = sqrt(1/P)
    else:
        alpha = sqrt(2/P)
    if v == 1:
        beta = sqrt(1/Q)
    else:
        beta = sqrt(2/Q)
    val = alpha * beta * cos((pi * (2 * x - 1) * (u - 1)) / (2 * P)) * cos((pi * (2 * y - 1) * (v - 1))/(2 * Q))
    return val
def DCT_matrix(P, Q):
    DCT = np.zeros((P ** 2, Q ** 2))
    i, j = -1, -1
    for x in range(1, P+1):
        for y in range(1, Q+1):
            i += 1
            for u in range(1, P+1):
                for v in range(1, Q+1):
                    j += 1
                    DCT[i, j % (Q ** 2)] = DCT_val(x, y, u, v, P, Q)
    return DCT
def sample_block(block, block_size, sample_rate):
    C = block.reshape(block_size ** 2, 1)
    sample_indices = np.random.choice(block_size ** 2, sample_rate, replace = False)
    return C, sample_indices
def l1_minimization(A, b, lambd, block_size):
    lasso = Lasso(alpha = lambd, max_iter = 1e4, fit_intercept=False)
    lasso.fit(A, b)
    x1d = np.array(lasso.coef_)
    try:
        x2d = x1d.reshape(block_size, block_size)
        return x2d
    except:
        print("Couldn't solve system (1 x 1 Error)")
def approximate_pixels(alpha2d, block_size):
    return idct2(alpha2d).reshape(block_size ** 2, 1)
def calc_MSE(y, yhat):
    return np.mean((y - yhat) ** 2)
def kfolds_cv(sample_indices, C, T, lambd, block_size):
    MSE = 0
    n_test_pnts = math.floor(len(sample_indices)/4)
    if n_test_pnts == 0:
        n_test_pnts = 1
    for fold in range(4):
        test_indices = sample_indices[fold * n_test_pnts:(fold + 1) * n_test_pnts]
        train_indices = np.setdiff1d(sample_indices, test_indices)
        B = C[train_indices, :]
        A = T[train_indices, :]
        alpha2d = l1_minimization(A, B, lambd, block_size)
        if alpha2d is not None:
            predicted_coeffs = approximate_pixels(alpha2d, block_size)
            MSE += calc_MSE(C[test_indices, :], predicted_coeffs[test_indices, :])
        else:
            break
    return MSE, lambd, predicted_coeffs
def random_cv(sample_indices, C, T, lambd, block_size):
    MSE = 0
    n_test_pnts = math.floor(len(sample_indices)/6)
    if n_test_pnts == 0:
        n_test_pnts = 1
    for _ in range(20):
        test_indices = np.random.choice(sample_indices, n_test_pnts)
        train_indices = np.setdiff1d(sample_indices, test_indices)
        B = C[train_indices, :]
        A = T[train_indices, :]
        alpha2d = l1_minimization(A, B, lambd, block_size)
        if alpha2d is not None:
            predicted_coeffs = approximate_pixels(alpha2d, block_size)
            MSE += calc_MSE(C[test_indices, :], predicted_coeffs[test_indices, :])
        else:
            break
    return MSE, lambd, predicted_coeffs
def compressive_sensing_ch(img, sz, sample, lambd_samples, median_f):
    try:
        img.shape[0]//sz - img.shape[0]/sz > 1e-5 or img.shape[1]//sz - img.shape[1]/sz > 1e-5
    except:
        print("Block size will result in uneven blocks")
    lambd_arr = np.zeros((img.shape[0]//sz, img.shape[1]//sz))
    MSE_arr = np.zeros((img.shape[0]//sz, img.shape[1]//sz))
    lambd_values = np.logspace(l_min, l_max, lambd_samples)
    S = round(sz ** 2 * sample)
    T = DCT_matrix(sz, sz)
    recon = np.zeros_like(img)
    MSE_total = 0
    for i in r_[:img.shape[0]:sz]:
        for j in r_[:img.shape[1]:sz]:
            block = img[i:(i+sz), j:(j+sz)]
            C, S_idx = sample_block(block, sz, S)
            MSEs, lambds, preds = [], [], []
            for lambd in lambd_values:
                if cv_method == 'Kfolds':
                    MSE, lambd, pred = kfolds_cv(S_idx, C, T, lambd, sz)
                if cv_method == 'random':
                    MSE, lambd, pred = random_cv(S_idx, C, T, lambd, sz)
                MSEs.append(MSE)
                lambds.append(lambd)
                preds.append(pred.reshape(sz, sz))
            min_idx = min(range(len(MSEs)), key=MSEs.__getitem__)
            MSE_total += MSEs[min_idx]
            recon[i:(i+sz), j:(j+sz)] = preds[min_idx]
            lambd_arr[i//sz, j//sz] = lambds[min_idx]
            MSE_arr[i//sz, j//sz] = MSEs[min_idx]
    if median_f == 'blur':
        recon = sps.medfilt2d(recon)
        med = np.median(recon[recon > 0])
        recon[recon == 0] = med
        MSE_total = calc_MSE(recon.flatten(), img.flatten())
    return MSE_total, recon, lambd_arr, MSE_arr
def driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    warnings.filterwarnings("ignore")
    img = img_read(str("{}.bmp".format(file_name)), img_color)
    filters = ['blur', 'noBlur']
    MSE_list_, time_list_ = [], []
    for k, median_filtering in enumerate(filters):
        MSE_list, time_list = [], []
        time_list = []
        for i, block_size in enumerate(block_sizes):
            MSEs, times = [], []
            for j, rate in enumerate(sampling_rates):
                start = time.time()
                if img_color == 'RGB':
                    reconstruction = np.zeros_like(img)
                    MSE_ = np.zeros((img.shape[0] // block_size, img.shape[1] // block_size))
                    MSE_sim = 0
                    for ch in range(3):
                        MSE, recon, lambd_arr, MSE_arr = compressive_sensing_ch(img[:, :, ch], block_size, rate, lambdas, median_filtering)
                        MSE_ = MSE_ + MSE_arr
                        reconstruction[:, :, ch] = recon
                        MSE_sim = MSE_sim + MSE
                else:
                    MSE_sim, reconstruction, lambd_arr, MSE_ = compressive_sensing_ch(img, block_size, rate, lambdas, median_filtering)
                end = time.time()
                f_name = str("lam_arr_{}x{}_S{}_{}_{}_{}".format(block_size, block_size, int(rate * 100), median_filtering, cv_method, file_name))
                plot_fig(np.log10(lambd_arr), 'jet', None, None, None, f_name, True)
                f_name = str("rec_{}x{}_S{}_{}_{}_{}".format(block_size, block_size, int(rate * 100), median_filtering, cv_method, file_name))
                plot_fig(reconstruction, 'gray', None, None, None, f_name, None)
                f_name = str("MSE_arr_{}x{}_S{}_{}_{}_{}".format(block_size, block_size, int(rate * 100), median_filtering, cv_method, file_name))
                plot_fig(np.log10(MSE_), 'jet', None, None, None, f_name, True)
                MSEs.append(MSE_sim)
                times.append(end - start)
                print("Simulation", (k * len(block_sizes) * len(sampling_rates)) + (i * len(sampling_rates)) + j + 1 , "completed")
            MSE_list.append(MSEs)
            time_list.append(times)
        MSE_list_.append(MSE_list)
        time_list_.append(time_list)
    MSE_plot(MSE_list_, str("{}_{}_MSE".format(file_name, cv_method)), "$\log_{10}$ MSE")
    MSE_plot(time_list_, str("{}_{}_times".format(file_name, cv_method)), "$\log_{10}$ Simulation Time (s)")
if __name__ == '__main__':
    ## Lena
    file_name = 'lena'
    block_sizes = [4, 8, 16, 32]
    sampling_rates = [.1, .15, .25, .5, .9, .99]
    l_min = -6
    l_max = -6
    lambdas = 100
    cv_method = 'random'
    img_color = 'RGB'
    driver()
  
    ## Fishing boat
    file_name = 'fishing_boat'
    block_sizes = [4, 8]
    sampling_rates = [.25, .5, .9, .99]
    l_min = -6
    l_max = -6
    lambdas = 1
    cv_method = 'Kfolds'
    img_color = 'gray'
    driver()