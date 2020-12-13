import glob

import numpy as np
from PIL import Image
import kmeans
from EM import FBSModel
import copy
import xlsxwriter
from sklearn import mixture

def get_model_init():
    for fn in sorted(glob.glob(path + 'model_init.png')):
        im = Image.open(fn).convert("RGB")
        model_init = (np.array(im, dtype=np.float64) / 255.0)[:, :, 0]
    return model_init

def get_means(ti):
    means = []
    means.append(np.mean(ti[0:9][0:9], (0, 1)))
    i_middle = len(ti) // 2
    i = slice(i_middle - 5, i_middle + 5)
    j_middle = len(ti[0]) // 2
    j = slice(j_middle - 5, j_middle + 5)
    means.append(np.mean(ti[i, j], (0, 1)))
    return means

def get_precisions_init(ti):
    precisions_init = []
    precisions_init.append(np.linalg.inv(np.cov(np.reshape(ti[0:9][0:9], (-1, 3)).T) + np.eye(3, 3) * 0.000001))
    i_middle = len(ti) // 2
    i = slice(i_middle - 5, i_middle + 5)
    j_middle = len(ti[0]) // 2
    j = slice(j_middle - 5, j_middle + 5)
    precisions_init.append(np.linalg.inv(np.cov(np.reshape(ti[i, j], (-1, 3)).T) + np.eye(3, 3) * 0.000001))
    return precisions_init

def fit_EM_model(tm, path, model_init):
    em = FBSModel(samples=tm, pu_si_1=model_init)
    old_ui = np.zeros(shape=(np.shape(em.ui)))
    old_theta = copy.deepcopy(em.gaussians)
    i = 0
    threshold = 0.01
    max_iter = 50
    while (em.diff_ui(old_ui) > threshold or em.diff_theta(old_theta) > threshold) and i < max_iter:
        print('Epoch #' + str(i))
        print('Difference in ui: ' + str(em.diff_ui(old_ui)))
        print('Difference in theta: ' + str(em.diff_theta(old_theta)))
        old_ui = em.ui.copy()
        old_theta = copy.deepcopy(em.gaussians)
        em.eStep()
        em.mStep()
        i = i + 1

    return em

def get_EM_shape_from_model(model, model_init):
    n, m = np.shape(model_init)
    classif_em = np.zeros(shape=(n, m, 3))
    for i in range(len(model.ui)):
        for j in range(len(model.ui[i])):
            classif_em[i][j] = np.repeat(model.pu_si[1][i][j], 3)
    model_image = Image.fromarray(np.uint8(classif_em*255))
    model_image.save('final_model.png')
    return np.round(classif_em[:, :, 0])

def get_EM_shape_from_image():
    for fn in sorted(glob.glob('final_model.png')):
        im = Image.open(fn).convert("RGB")
        classif_em = (np.array(im, dtype=np.float64) / 255.0)[:, :, 0]
    return np.round(classif_em)

# =========================================================
# Load image samples (250x289 images with 3 channels - RGB)
# =========================================================
path = '../data/'
tm = []
filenames = sorted(glob.glob(path + 'hand_[0-9][0-9].png'))
for fn in filenames:
    im = Image.open(fn).convert("RGB")
    arr = np.array(im, dtype=np.float64) / 255.0
    tm.append(arr)
tm = np.array(tm)

testLabels = []
for fn in sorted(glob.glob(path + 'hand_[0-9][0-9]_seg.png')):
    im = Image.open(fn).convert("RGB")
    arr = np.array(im, dtype=np.float64) / 255.0
    labels = np.zeros(shape=(np.size(arr, 0), np.size(arr, 1)))
    labels[(arr == [1, 1, 1])[:, :, 0]] = 1
    testLabels.append(labels)
testLabels = np.array(testLabels)


workbook = xlsxwriter.Workbook('Tables.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'EM-Algorithm (shape model)')
worksheet.write(0, 1, 'EM-Algorithm (alphas)')
worksheet.write(0, 2, 'K-Means')
worksheet.write(0, 3, 'Mixture of Gaussians')

model_init = get_model_init()
em_model = fit_EM_model(tm, path, model_init)
classif_em = get_EM_shape_from_model(em_model, model_init)
# classif_em = get_EM_shape_from_image()

# =========================================================
# K-means & Mixture of Gaussians classification
# =========================================================
i = 0
for ti in tm:
    n, m, channels = np.shape(ti)
    means = get_means(ti)
    precisions_init = get_precisions_init(ti)

    # K-means classif
    memberships = kmeans.classify(ti, means, 100)

    # Mixture models classif
    ti_gm = np.reshape(ti, (-1, 3))
    gm = mixture.GaussianMixture(n_components=2, means_init=means, precisions_init=precisions_init, max_iter=100)
    gm.fit(ti_gm)
    classif_gm = gm.predict(ti_gm)
    classif_gm = np.reshape(classif_gm, (n, m))

    # EM alphas classif
    classif_alphas = em_model.alpha_classif_for_image(i)
    model_image = Image.fromarray(np.uint8(classif_alphas * 255))
    model_image.save('classif' + ('0' if i < 10 else '') + str(i) + '.png')

    errorEM = np.size(classif_em[classif_em != testLabels[i]]) / np.size(classif_em)
    errorAlphas = np.size(classif_alphas[classif_alphas != testLabels[i]]) / np.size(classif_alphas)
    errorKmeans = np.size(memberships[memberships != testLabels[i]]) / np.size(memberships)
    errorMixture = np.size(classif_gm[classif_gm != testLabels[i]]) / np.size(classif_gm)
    print('EM (shape model): ' + str(errorEM))
    print('EM (alphas): ' + str(errorAlphas))
    print('K-means: ' + str(errorKmeans))
    print('Mixture of Gaussians: ' + str(errorMixture))
    i = i + 1
    worksheet.write(i, 0, errorEM)
    worksheet.write(i, 1, errorAlphas)
    worksheet.write(i, 2, errorKmeans)
    worksheet.write(i, 3, errorMixture)

workbook.close()
