import numpy as np
from gmvd import GaussMVD as gmvd

# =========================================================
# Foreground / Background / Shape Image Model
# =========================================================
class FBSModel:
    # ===============================================
    def __init__(self, samples, pu_si_1):
        self.samples = samples
        self.pu_si = [1 - pu_si_1, pu_si_1]
        self.ui = [np.log(pu_si_1) - np.log(1 + pu_si_1)]
        self.alpha = self.pu_si.copy()
        self.gaussians = [[gmvd() for i in range(len(samples))], [gmvd() for i in range(len(samples))]]
        # ptheta_xi[si = {0, 1}, l = 0..m, i = 0..250, i = 0..289]
        self.ptheta_xi = [[self.gaussians[0][i].compute_probs(samples[i]) for i in range(len(samples))],
                          [self.gaussians[1][i].compute_probs(samples[i]) for i in range(len(samples))]]
        sum_ptheta_si = np.sum(self.ptheta_xi, 0)
        self.ptheta_xi[0] = np.divide(self.ptheta_xi[0], sum_ptheta_si)
        self.ptheta_xi[1] = np.divide(self.ptheta_xi[1], sum_ptheta_si)

    # ===============================================
    def eStep(self):
        self.alpha[0] = np.multiply(self.pu_si[0], self.ptheta_xi[0])
        self.alpha[1] = np.multiply(self.pu_si[1], self.ptheta_xi[1])
        sum_alpha_si = np.sum(self.alpha, 0)
        self.alpha[0] = np.divide(self.alpha[0], sum_alpha_si)
        self.alpha[1] = np.divide(self.alpha[1], sum_alpha_si)

    # ===============================================
    def mStep(self):
        sum_alpha_s1_samples = np.sum(self.alpha[1], 0)
        den = len(self.samples) - sum_alpha_s1_samples
        self.ui = np.log(sum_alpha_s1_samples)
        self.ui -= np.log(den)
        exp_ui = np.exp(self.ui)
        self.pu_si[0] = np.clip(np.divide(1, 1 + exp_ui), 0.00001, 0.99999)
        self.pu_si[1] = np.clip(np.divide(exp_ui, 1 + exp_ui), 0.00001, 0.99999)
        [self.gaussians[0][i].estimate(self.samples[i], self.alpha[0][i]) for i in range(len(self.gaussians[0]))]
        [self.gaussians[1][i].estimate(self.samples[i], self.alpha[1][i]) for i in range(len(self.gaussians[1]))]
        self.ptheta_xi = [[self.gaussians[0][i].compute_probs(self.samples[i]) for i in range(len(self.samples))],
                          [self.gaussians[1][i].compute_probs(self.samples[i]) for i in range(len(self.samples))]]
        sum_ptheta_si = np.sum(self.ptheta_xi, 0)
        self.ptheta_xi[0] = np.divide(self.ptheta_xi[0], sum_ptheta_si)
        self.ptheta_xi[1] = np.divide(self.ptheta_xi[1], sum_ptheta_si)

    # ===============================================
    def diff_ui(self, old):
        return np.sum(np.square(self.ui - old)) / np.sum(np.square(self.ui))

    # ===============================================
    def diff_theta(self, old):
        diff = np.zeros(shape=(np.shape(old)))
        for i in range(len(self.gaussians)):
            for l in range(len(self.gaussians[i])):
                diff[i][l] = self.gaussians[i][l].compute_distance(old[i][l])

        return np.max(diff)

    def alpha_classif_for_image(self, l):
        image_alphas = [self.alpha[0][l], self.alpha[1][l]]
        classif = np.zeros(np.shape(image_alphas[0]))
        classif[image_alphas[0] < image_alphas[1]] = 1
        return classif