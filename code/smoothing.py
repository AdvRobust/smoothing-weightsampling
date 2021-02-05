import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.special import softmax
import torch.nn.functional as F
import time


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self.sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self.sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self.sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        raise NotImplementedError

    @staticmethod
    def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    @staticmethod
    def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class BaseSmooth(Smooth):
    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        super(BaseSmooth, self).__init__(base_classifier, num_classes, sigma)

    def sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self.count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts


class SmoothedDetector(Smooth):
    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        super(SmoothedDetector, self).__init__(base_classifier, num_classes, sigma)

    def sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1))
                # print(batch.shape)
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self.count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts


class ImpSmooth(Smooth):
    # Smoothing for CIFAR10 and MNIST
    def __init__(self, base_classifier: torch.nn.Module, detector: SmoothedDetector, detector_nd: int, num_classes: int, sigma: float):
        super(ImpSmooth, self).__init__(base_classifier, num_classes, sigma)
        self.detector = detector
        self.detector_nd = detector_nd
        self.flag = False
        self.penalize_weights = []

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        self.base_classifier.eval()
        self.eval_base_classifier(x)
        counts_selection = self.sample_noise(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        counts_estimation = self.sample_noise(x, n, batch_size)
        nA = counts_estimation[cAHat].item()
        n_new = np.sum(counts_estimation)
        pABar = self._lower_confidence_bound(nA, n_new, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def eval_base_classifier(self, x):
        x = x.unsqueeze(0).cuda()
        output = self.base_classifier(x).detach().cpu().numpy()
        pred = output.argmax(axis=1)[0]
        softmax_output = softmax(output)
        softmax_diff = [(softmax_output[0][pred] - softmax_output[0][i]) for i in range(self.num_classes)]
        self.penalize_weights = softmax_diff

    def sample_noise(self, x, num, batch_size):
        with torch.no_grad():
            counts_with_penalize = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                outputs = self.base_classifier(batch + noise)
                detector_rs = [self.detector.predict(output, n=self.detector_nd, alpha=0.001, batch_size=400) for output in outputs]
                predictions = outputs.argmax(1)
                counts_with_penalize += self.count_arr_detector_penalize(predictions.cpu().numpy(), detector_rs, self.num_classes)
            return counts_with_penalize

    def count_arr_detector_penalize(self, arr, detector_rs, length):
        counts = np.zeros(length, dtype=float)
        for i in range(len(arr)):
            if detector_rs[i] == 0:
                counts[arr[i]] += 1
            else:
                counts[arr[i]] += -self.penalize_weights[arr[i]]
        result = np.zeros(length, dtype=int)
        for i in range(len(counts)):
            if counts[i] <= 0:
                result[i] = 0
            else:
                result[i] = round(counts[i])
        return result
