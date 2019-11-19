import numpy as np
from RNGenerator import RCarry
import matplotlib.pyplot as plt
from functools import reduce
import operator
from sklearn.neighbors import NearestNeighbors


class DataAnalyser:

    def __init__(self, n_points: int, n_dims: int):
        self.register = np.array(
            [15447500, 3865874, 9333626, 1165797, 11995182, 323598, 11737196,
             16508674, 8815592, 5831275, 15008810, 6485217, 9434883, 10508632,
             2884549, 14453457, 1563784, 9025080, 10942422, 1640972, 227199,
             5107938, 1082897, 12327101])
        self.modulus = 2 ** 24
        generator = RCarry(self.modulus, 10, self.register)

        self.n_points = n_points
        self.n_dims = n_dims
        self.data = generator.generate_set((n_points, n_dims))
        # self.data = np.random.rand(n_points, n_dims)

    def hist_data(self, n_bins: int):
        """
        Calculate the histogram data of self.data divided over n_bins. If
        self.data is of anything other than 1D array, the flattened array is
        used.
        :param n_bins:
        :return:
        """
        return np.histogram(self.data, bins=n_bins)

    def calc_chisquared(self, n_bins: int):
        hist_data = self.hist_data(n_bins)

        expected_n = len(self.data) / n_bins
        chi_squared = 0
        for d in hist_data[0]:
            chi_squared += (d - expected_n) ** 2 / expected_n
        return chi_squared

    def calc_discrepancy(self):
        """
        Calculates the quadratic discrepancy of the dataset self.data.

        Tells you something about the uniformity of the point set. A low
        discrepancy means that the points are neatly distributed. This is quite
        an equation to calculate, so is not that optimal to use for optimising
        the point set.
        :return:
        """
        expectation = (2 ** (-self.n_dims) - 3 ** (-self.n_dims)) / \
                      self.n_points
        print("Expected discrepancy:", expectation)

        # The problem can be viewed as a matrix containing all the possible
        #  combinations between self.data with itself. This means that the
        #  matrix is symmetric, so we can save time by calculating one half
        #  of the matrix and take these components twice.
        first_term, second_term = 0, 0
        for i in range(self.n_points):
            for j in range(i, self.n_points):
                val = np.zeros(2)
                for d in range(self.n_dims):
                    val[d] = 1 - np.maximum(self.data[i][d], self.data[j][d])

                if j == i:
                    # All the diagonal terms must only be taken once
                    first_term += reduce(operator.mul, val)
                else:
                    first_term += 2 * reduce(operator.mul, val)

            second_term += reduce(operator.mul, 1 - self.data[i] ** 2)

        # This is the quadratic discrepancy, L_2^*.
        discrepancy = first_term / (self.n_points ** 2) - \
                      2 * second_term / (2 ** self.n_dims * self.n_points) + \
                      (1. / 3) ** self.n_dims
        print("Calculated discrepancy:", discrepancy)

        return discrepancy

    def find_neighbors(self):
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        nbrs.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)

        return distances, indices

    def replace_points(self):
        dists, ind = self.find_neighbors()
        # Take one point for starters, first point of the array.
        to_replace = self.data[ind[0, 0]]
        furthest_point = self.data[ind[0, -1]]
        distance = dists[0, -1]

        # angle =

        xy_distance = furthest_point - to_replace
        adjustment = -0.5 * xy_distance
        new_x = to_replace + adjustment

        self.data[ind[:, 0]] = new_x

    def generate_next_set(self):
        self.register = self.modulus * self.data[-24:]
        print("New register: ")
        print(self.register)
        r_carry = RCarry(self.modulus, 10, self.register.astype(int))

        r_carry.fill_array(self.data)

    def plot_data(self):
        if self.n_dims > 2:
            print("Only 1 or 2 dimensional plots possible.")
            return 0

        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1])

        d, i = self.find_neighbors()
        nn = self.data[i[0]]
        ax.scatter(nn[:, 0], nn[:, 1], marker='+')
        ax.scatter(nn[:, 0].mean(), nn[:, 1].mean(), marker='x')
        fig.show()
