import numpy as np
from RNGenerator import RCarry
import matplotlib.pyplot as plt


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

    # def calc_discrepancy(self):


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
        fig.show()
