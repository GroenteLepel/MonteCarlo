import numpy as np
from RNGenerator import RCarry


class DataAnalyser:

    def __init__(self, filename, n_bins):
        """
        Initialize the analyser with a filename containing N numbers on the
        first column
        :param filename:
        """
        self.data = np.loadtxt(filename, unpack=True)
        self.n_bins = n_bins
        self.bin_data = np.histogram(self.data, bins=self.n_bins)

    def calc_chisquared(self):
        expected_n = len(self.data) / self.n_bins
        chi_squared = 0
        for d in self.bin_data[0]:
            chi_squared += (d - expected_n) ** 2 / expected_n
        return chi_squared

    def generate_next_set(self):
        mod = 2 ** 24
        register = mod * self.data[-24:]
        r_carry = RCarry(mod, 10, register.astype(int))

        for i in range(len(self.data)):
            b = r_carry.generate()
            self.data[i] = b
            print(str(i) + " " + str(self.data[i]))
