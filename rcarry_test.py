import numpy as np
from RNGenerator import RCarry

reg = np.array([6, 28, 4, 93, 23, 31, 53, 43, 48, 62, 20, 56, 67,
                84, 44, 86, 61, 15, 38, 4, 24, 86, 2, 73])

mod = 2 ** 24
# reg = (reg * mod) % mod

r_carry = RCarry(mod, 10, reg)
len_mod = len(str(mod))
f = open("rcarry_generate.txt", "w")
for i in range(int(1e5)):
    f.writelines('{0:.5e}\n'.format(r_carry.generate()))

f.close()
