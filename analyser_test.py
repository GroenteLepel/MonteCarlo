from DataAnalyser import DataAnalyser

da = DataAnalyser(1000, 2)

da.plot_data()
d1 = da.calc_discrepancy()
for i in range(10):
    da.replace_points()
da.plot_data()
d2 = da.calc_discrepancy()

print(d1 - d2)
# d, i = da.find_neighbors()