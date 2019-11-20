from DataAnalyser import DataAnalyser

da = DataAnalyser(1000, 2)

# da.plot_data()
# d, i = da.find_neighbors()
d1 = da.calc_discrepancy()

for i in range(100):
    if i % 10 == 0:
        d2 = da.calc_discrepancy()
        print("weighted difference:")
        print((d1 - d2) / d1)
        da.plot_data()
    da.shift_points(step_size=5e-4)
