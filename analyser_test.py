from DataAnalyser import DataAnalyser

data_analyser = DataAnalyser(1000, 2)

data_analyser.plot_data()
print(data_analyser.calc_discrepancy())