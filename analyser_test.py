from DataAnalyser import DataAnalyser

data_analyser = DataAnalyser("rcarry_generate.txt", 10)
a = data_analyser.data
print(data_analyser.calc_chisquared())

data_analyser.generate_next_set()
b = data_analyser.data
print(data_analyser.calc_chisquared())
c = a - b
print(c)