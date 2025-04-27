from numpy import load

data = load('brier_data.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])