import csv 

with open('DepthAnything_s_full_our.csv', mode='r') as file:
    reader = csv.reader(file)
    store = []
    for row in reader:
        store.append(row)

for i in range(1,7):
    print("&", "{:.3f}".format(float(store[i][1])), "&", "{:.3f}".format(float(store[i][2])), "&", "{:.3f}".format(float(store[i][3])))