import csv 

with open('metric_l_full.csv', mode='r') as file:
    reader = csv.reader(file)
    store = []
    for row in reader:
        store.append(row)

for i in range(1,7):
    print("&", "{:.4f}".format(float(store[i][1])), "&", "{:.4f}".format(float(store[i][2])), "&", "{:.4f}".format(float(store[i][3])))