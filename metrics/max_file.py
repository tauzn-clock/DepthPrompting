import csv

csv_list = {
    "depth_prompting": "/DepthPrompting/metrics/DepthPromptingNYU.csv",
    "depth_prompting_prop": "/DepthPrompting/metrics/PropNYU.csv",
    "depth_prompting_cs": "/DepthPrompting/metrics/OursNYU.csv",
    "da_s": "/DepthPrompting/metrics/da_s.csv",
    "da_b": "/DepthPrompting/metrics/da_b.csv",
    "da_l": "/DepthPrompting/metrics/da_l.csv",
    "da_s_full": "/DepthPrompting/metrics/da_s_full.csv",
    "da_b_full": "/DepthPrompting/metrics/da_b_full.csv",
    "da_l_full": "/DepthPrompting/metrics/da_l_full.csv",
    "metric_s": "/DepthPrompting/metrics/metric_s.csv",
    "metric_l": "/DepthPrompting/metrics/metric_l.csv",
    "metric_s_full": "/DepthPrompting/metrics/metric_s_full.csv",
    "metric_l_full": "/DepthPrompting/metrics/metric_l_full.csv"
}

csv_array = {}

for k in csv_list:
    with open(csv_list[k], "r") as f:
        reader = csv.reader(f)
        store = []
        for row in reader:
            try: 
                store.append([float(v) for v in row])
            except:
                pass
    csv_array[k] = store

i = 5
j = 1
csv_array_keys = list(csv_array.keys())
best = csv_array_keys[0]
for k in csv_array:
    print(k,csv_array[k][i][j])
    if csv_array[k][i][j] > csv_array[best][i][j] and j == 3:
        best = k
    if csv_array[k][i][j] < csv_array[best][i][j] and j != 3:
        best = k
print(f"Best for {i}, {j}: {best}")

