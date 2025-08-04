import matplotlib.pyplot as plt
import csv
import numpy as np

def read_csv(csv_path, MAX_PT):
    arr = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try: 
                arr.append([float(v) for v in row])
            except:
                pass
    arr = np.array(arr)
    arr[:,0] /= MAX_PT  
    
    return arr
    
def add_new_plot(ax, x, y, prop):
    ax.plot(x,y,prop)
    return ax

MAX_PT = 224*302

depthprompting = read_csv("/DepthPrompting/metrics/DepthPromptingNYU.csv", MAX_PT)
prop = read_csv("/DepthPrompting/metrics/PropNYU.csv", MAX_PT)

fig, ax = plt.subplots(1,1,)
ax = add_new_plot(ax, prop[:-1,0], prop[:-1,1], 'r--')
ax = add_new_plot(ax, depthprompting[:-1,0], depthprompting[:-1,1],'b--')

fig.savefig("test.png")