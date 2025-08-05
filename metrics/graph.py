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
    
def add_new_plot(ax, x, y, prop,label):
    ax.plot(x,y,prop,label=label)
    return ax

def plot_all(name, ylabel, i, ylimit):
    fig, ax = plt.subplots(1,1,)
    ax = add_new_plot(ax, vitb[:,0], vitb[:,i], 'k--', 'vit-b')
    ax = add_new_plot(ax, vits[:,0], vits[:,i],'b--', 'vit-s')
    ax = add_new_plot(ax, vitl[:,0], vitl[:,i],'r--', 'vit-l')
    ax.set_xscale('log')
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray')
    ax.grid(True, which='minor', linestyle='--', linewidth=0.4, color='lightgray')
    ax.legend()
    ax.set_xlabel("Sampled Points Percentage")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylimit[0],ylimit[1])
    
    fig.savefig(name, bbox_inches="tight", pad_inches=0)

MAX_PT = 224*302

vitb = read_csv("/DepthPrompting/metrics/DepthAnything_b_our.csv", MAX_PT)
vits = read_csv("/DepthPrompting/metrics/DepthAnything_s_our.csv", MAX_PT)
vitl = read_csv("/DepthPrompting/metrics/DepthAnything_l_our.csv", MAX_PT)

plot_all("RMSE_depth_prompting.png", "RMSE/m",1,[0,0.35])
plot_all("MAE_depth_prompting.png", "MAE/m",2,[0,0.25])
plot_all("DELTA1_depth_prompting.png", "DELTA1",3,[0.92,1])
