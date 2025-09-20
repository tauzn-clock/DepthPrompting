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

MAX_PT = 228*304
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.linewidth'] = 1          # Axis border (spine) width
#plt.rcParams['xtick.major.width'] = 1       # Major tick width (x-axis)
#plt.rcParams['ytick.major.width'] = 1       # Major tick width (y-axis)
#plt.rcParams['xtick.minor.width'] = 1       # Optional: minor ticks
#plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['font.size'] = 18                # Base font size

depth_prompting = depthprompting = read_csv("/DepthPrompting/metrics/DepthPromptingNYU.csv", MAX_PT)
depth_prompting_prop = read_csv("/DepthPrompting/metrics/PropNYU.csv", MAX_PT)
depth_prompting_cs = read_csv("/DepthPrompting/metrics/OursNYU.csv", MAX_PT)

da_s = read_csv("/DepthPrompting/metrics/da_s.csv", MAX_PT)
da_b = read_csv("/DepthPrompting/metrics/da_b.csv", MAX_PT)
da_l = read_csv("/DepthPrompting/metrics/da_l.csv", MAX_PT)

da_s_full = read_csv("/DepthPrompting/metrics/da_s_full.csv", MAX_PT*4)
da_b_full = read_csv("/DepthPrompting/metrics/da_b_full.csv", MAX_PT*4)
da_l_full = read_csv("/DepthPrompting/metrics/da_l_full.csv", MAX_PT*4)

metric_s = read_csv("/DepthPrompting/metrics/metric_s.csv", MAX_PT)
metric_l = read_csv("/DepthPrompting/metrics/metric_l.csv", MAX_PT)

metric_s_full = read_csv("/DepthPrompting/metrics/metric_s_full.csv", MAX_PT*4)
metric_l_full = read_csv("/DepthPrompting/metrics/metric_l_full.csv", MAX_PT*4)

fig, ax = plt.subplots(1,3,figsize=(25, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

ax[0].set_xlabel("Sampled Ratio")
ax[1].set_xlabel("Sampled Ratio")
ax[2].set_xlabel("Sampled Ratio")

ax[0].set_ylabel("RMSE/m")
ax[1].set_ylabel("MAE/m")
ax[2].set_ylabel("DELTA1")

for i in range(1,4):
    ax[i-1].plot(depth_prompting[:,0], depth_prompting[:,i], 'k--', marker="^", label='DepthPrompting')
    #ax[i-1].plot(depth_prompting_prop[:,0], depth_prompting_prop[:,i], 'k-', label='DepthFormer + Proportional')
    ax[i-1].plot(depth_prompting_cs[:,0], depth_prompting_cs[:,i], 'r--', marker="^", label='DepthFormer + CS')

    ax[i-1].plot(da_s[:,0], da_s[:,i], 'g--', marker="^", label='DepthAnythingV2-Small + CS')
    #ax[i-1].plot(da_b[:,0], da_b[:,i], 'g-', marker="s", label='DepthAnythingV2-Base + CS')
    #ax[i-1].plot(da_l[:,0], da_l[:,i], 'g-', marker="o", label='DepthAnythingV2-Large + CS')

    ax[i-1].plot(metric_s[:,0], metric_s[:,i], 'b--', marker="^", label='Metric3DV2-Small + CS')
    #ax[i-1].plot(metric_l[:,0], metric_l[:,i], 'b-', marker="o", label='Metric3D-Large + CS')

    ax[i-1].plot(da_s_full[:,0], da_s_full[:,i], 'g-', marker="^", label='DepthAnythingV2-Small + CS (Full Res)')
    
    ax[i-1].plot(metric_s_full[:,0], metric_s_full[:,i], 'b-', marker="^", label='Metric3DV2-Small + CS (Full Res)')

    #ax[i-1].set_xscale('log')
    ax[i-1].grid(True)
    ax[i-1].minorticks_on()
    ax[i-1].grid(True, which='major', linestyle='-', linewidth=0.8, color='gray')
    ax[i-1].grid(True, which='minor', linestyle='--', linewidth=0.4, color='lightgray')

    ax[i-1].set_xlim(0.01, 1)

ax[0].set_ylim(0,0.1)
ax[1].set_ylim(0,0.025)
ax[2].set_ylim(0.995,1)

ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, edgecolor='black')

plt.savefig("graph.png", bbox_inches="tight", pad_inches=0)