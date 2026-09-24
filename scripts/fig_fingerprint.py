import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.linewidth":0.7,"savefig.dpi":300,
                     "figure.facecolor":"white"})
D=[("Hassler et al.",1,0.918),("MAVLink (GUIDE)",2,0.712),
   ("SHADOW-GCS",14,0.588),("UAVCAN 2022",10,0.134),("UAVCAN 2026",10,0.093)]
BLUE,RED,GREY="#2166AC","#B4453C","#6E6E6E"
fig,ax=plt.subplots(figsize=(5.6,3.5))
ax.axvspan(0.8,1.6,color="#E8D5D2",alpha=.7,lw=0,zorder=0)
ax.text(1.02,0.055,"class and capture\nare the same variable",fontsize=7.4,
        color="#8C2F28",va="bottom",zorder=4)
ax.axhline(0.5,color=GREY,lw=.8,ls=":",zorder=1)
ax.text(30,0.515,"flagged",fontsize=7.4,color=GREY,ha="right")
for nm,c,i in D:
    bad = c<2
    ax.scatter([c],[i],s=62,color=RED if bad else BLUE,zorder=3,
               edgecolor="white",linewidth=.9)
    dx,ha=(1.13,"left") if nm!="UAVCAN 2026" else (0.88,"right")
    dy = -0.045 if nm=="UAVCAN 2026" else 0.0
    ax.annotate(nm,(c,i),xytext=(c*dx,i+0.028+dy),fontsize=8,ha=ha,
                color=RED if bad else "#1A1A1A")
ax.set_xscale("log"); ax.set_xlim(0.8,34); ax.set_ylim(0,1.0)
ax.set_xticks([1,2,5,10,20]); ax.set_xticklabels(["1","2","5","10","20"],fontsize=8)
ax.minorticks_off()
ax.set_xlabel("smallest number of captures in any class")
ax.set_ylabel("capture-fingerprint index  $\\iota$")
ax.grid(axis="y",color="#EAEAEA",lw=.6,zorder=0); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
os.makedirs("runs/audit", exist_ok=True)
fig.tight_layout(); fig.savefig("runs/audit/fig_fingerprint.png",
                                bbox_inches="tight",pad_inches=.05)
print("ok")
