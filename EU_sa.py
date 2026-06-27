import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, probplot

index = { "^FCHI"       : "CAC40",
          "^GDAXI"      : "DAX40",
          "^FTSE"       : "FTSE100", 
          "^STOXX50E"   : "EUROSTOXX50"}

index_p = yf.download(list(index.keys()), start="1990-03-01")["Close"]
index_p = index_p.rename(columns=index)

index_retD = np.log(index_p).diff()
index_retD = index_retD.dropna()

fig, axes = plt.subplots(4,2,figsize=(18,24))

y_data = [(index_retD[i], i) for i in list(index.values())]

y_ticks = np.arange(25,250,25)
x_ticks = np.linspace(-.1, .1, 11)

norm_fit = np.linspace(x_ticks.min(),x_ticks.max(), 500)
bin_l = 250

qq_x = np.arange(-4,5,1)
qq_y = np.arange(-.15, .15, .05)

global_min = min(index_retD[i].min() for i in index.values()) * 1.1
global_max = max(index_retD[i].max() for i in index.values()) * 1.1

for idx, (y_val, title) in enumerate(y_data):
    ax = axes[idx,0]
    ax_qq = axes[idx,1]

    ax.hist(y_val, bins=bin_l, color="#B7CCFF", edgecolor="#A5B9E4")

    ax.set_xticks(x_ticks, labels=[f"{i:.1%}" for i in x_ticks])
    ax.set_xlim(xmin=-0.095, xmax=0.095)

    ax.set_yticks(y_ticks)
    
    bin_w = (y_val.max() - y_val.min())/bin_l
    ax.plot(norm_fit, norm.pdf(norm_fit, y_val.mean(), y_val.std())*len(y_val)*bin_w, label="normal fit")

    stats = (f"Skew: {skew(y_val):>6.2f} \n"
             f"Kurt (excess): {kurtosis(y_val):>6.2f} \n"
             f"SD: {y_val.std():>6.2%}\n")
    ax.text(.97,.97,stats,transform=ax.transAxes,family="monospace", ha="right", va="top")

    ax.set_ylabel("Frequency")
    ax.set_xlabel("1D Return (%)")
    ax.legend()
    ax.set_title(f"{title} Return Distribution")

    (t_q, s_q), (slope, intercept, _) = probplot(y_val, dist="norm")
    ax_qq.scatter(t_q,s_q, s=1, color="#0A6CCD")
    ax_qq.plot(t_q, t_q*slope+intercept, color="#111D61", lw=0.7)

    ax_qq.set_xticks(qq_x, labels=[f"{i:.0f}" for i in qq_x])
    ax_qq.set_xlim(xmin=-3.8, xmax=3.8)

    ax_qq.set_yticks(qq_y, labels=[f"{i:.2f}" for i in qq_y])
    ax_qq.set_ylim(ymin=global_min, ymax=global_max)

    ax_qq.set_ylabel("Sample Quantile")
    ax_qq.set_xlabel("Theoretical Quantile")
    ax_qq.set_title(f"{title} QQ Plot")

plt.tight_layout()
plt.show()