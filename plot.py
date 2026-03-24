import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out.csv")

for t in df["TYPE"].unique():
    sub = df[df["TYPE"] == t]
    plt.plot(sub["N"], sub["GFLOPS"], marker='o', label=t)

plt.xscale("log")
plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPS")
plt.legend()
plt.grid()

plt.savefig("gflops_plot.png", dpi=300, bbox_inches='tight')

plt.show()