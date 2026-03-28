import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out.csv")

# for t in df["TYPE"].unique():
#     sub = df[df["TYPE"] == t]
#     plt.plot(sub["N"], sub["GFLOPS"], marker='o', label=t)

# plt.xscale("log")
# plt.xlabel("Matrix Size (N)")
# plt.ylabel("GFLOPS")
# plt.legend()
# plt.grid()

# plt.savefig("gflops_plot.png", dpi=300, bbox_inches='tight')

# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("out.csv")

# Ensure sorting (important for clean lines)
df = df.sort_values(by="N")

# Convert N → log2(N)
df["log2N"] = np.log2(df["N"])

plt.figure(figsize=(10, 6))

for t in df["TYPE"].unique():
    sub = df[df["TYPE"] == t]
    plt.plot(sub["log2N"], sub["GFLOPS"], marker='o', label=t)

# ---- Custom ticks (powers of 2) ----
unique_N = sorted(df["N"].unique())
log_ticks = np.log2(unique_N)

plt.xticks(log_ticks, labels=[str(n) for n in unique_N], rotation=45)

plt.xlabel("Matrix Size (N) [powers of 2]")
plt.ylabel("GFLOPS")
plt.title("GEMM Performance Scaling (FP32 vs FP16 vs INT8)")

plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()

plt.savefig("gflops_plot.png", dpi=300)
plt.show()