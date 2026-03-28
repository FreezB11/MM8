import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv", comment="#")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(df["N"], df["Time_ms"], marker="o", markersize=3)
ax1.set_xlabel("N")
ax1.set_ylabel("Time (ms)")
ax1.set_title("mm83 — Time vs Matrix Size")
ax1.grid(True)

ax2.plot(df["N"], df["GFLOPS"], marker="o", markersize=3, color="orange")
ax2.axhline(y=5200, color="red", linestyle="--", label="RTX 3050 FP32 peak (5200 GFLOPS)")
ax2.axhline(y=df["GFLOPS"].max(), color="green", 
            linestyle="--", label=f'your peak ({df["GFLOPS"].max():.0f} GFLOPS)')
ax2.set_xlabel("N")
ax2.set_ylabel("GFLOPS")
ax2.set_title("mm83 — GFLOPS vs Matrix Size")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("bench.png", dpi=150)
plt.show()

print(f"peak GFLOPS: {df['GFLOPS'].max():.2f} at N={df.loc[df['GFLOPS'].idxmax(), 'N']}")
print(f"max N:       {df['N'].max()}")
print(f"max time:    {df['Time_ms'].max():.2f} ms at N={df.loc[df['Time_ms'].idxmax(), 'N']}")