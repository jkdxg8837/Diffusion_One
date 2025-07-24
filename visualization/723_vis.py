import matplotlib.pyplot as plt

# 步数作为横坐标
baseline_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # 最后一步为 output，没有checkpoint号，按500算

# 对应的 (CLIPT, CLIPI)
baseline_clip_scores = [
    (19.60993, 93.6359),
    (19.50042, 93.1519),
    (18.99648, 93.9202),
    (18.91443, 94.457),
    (18.70858, 94.4168),
    (18.60702, 95.2059),
    (18.9401, 93.5527),
    (18.65392, 94.4865),
    (19.07147, 95.3927),
    (18.73343, 95.2628),
]


# 数据
stable_gammas = [25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625, 729, 841, 961]

step_0 = [
    (21.78227, 67.6509),
    (21.35617, 71.6198),
    (21.9106, 70.6977),
    (21.01493, 73.3663),
    (20.67795, 72.013),
    (20.86592, 72.0433),
    (20.81585, 70.5635),
    (23.15618, 70.2627),
    (20.8451, 66.9023),
    (21.04422, 72.8768),
    (21.21458, 73.5467),
    (21.26018, 70.4063),
    (20.96385, 72.0515),
    (20.77933, 71.6746),
]

step_50 = [
    (20.87255, 66.1726),
    (19.6914, 92.2586),
    (19.48997, 91.6396),
    (19.63063, 91.4373),
    (19.04873, 91.27),
    (19.1379, 92.0978),
    (19.82278, 87.4386),
    (19.81753, 90.8324),
    (20.02095, 92.0346),
    (20.34752, 85.8842),
    (19.89285, 91.0223),
    (19.70812, 93.1378),
    (19.55768, 91.9177),
    (19.72787, 92.6661),
]

step_100 = [
    (21.64255, 70.1817),
    (20.0286, 95.0874),
    (19.99643, 93.2909),
    (19.32455, 94.9164),
    (19.24298, 93.6725),
    (19.18887, 94.116),
    (20.13627, 92.0697),
    (20.01403, 92.968),
    (19.9276, 92.7564),
    (19.9089, 91.679),
    (19.35255, 93.6501),
    (19.33317, 94.1162),
    (20.09528, 94.0023),
    (19.69545, 93.3481),
]

# 拆分成两个指标的列表
clipT_0, clipI_0 = zip(*step_0)
clipT_50, clipI_50 = zip(*step_50)
clipT_100, clipI_100 = zip(*step_100)

# 绘图
plt.figure(figsize=(14, 6))

# --- 子图1: CLIPT
plt.subplot(1, 2, 1)
plt.plot(stable_gammas, clipT_0, label="Step 0", marker='o')
plt.plot(stable_gammas, clipT_50, label="Step 50", marker='o')
plt.plot(stable_gammas, clipT_100, label="Step 100", marker='o')
plt.title("CLIPT vs Stable Gamma")
plt.xlabel("Stable Gamma")
plt.ylabel("CLIPT")
plt.grid(True)
plt.legend()

# --- 子图2: CLIPI
plt.subplot(1, 2, 2)
plt.plot(stable_gammas, clipI_0, label="Step 0", marker='o')
plt.plot(stable_gammas, clipI_50, label="Step 50", marker='o')
plt.plot(stable_gammas, clipI_100, label="Step 100", marker='o')
plt.title("CLIPI vs Stable Gamma")
plt.xlabel("Stable Gamma")
plt.ylabel("CLIPI")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("clip_scores_vs_stable_gamma.png")