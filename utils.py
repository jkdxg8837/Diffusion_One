import torch
def _calculate_he(W: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    计算单个权重矩阵W的超球面能量 (Hyperspherical Energy)。

    根据公式: HE(W) = Σ_{i≠j} ||w_i - w_j||⁻¹
    其中 w_i 是 W 的第 i 个列向量。

    参数:
        W (torch.Tensor): 输入的权重矩阵，形状为 (d, n)，
                          其中 d 是特征维度，n 是神经元数量（即列向量的数量）。
        epsilon (float): 一个很小的数值，用于防止除以零的错误。

    返回:
        torch.Tensor: 一个标量张量，表示矩阵W的超球面能量。
    """
    # 检查输入是否为2D张量
    if W.dim() != 2:
        raise ValueError(f"输入矩阵必须是2D的, 但得到的形状是 {W.shape}")

    # torch.pdist 用于计算行向量之间的成对距离。
    # 我们的 w_i 是列向量，所以需要先将矩阵 W 转置。
    # W.T 的形状将变为 (n, d)。
    W_T = W.T

    # 计算所有成对的欧几里得距离 (p=2)。
    # 这会返回一个包含 n * (n - 1) / 2 个距离值的一维张量。
    pairwise_distances = torch.pdist(W_T.float(), p=2)

    # 计算距离的倒数。加上 epsilon 以确保数值稳定性。
    reciprocal_distances = 1.0 / (pairwise_distances + epsilon)

    # 公式中的求和 Σ_{i≠j} 包含了 (i, j) 和 (j, i) 两种情况。
    # 由于 ||w_i - w_j|| = ||w_j - w_i||，这个总和是 pdist 结果总和的两倍。
    he_value = torch.sum(reciprocal_distances)

    return he_value.item()