import numpy as np

MORAL_LABELS = ["Care", "Harm", "Fairness", "Cheating", "Loyalty", "Betrayal", "Authority", "Subversion", "Purity", "Degradation"]

def compute_rank_aware_distribution(moral_vectors, use_mrr=True):
    """RADio 框架公式 (5)：计算排序感知下的离散概率分布"""
    moral_vectors = np.array(moral_vectors)
    if len(moral_vectors) == 0:
        return np.zeros(len(MORAL_LABELS))
        
    n_items = len(moral_vectors)
    weights = 1.0 / np.arange(1, n_items + 1) if use_mrr else np.ones(n_items)
    
    if moral_vectors.ndim == 1:
        oh_matrix = np.zeros((n_items, len(MORAL_LABELS)))
        for i, val in enumerate(moral_vectors):
            if int(val) < len(MORAL_LABELS):
                oh_matrix[i, int(val)] = 1.0
        moral_vectors = oh_matrix
        
    weighted_sum = np.dot(weights, moral_vectors)
    total_weight = np.sum(weights)
    
    distribution = weighted_sum / total_weight if total_weight > 0 else np.zeros(len(MORAL_LABELS))
    sum_dist = np.sum(distribution)
    return distribution / sum_dist if sum_dist > 0 else distribution

def compute_radio_js_divergence(P, Q, alpha=1e-4, eps=1e-9):
    """RADio 框架公式 (2, 4)：计算有界的平方根 JS 散度"""
    if np.sum(P) == 0 and np.sum(Q) == 0:
        return 0.0
        
    P_norm = P / np.sum(P) if np.sum(P) > 0 else np.ones(len(P)) / len(P)
    Q_norm = Q / np.sum(Q) if np.sum(Q) > 0 else np.ones(len(Q)) / len(Q)

    P_smooth = (1.0 - alpha) * P_norm + alpha * Q_norm
    Q_smooth = (1.0 - alpha) * Q_norm + alpha * P_norm
    
    P_smooth /= np.sum(P_smooth)
    Q_smooth /= np.sum(Q_smooth)
    
    M = 0.5 * (P_smooth + Q_smooth)
    
    def kl_divergence(p, q):
        return np.sum(p * np.log2((p + eps) / (q + eps)))
        
    js_divergence = 0.5 * kl_divergence(P_smooth, M) + 0.5 * kl_divergence(Q_smooth, M)
    return np.sqrt(max(0.0, js_divergence))