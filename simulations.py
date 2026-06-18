import numpy as np
from metrics import compute_rank_aware_distribution, compute_radio_js_divergence, MORAL_LABELS
from data import get_track_moral_vector

TOTAL_LOOPS = 100
TOP_K_INJECT = 5
CANDIDATE_SIZE = 100

def run_acoustic_similarity_simulation(P_distribution, initial_vectors, initial_seen, scaled_embeddings, id_to_name, cursor, conn):
    """【算法 1】实验组 - 纯声学特征相似度演化模型"""
    current_playlist_vectors = list(initial_vectors)
    already_seen_ids = set(initial_seen)
    radio_evolution = []

    for loop in range(1, TOTAL_LOOPS + 1):
        user_music_profile = np.mean(current_playlist_vectors, axis=0)
        music_scores = np.dot(scaled_embeddings, user_music_profile)
        top_music_candidates = np.argsort(music_scores)[::-1]
        
        rerank_list = []
        for cid in top_music_candidates:
            if cid in already_seen_ids: continue
            c_meta = id_to_name[cid]
            c_moral_vec = get_track_moral_vector(c_meta['track_name'], c_meta['artist_name'], cursor, conn, allow_network=False)
            rerank_list.append((cid, music_scores[cid], c_moral_vec))
            if len(rerank_list) >= CANDIDATE_SIZE: break
        
        if len(rerank_list) == 0: break
            
        actual_recommendation_moral_vectors = []
        for idx in range(min(TOP_K_INJECT, len(rerank_list))):
            best_cid, _, best_moral_vec = rerank_list[idx]
            actual_recommendation_moral_vectors.append(best_moral_vec)
            current_playlist_vectors.append(scaled_embeddings[best_cid])
            already_seen_ids.add(best_cid)

        Q_distribution = compute_rank_aware_distribution(actual_recommendation_moral_vectors, use_mrr=True)
        loop_divergence = compute_radio_js_divergence(P_distribution, Q_distribution)
        radio_evolution.append(loop_divergence)
        
        # 📊 实时打印：第 1 轮和第 100 轮输出全量 10 维道德概率流，中间轮次仅打印主导道德
        if loop == 1 or loop == TOTAL_LOOPS:
            max_idx = np.argmax(Q_distribution)
            print(f"  |-- [Acoustic] Loop {loop:03d} | RADio Divergence: {loop_divergence:.4f} | Dominant Moral: [{MORAL_LABELS[max_idx]}] ({Q_distribution[max_idx]*100:.1f}%)")
            # 格式化输出 10 维全景向量
            distribution_str = " ".join([f"{L[:4]}:{V*100:4.1f}%" for L, V in zip(MORAL_LABELS, Q_distribution)])
            print(f"      👉 [Full Moral Vector Spectrum]: {distribution_str}")
        elif loop <= 3 or loop % 50 == 0:
            max_idx = np.argmax(Q_distribution)
            print(f"  |-- [Acoustic] Loop {loop:03d} | RADio Divergence: {loop_divergence:.4f} | Dominant Moral: [{MORAL_LABELS[max_idx]}]")
            
    return radio_evolution

def run_random_simulation(P_distribution, initial_seen, id_to_name, cursor, conn):
    """【算法 2】对照组 - 纯随机投喂大盘模型"""
    already_seen_ids = set(initial_seen)
    radio_evolution = []
    all_candidate_ids = list(id_to_name.keys())

    for loop in range(1, TOTAL_LOOPS + 1):
        available_candidates = [cid for cid in all_candidate_ids if cid not in already_seen_ids]
        if len(available_candidates) < TOP_K_INJECT: break
            
        selected_cids = np.random.choice(available_candidates, size=TOP_K_INJECT, replace=False)
        actual_recommendation_moral_vectors = []
        
        for best_cid in selected_cids:
            c_meta = id_to_name[best_cid]
            c_moral_vec = get_track_moral_vector(c_meta['track_name'], c_meta['artist_name'], cursor, conn, allow_network=False)
            actual_recommendation_moral_vectors.append(c_moral_vec)
            already_seen_ids.add(best_cid)

        Q_distribution = compute_rank_aware_distribution(actual_recommendation_moral_vectors, use_mrr=True)
        loop_divergence = compute_radio_js_divergence(P_distribution, Q_distribution)
        radio_evolution.append(loop_divergence)
        
        # 📊 实时打印：随机组的第一轮和最后一轮全量输出
        if loop == 1 or loop == TOTAL_LOOPS:
            max_idx = np.argmax(Q_distribution)
            print(f"  |-- [Random]   Loop {loop:03d} | RADio Divergence: {loop_divergence:.4f} | Dominant Moral: [{MORAL_LABELS[max_idx]}] ({Q_distribution[max_idx]*100:.1f}%)")
            distribution_str = " ".join([f"{L[:4]}:{V*100:4.1f}%" for L, V in zip(MORAL_LABELS, Q_distribution)])
            print(f"      👉 [Full Moral Vector Spectrum]: {distribution_str}")
        
    return radio_evolution

def run_user_cf_simulation(P_distribution, initial_vectors, initial_seen, embeddings, id_to_name, cursor, conn):
    """【算法 3】协同组 - 经典品味协同过滤 User-CF 模型"""
    current_playlist_vectors = list(initial_vectors)
    already_seen_ids = set(initial_seen)
    radio_evolution = []

    for loop in range(1, TOTAL_LOOPS + 1):
        virtual_user_profile = np.mean(current_playlist_vectors, axis=0)
        cf_scores = np.dot(embeddings, virtual_user_profile)
        top_cf_candidates = np.argsort(cf_scores)[::-1]
        
        rerank_list = []
        for cid in top_cf_candidates:
            if cid in already_seen_ids: continue
            c_meta = id_to_name[cid]
            c_moral_vec = get_track_moral_vector(c_meta['track_name'], c_meta['artist_name'], cursor, conn, allow_network=False)
            rerank_list.append((cid, cf_scores[cid], c_moral_vec))
            if len(rerank_list) >= CANDIDATE_SIZE: break
                
        if len(rerank_list) == 0: break
            
        actual_recommendation_moral_vectors = []
        for idx in range(min(TOP_K_INJECT, len(rerank_list))):
            best_cid, _, best_moral_vec = rerank_list[idx]
            actual_recommendation_moral_vectors.append(best_moral_vec)
            current_playlist_vectors.append(embeddings[best_cid])
            already_seen_ids.add(best_cid)

        Q_distribution = compute_rank_aware_distribution(actual_recommendation_moral_vectors, use_mrr=True)
        loop_divergence = compute_radio_js_divergence(P_distribution, Q_distribution)
        radio_evolution.append(loop_divergence)
        
        # 📊 实时打印：User-CF 组的第一轮和最后一轮全量输出
        if loop == 1 or loop == TOTAL_LOOPS:
            max_idx = np.argmax(Q_distribution)
            print(f"  |-- [User-CF]  Loop {loop:03d} | RADio Divergence: {loop_divergence:.4f} | Dominant Moral: [{MORAL_LABELS[max_idx]}] ({Q_distribution[max_idx]*100:.1f}%)")
            distribution_str = " ".join([f"{L[:4]}:{V*100:4.1f}%" for L, V in zip(MORAL_LABELS, Q_distribution)])
            print(f"      👉 [Full Moral Vector Spectrum]: {distribution_str}")
        
    return radio_evolution