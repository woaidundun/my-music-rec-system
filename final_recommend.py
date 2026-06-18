import os
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔗 核心模块跨文件导入
from metrics import compute_rank_aware_distribution, MORAL_LABELS
from data import load_base_data, extract_initial_playlist_state
from simulations import (
    run_acoustic_similarity_simulation, 
    run_random_simulation, 
    run_user_cf_simulation, 
    TOTAL_LOOPS
)

# ==================== 路径全局控制中心 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_FILE_PATH = os.path.join(BASE_DIR, 'resources', 'data', 'tracks_info.json') 
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'resources', 'data', 'embeddings', 'song_embeddings_128.npy')
MPD_SLICE_PATH = os.path.join(BASE_DIR, 'mpd.slice.0-999.json')
DB_PATH = os.path.join(BASE_DIR, 'music_moral_analysis.db')
# =========================================================

# 🌟 路径安全性校验
if not all(os.path.exists(p) for p in [INFO_FILE_PATH, EMBEDDINGS_PATH, MPD_SLICE_PATH]):
    raise FileNotFoundError("❌ CRITICAL ERROR: Asset files (npy/json) are incomplete. Please verify the resources directory.")

if __name__ == "__main__":
    # 关闭多余的显存占用，强制走高性能并行 CPU 逻辑
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # 1. 跨文件加载全局大盘基础特征
    embs, n2id, id2n = load_base_data(INFO_FILE_PATH, EMBEDDINGS_PATH)
    
    if embs is not None:
        print(f"▶️ LOADING MPD PLAYLIST SLICE FROM SPOTIFY: {MPD_SLICE_PATH}")
        with open(MPD_SLICE_PATH, 'r', encoding='utf-8') as f:
            mpd_data = json.load(f)
            
        playlists = mpd_data.get('playlists', [])
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()
        
        # 为实验组 1 预计算长尾归一化标准矩阵
        alpha_norm = 0.5
        track_norms = np.linalg.norm(embs, axis=1, keepdims=True)
        track_norms = np.where(track_norms == 0, 1, track_norms)
        scaled_embs = embs / (track_norms ** alpha_norm)
        
        simulated_count = 0
        all_acoustic_trajectories = []
        
        # 2. 调度循环，完全切断任何志愿者外部 csv 污染
        for plist in playlists:
            tracks = plist.get('tracks', [])
            pid = plist.get('pid', 'Unknown')
            
            if len(tracks) >= 15:
                print(f"\n⚡ [Simulation Environment {simulated_count+1}/5] Initializing Playlist ID: [{pid}], extracting baseline features...")
                
                # 数据解耦抽取
                init_vecs, init_morals, init_seen = extract_initial_playlist_state(tracks, embs, n2id, cursor, conn)
                if len(init_vecs) == 0 or len(init_morals) == 0: continue
                    
                # 确立该播放列表的无偏真实道德初心基准 P
                P_dist = compute_rank_aware_distribution(init_morals, use_mrr=False)
                
                # 3. 跨文件运行 3 大算法引擎
                print(f"  👉 Running: Acoustic-Similarity Model simulation...")
                acoustic_traj = run_acoustic_similarity_simulation(P_dist, init_vecs, init_seen, scaled_embs, id2n, cursor, conn)
                
                print(f"  👉 Running: Pure Random Baseline simulation...")
                random_traj = run_random_simulation(P_dist, init_seen, id2n, cursor, conn)
                
                print(f"  👉 Running: Classic User-CF Model simulation...")
                ucf_traj = run_user_cf_simulation(P_dist, init_vecs, init_seen, embs, id2n, cursor, conn)
                
                # 4. 独立渲染画布：每个歌单独立出对比图
                if len(acoustic_traj) == TOTAL_LOOPS and len(random_traj) == TOTAL_LOOPS and len(ucf_traj) == TOTAL_LOOPS:
                    all_acoustic_trajectories.append(acoustic_traj)
                    
                    sns.set_theme(style="whitegrid")
                    plt.figure(figsize=(9, 4.8))
                    loops = np.arange(1, TOTAL_LOOPS + 1)
                    
                    # 🟥 声学内容特征线
                    plt.plot(loops, acoustic_traj, color='#d32f2f', linewidth=2.5, label='Acoustic-Similarity Model')
                    plt.fill_between(loops, acoustic_traj, color='#ffcdd2', alpha=0.12)
                    
                    # 🟨 协同过滤 User-CF 线
                    plt.plot(loops, ucf_traj, color='#f57c00', linewidth=2.2, linestyle='-.', label='Classic User-CF Model')
                    
                    # 🟦 纯随机 Baseline 线
                    plt.plot(loops, random_traj, color='#1976d2', linewidth=1.8, linestyle='--', label='Pure Random Baseline')
                    
                    # 细节美化
                    plt.title(f"Normative Moral Divergence Shifting — Playlist ID: [{pid}]\nMulti-Algorithm Long-Term Evolution Comparison (RADio Framework)", 
                              fontsize=11, fontweight='bold')
                    plt.xlabel("Simulation Feedback Loops (Temporal Iterations)", fontsize=10)
                    plt.ylabel("Rank-Aware JS Divergence Metric Score [0, 1]", fontsize=10)
                    plt.ylim(-0.05, 1.05)
                    plt.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='upper right')
                    plt.tight_layout()
                    
                    user_save_path = os.path.join(BASE_DIR, f"part2_playlist_{pid}_three_models_comparison.png")
                    plt.savefig(user_save_path, dpi=300)
                    print(f"🎨 [SUCCESS] Comparison chart for Playlist [{pid}] has been exported to:\n{user_save_path}")
                    plt.show()
                    plt.close()
                    
                    simulated_count += 1
                if simulated_count >= 5: break
                    
        conn.commit()
        conn.close()
        print("\n✅ [COMPLETED] Multi-file project simulation loop ended successfully!")