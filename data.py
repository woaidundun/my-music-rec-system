import os
import json
import numpy as np
import sqlite3
import torch
from transformers import pipeline
import lyricsgenius
from metrics import MORAL_LABELS

print("Initializing AI components (BART-Large-MNLI & Genius)...")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# 传入先前写死的 Genius TOKEN 资产保护
GENIUS_TOKEN = "Tz1ck7o-X423kUbffOykb1xRD0sbPNerAaTlt9hanD3zn74aY_in_HZ1pML-7BuR"
genius = lyricsgenius.Genius(GENIUS_TOKEN, skip_non_songs=True)

def load_base_data(info_path, embeddings_path):
    """加载声学特征矩阵和大库元数据"""
    embeddings = np.load(embeddings_path)
    with open(info_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    name_to_id, id_to_name = {}, {}
    for uri, info in full_data.items():
        idx = info.get('id')
        if idx is not None:
            name_key = f"{str(info.get('track_name', '')).lower().strip()} - {str(info.get('artist_name', '')).lower().strip()}"
            name_to_id[name_key] = idx
            id_to_name[idx] = {'track_name': info.get('track_name'), 'artist_name': info.get('artist_name')}
            
    return embeddings, name_to_id, id_to_name

def get_track_moral_vector(track_name, artist_name, cursor, conn, allow_network=False):
    """全自动贴标签机：集成了 本地缓存 -> 智能NLP分析 -> 联网Genius兜底"""
    vec = np.zeros(len(MORAL_LABELS))
    try:
        cursor.execute("SELECT moral_tag, lyrics FROM songs WHERE name = ? LIMIT 1", (track_name,))
        row = cursor.fetchone()
        if row:
            tag, lyrics = row
            if tag in MORAL_LABELS:
                vec[MORAL_LABELS.index(tag)] = 1.0
                return vec
            elif lyrics:
                analysis = classifier(lyrics[:400], candidate_labels=MORAL_LABELS)
                best_label = analysis['labels'][0]
                vec[MORAL_LABELS.index(best_label)] = 1.0
                cursor.execute("UPDATE songs SET moral_tag = ? WHERE name = ?", (best_label, track_name))
                conn.commit() 
                return vec
    except Exception:
        pass

    if allow_network:
        try:
            song = genius.search_song(title=track_name, artist=artist_name)
            if song and song.lyrics:
                analysis = classifier(song.lyrics[:400], candidate_labels=MORAL_LABELS)
                best_label = analysis['labels'][0]
                vec[MORAL_LABELS.index(best_label)] = 1.0
                cursor.execute("INSERT INTO songs (name, artist, lyrics, moral_tag, genre) VALUES (?, ?, ?, ?, 'Unknown')", 
                               (track_name, artist_name, song.lyrics, best_label))
                conn.commit() 
                return vec
        except Exception:
            pass

    return np.zeros(len(MORAL_LABELS))

def extract_initial_playlist_state(playlist_tracks, embeddings, name_to_id, cursor, conn):
    """切分出用户前 80% 的听歌历史，并提取初始状态特征"""
    split_idx = int(len(playlist_tracks) * 0.8)
    history_tracks = playlist_tracks[:split_idx]
    
    music_vectors = []
    moral_vectors = []
    seen_ids = set()

    for track in history_tracks:
        raw_title = track['track_name']
        raw_artist = track['artist_name']
        key = f"{raw_title.lower().strip()} - {raw_artist.lower().strip()}"
        
        m_vec = get_track_moral_vector(raw_title, raw_artist, cursor, conn, allow_network=True)
        
        if key in name_to_id:
            track_id = name_to_id[key]
            music_vectors.append(embeddings[track_id])
            if np.sum(m_vec) > 0:
                moral_vectors.append(m_vec)
            seen_ids.add(track_id)
            
    return music_vectors, moral_vectors, seen_ids