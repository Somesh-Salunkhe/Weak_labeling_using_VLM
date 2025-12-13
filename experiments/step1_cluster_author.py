# ------------------------------------------------------------------------
# Weak Labeling using VLM - Step 1: Clustering (Config-Based)
# ------------------------------------------------------------------------

import sys
import os
import argparse
import warnings
import json
import numpy as np
from pprint import pprint

# --- IMPORT SETUP ---
current_script_path = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(experiments_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clustering import apply_gmm
    from utils.os_utils import load_config
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules from {project_root}")
    print(f"   Details: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

def main(args):
    # Load Config
    # Note: We use the 4sec config
    config_path = os.path.join(project_root, 'configs', args.dataset, f'annotation_pipeline_{args.clip_length}sec.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found at: {config_path}")
        # Fallback to local file if path resolution fails
        if os.path.exists(f'annotation_pipeline_{args.clip_length}sec.yaml'):
             config_path = f'annotation_pipeline_{args.clip_length}sec.yaml'
        else:
             return

    config = load_config(config_path)
    
    # Override config params
    config['init_rand_seed'] = args.seed
    config['clustering']['nb_clusters'] = args.cluster
    
    output_dir = os.path.join(project_root, 'output', 'clusters')
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Clustering with Config: {os.path.basename(config_path)} ---")

    # --- MAIN LOOP ---
    # We iterate 18 times for the 18 subjects
    for i in range(18):
        sbj_id = f"sbj_{i}"
        print(f"\nProcessing Subject {sbj_id} ({i+1}/18)")
        
        # Initialize containers
        conv3d_emb = np.empty((0, config['dataset']['conv3d_dim']))
        raft_emb = np.empty((0, config['dataset']['raft_dim']))
        dino_emb = np.empty((0, config['dataset']['dino_dim']))
        clip_emb = np.empty((0, config['dataset']['clip_dim']))

        # Helper to get folder path from config
        def get_folder(key):
            path = config['dataset'][key]
            if path.startswith('./'): path = path[2:] # Remove ./ prefix
            return os.path.join(project_root, path)

        # Load Features
        try:
            # We construct the filename. Author's code usually expects 'sbj_X.npy'
            fname = f"{sbj_id}.npy"
            
            c3d_path = os.path.join(get_folder('conv3d_folder'), fname)
            r_path   = os.path.join(get_folder('raft_folder'), fname)
            d_path   = os.path.join(get_folder('dino_folder'), fname)
            c_path   = os.path.join(get_folder('clip_folder'), fname)

            # Check if files exist before loading to give clear errors
            if not os.path.exists(c_path):
                print(f"File not found: {c_path}")

                # Try 'sbj_01' format instead of 'sbj_0' if that fails
                fname_alt = f"sbj_{i:02d}.npy"
                c_path_alt = os.path.join(get_folder('clip_folder'), fname_alt)
                if os.path.exists(c_path_alt):
                    print(f"  -> Found alternate name: {fname_alt}")
                    fname = fname_alt
                    c3d_path = os.path.join(get_folder('conv3d_folder'), fname)
                    r_path   = os.path.join(get_folder('raft_folder'), fname)
                    d_path   = os.path.join(get_folder('dino_folder'), fname)
                    c_path   = os.path.join(get_folder('clip_folder'), fname)
                else:
                    continue

            # Load
            c3d_data = np.load(c3d_path).astype(np.float32)
            r_data = np.load(r_path).astype(np.float32)
            d_data = np.load(d_path).astype(np.float32)
            c_data = np.load(c_path).astype(np.float32)

            conv3d_emb = np.append(conv3d_emb, c3d_data, axis=0)
            raft_emb = np.append(raft_emb, r_data, axis=0)
            dino_emb = np.append(dino_emb, d_data, axis=0)
            clip_emb = np.append(clip_emb, c_data, axis=0)

        except Exception as e:
            print(f"Error loading {sbj_id}: {e}")
            continue

        # Align lengths
        min_len = min(len(conv3d_emb), len(raft_emb), len(dino_emb), len(clip_emb))
        dino_emb = dino_emb[:min_len]
        clip_emb = clip_emb[:min_len]
        raft_emb = raft_emb[:min_len]
        conv3d_emb = conv3d_emb[:min_len]

        # Select Features 
        if args.features == 'clip': emb_data = clip_emb
        elif args.features == 'dino': emb_data = dino_emb
        else: emb_data = clip_emb # Default fallback

        # Clustering
        print(f"  Running GMM Clustering on {emb_data.shape} matrix...")
        try:
            v_cluster_labels, v_cluster_dist, gmm_model = apply_gmm(
                emb_data, 
                config['clustering']['nb_clusters'], 
                seed=config['init_rand_seed']
            )
        except Exception as e:
            print(f"Clustering failed: {e}")
            continue
        
        # Save Results
        if hasattr(gmm_model, 'means_'):
            centroids = gmm_model.means_
        else:
            centroids = np.zeros((config['clustering']['nb_clusters'], emb_data.shape[1]))

        # Save using subject name
        centroids_path = os.path.join(output_dir, f'{sbj_id}_centroids.npy')
        labels_path = os.path.join(output_dir, f'{sbj_id}_labels.npy')
        
        np.save(centroids_path, centroids)
        np.save(labels_path, v_cluster_labels)
        print(f"Saved results for {sbj_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wear', help='dataset name')
    parser.add_argument('--features', type=str, default='clip', help='features to use')
    parser.add_argument('--cluster', type=int, default=100, help='number of clusters')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    # Change default to 4 
    parser.add_argument('--clip_length', type=int, default=4, help='clip length in seconds') 
    
    args = parser.parse_args()
    main(args)
