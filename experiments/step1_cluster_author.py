# ------------------------------------------------------------------------
# Weak Labeling using VLM - Step 1: Clustering (Author's Method)
# ------------------------------------------------------------------------

import sys
import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
from pprint import pprint

# --- IMPORT SETUP ---
# 1. Calculate paths
current_script_path = os.path.abspath(__file__)          # .../experiments/step1_cluster_author.py
experiments_dir = os.path.dirname(current_script_path)   # .../experiments
project_root = os.path.dirname(experiments_dir)          # .../Weak_labeling_using_VLM

# 2. Add project root to sys.path so we can import 'clustering.py' and 'utils/' located there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Import author's modules
try:
    from clustering import apply_gmm
    from utils.os_utils import load_config  # Assuming utils is also in root
    print(f"✅ Successfully imported modules from {project_root}")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import modules from {project_root}")
    print(f"   Details: {e}")
    print("   Ensure 'clustering.py' and the 'utils' folder are in the project root.")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Set numpy print options
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def main(args):
    """
    Main function that performs ONLY clustering and saves the results.
    """
    
    # --- CONFIGURATION & SETUP ---
    # Construct config path relative to project root
    config_path = os.path.join(project_root, 'configs', args.dataset, f'annotation_pipeline_{args.clip_length}sec.yaml')
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at: {config_path}")
        return

    # Load Config
    config = load_config(config_path)
    
    # Override config with command line args
    config['init_rand_seed'] = args.seed
    config['clustering']['nb_clusters'] = args.cluster
    
    # Output directory
    output_dir = os.path.join(project_root, 'output', 'clusters')
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Clustering for {args.dataset.upper()} ---")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Clusters: {args.cluster}")

    # --- DATA LOADING ---
    
    # Iterate over all annotation splits defined in the config
    # The config usually contains a list of JSON files defining train/val splits
    for i, anno_split_rel in enumerate(config['anno_json']):
        print(f"\nProcessing Split {i + 1} / {len(config['anno_json'])}")
        
        # Resolve full path to annotation json
        anno_split_path = os.path.join(project_root, anno_split_rel)
        if not os.path.exists(anno_split_path):
             # Try assuming path is absolute or relative to data folder if first fail
             anno_split_path = anno_split_rel
        
        try:
            with open(anno_split_path) as f:
                file = json.load(f)
                anno_file = file['database']
        except FileNotFoundError:
             print(f"Skipping split, file not found: {anno_split_path}")
             continue
            
        # Identify subjects for this split (using 'Validation' set as per original script logic)
        train_names = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        
        # Initialize containers for embeddings
        # We handle all feature types to be safe, though we likely only use CLIP
        conv3d_emb = np.empty((0, config['dataset']['conv3d_dim']))
        raft_emb = np.empty((0, config['dataset']['raft_dim']))
        dino_emb = np.empty((0, config['dataset']['dino_dim']))
        clip_emb = np.empty((0, config['dataset']['clip_dim']))

        print(f"  Loading data for {len(train_names)} subjects...")

        # Load data for each subject
        for sbj in train_names:
            # Construct paths (Handling relative paths in config)
            # We assume config paths like 'data/WEAR/processed' are relative to project root
            def get_path(folder_key):
                p = config['dataset'][folder_key]
                return p if os.path.isabs(p) else os.path.join(project_root, p)

            # Load features
            # Note: We use try-except to handle missing files gracefully
            try:
                c3d_data = np.load(os.path.join(get_path('conv3d_folder'), sbj + '.npy')).astype(np.float32)
                r_data = np.load(os.path.join(get_path('raft_folder'), sbj + '.npy')).astype(np.float32)
                d_data = np.load(os.path.join(get_path('dino_folder'), sbj + '.npy')).astype(np.float32)
                c_data = np.load(os.path.join(get_path('clip_folder'), sbj + '.npy')).astype(np.float32)

                conv3d_emb = np.append(conv3d_emb, c3d_data, axis=0)
                raft_emb = np.append(raft_emb, r_data, axis=0)
                dino_emb = np.append(dino_emb, d_data, axis=0)
                clip_emb = np.append(clip_emb, c_data, axis=0)
            except FileNotFoundError as e:
                print(f"    Warning: Missing file for {sbj} ({e.filename}). Skipping.")

        # If no data loaded, skip
        if len(clip_emb) == 0:
            print("  No data loaded for this split. Skipping.")
            continue

        # Align lengths (Author's logic to handle frame mismatches)
        min_len = min(len(conv3d_emb), len(raft_emb), len(dino_emb), len(clip_emb))
        dino_emb = dino_emb[:min_len]
        clip_emb = clip_emb[:min_len]
        raft_emb = raft_emb[:min_len]
        conv3d_emb = conv3d_emb[:min_len]

        # Select Features based on arguments
        if args.features == 'dino':
            emb_data = dino_emb
        elif args.features == 'clip':
            emb_data = clip_emb
        elif args.features == 'raft':
            emb_data = raft_emb
        elif args.features == 'conv3d':
            emb_data = conv3d_emb
        elif args.features == 'i3d':
            emb_data = np.concatenate((conv3d_emb, raft_emb), axis=1)
        elif args.features == 'dino+raft':
            emb_data = np.concatenate((raft_emb, dino_emb), axis=1)
        elif args.features == 'clip+raft':
            emb_data = np.concatenate((clip_emb, raft_emb), axis=1)
        else:
             print(f"Unknown feature type: {args.features}")
             return

        # --- CORE TASK: CLUSTERING ---
        print(f"  Running GMM Clustering on {emb_data.shape} matrix...")
        
        # Call author's function
        # Expects: v_cluster_labels, v_cluster_dist, gmm_model
        try:
            v_cluster_labels, v_cluster_dist, gmm_model = apply_gmm(
                emb_data, 
                config['clustering']['nb_clusters'], 
                seed=config['init_rand_seed']
            )
        except Exception as e:
            print(f"  ❌ Clustering failed: {e}")
            continue
        
        # --- SAVE RESULTS ---
        # We save the centroids (means) from the GMM model
        if hasattr(gmm_model, 'means_'):
            centroids = gmm_model.means_
        else:
            # Fallback if model isn't returned or has different attrib
            # If the author's apply_gmm doesn't return the model object, we can't get centroids easily
            # But usually sklearn GMM object is returned as 3rd arg
            centroids = np.zeros((config['clustering']['nb_clusters'], emb_data.shape[1]))
            print("  Warning: Could not extract centroids from GMM model directly. Saving zeros.")

        # Save files
        split_name = os.path.basename(anno_split_rel).replace('.json', '')
        
        centroids_path = os.path.join(output_dir, f'{split_name}_centroids.npy')
        labels_path = os.path.join(output_dir, f'{split_name}_labels.npy')
        
        np.save(centroids_path, centroids)
        np.save(labels_path, v_cluster_labels)
        
        print(f"✅ Results saved:")
        print(f"   - {centroids_path}")
        print(f"   - {labels_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wear', help='dataset to use (folder name in configs/)')
    parser.add_argument('--features', type=str, default='clip', help='features to use')
    parser.add_argument('--cluster', type=int, default=100, help='number of clusters')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--clip_length', type=int, default=2, help='clip length in seconds')
    
    # Arguments required by config loading but unused in this specific script
    # We keep them to prevent errors if load_config relies on them
    parser.add_argument('--supervision', type=str, default='weak')
    parser.add_argument('--training_type', type=str, default='ul')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--sampling_strategy', type=str, default='nearest')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.0)
    
    args = parser.parse_args()
    main(args)
