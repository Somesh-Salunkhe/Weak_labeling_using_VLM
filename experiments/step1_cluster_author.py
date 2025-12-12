# ------------------------------------------------------------------------
# Modified script to ONLY perform clustering using author's logic
# ------------------------------------------------------------------------

import warnings
from clustering import apply_gmm  # Using author's GMM function
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pprint import pprint
from utils.os_utils import load_config, Logger

# set the precision of numpy arrays to 2 decimal places
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def main(args):
    """
    Main function that performs ONLY clustering and saves the results.
    """
    
    # --- CONFIGURATION & SETUP ---
    args.config = 'configs/{}/annotation_pipeline_{}sec.yaml'.format(args.dataset, args.clip_length)
    
    # We will save results to a dedicated folder
    output_dir = 'output/clusters'
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['clustering']['nb_clusters'] = args.cluster

    # Setup Logging (minimal)
    print(f"--- Starting Clustering for {args.dataset} ---")
    pprint(config['clustering'])

    # --- DATA LOADING & CLUSTERING ---
    
    # Iterate over all subjects defined in the config
    for i, anno_split in enumerate(config['anno_json']):
        print(f"\nProcessing Split {i + 1} / {len(config['anno_json'])}")
        
        # Load the split info
        with open(anno_split) as f:
            file = json.load(f)
            anno_file = file['database']
            
        # Identify subjects for this split
        # (The author uses 'Validation' subset for the 'test_annotation' script logic)
        train_names = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        
        # Initialize containers for this split
        # We only need the embeddings for clustering
        conv3d_emb = np.empty((0, config['dataset']['conv3d_dim']))
        raft_emb = np.empty((0, config['dataset']['raft_dim']))
        dino_emb = np.empty((0, config['dataset']['dino_dim']))
        clip_emb = np.empty((0, config['dataset']['clip_dim']))

        # Load data for each subject in this split
        for sbj in train_names:
            # We don't need sensor data for clustering, just the visual embeddings
            c3d_data = np.load(os.path.join(config['dataset']['conv3d_folder'], sbj + '.npy')).astype(np.float32)
            r_data = np.load(os.path.join(config['dataset']['raft_folder'], sbj + '.npy')).astype(np.float32)
            d_data = np.load(os.path.join(config['dataset']['dino_folder'], sbj + '.npy')).astype(np.float32)
            c_data = np.load(os.path.join(config['dataset']['clip_folder'], sbj + '.npy')).astype(np.float32)

            conv3d_emb = np.append(conv3d_emb, c3d_data, axis=0)
            raft_emb = np.append(raft_emb, r_data, axis=0)
            dino_emb = np.append(dino_emb, d_data, axis=0)
            clip_emb = np.append(clip_emb, c_data, axis=0)
            
            print(f"  Loaded {sbj}: CLIP shape {c_data.shape}")

        # Align lengths (Author's logic)
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

        # --- CORE TASK: CLUSTERING ---
        print(f"  Running GMM Clustering with {config['clustering']['nb_clusters']} clusters...")
        
        # Call author's function
        # Returns: labels (assignments), dist (distance to center), centers (centroids)
        # Note: 'apply_gmm' might return 2 or 3 values depending on exact version. 
        # Based on your file: v_cluster_labels, v_cluster_dist, _ = apply_gmm(...)
        v_cluster_labels, v_cluster_dist, gmm_model = apply_gmm(
            emb_data, 
            config['clustering']['nb_clusters'], 
            seed=config['init_rand_seed']
        )
        
        # --- SAVE RESULTS ---
        # We save the centroids (means) from the GMM model
        if hasattr(gmm_model, 'means_'):
            centroids = gmm_model.means_
        else:
            # Fallback if model isn't returned or has different attrib
            centroids = np.zeros((config['clustering']['nb_clusters'], emb_data.shape[1]))
            print("Warning: Could not extract centroids from GMM model directly.")

        # Save for this split
        split_name = f"split_{i}"
        np.save(os.path.join(output_dir, f'{split_name}_centroids.npy'), centroids)
        np.save(os.path.join(output_dir, f'{split_name}_labels.npy'), v_cluster_labels)
        
        print(f"✅ Saved results for {split_name} to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wear', help='dataset to use')
    parser.add_argument('--features', type=str, default='clip', help='features to use')
    parser.add_argument('--supervision', type=str, default='weak', help='type of supervision')
    parser.add_argument('--training_type', type=str, default='ul', help='type of training')
    parser.add_argument('--cluster', type=int, default=100, help='number of clusters')
    parser.add_argument('--samples', type=int, default=1, help='number of samples')
    parser.add_argument('--sampling_strategy', type=str, default='nearest', help='sampling strategy')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--loss', type=str, default='ce', help='loss function')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--tau', type=float, default=0.0, help='tau')
    parser.add_argument('--clip_length', type=int, default=2, help='clip length')
    
    args = parser.parse_args()
    main(args)
