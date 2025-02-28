#!/usr/bin/env python3
"""Script to evaluate conversation clustering performance of different models.

This script takes a directory containing label files from different models and a ground truth file,
calculates the Adjusted Rand Index (ARI) for conversation clustering, and outputs the results to a CSV file.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import argparse
import logging
import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    """Load ground truth conversation labels."""
    try:
        ground_truth = pd.read_csv(ground_truth_path)
        required_columns = {'id', 'conv_id'}
        if not all(col in ground_truth.columns for col in required_columns):
            raise ValueError(f"Ground truth file must contain columns: {required_columns}")
        # Rename columns to match the format we use
        ground_truth = ground_truth.rename(columns={
            'id': 'message_id',
            'conv_id': 'conversation_id'
        })
        return ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth file: {e}")
        raise

def load_model_predictions(label_file: Path) -> pd.DataFrame:
    """Load model predictions for conversation clustering."""
    try:
        df = pd.read_csv(label_file)
        required_columns = {'message_id', 'conversation_id'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Label file must contain columns: {required_columns}")
        if df.empty:
            logger.warning(f"Label file {label_file} contains zero messages.")
            return None
        return df[['message_id', 'conversation_id']]
    except Exception as e:
        logger.error(f"Error loading predictions from {label_file}: {e}")
        raise

def calculate_ari(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Calculate Adjusted Rand Index for conversation clustering."""
    # Merge ground truth with predictions
    merged = pd.merge(
        ground_truth,
        predictions,
        on='message_id',
        how='inner',
        suffixes=('_true', '_pred')
    )

    # Drop any rows with NaN values
    merged = merged.dropna(subset=['conversation_id_true', 'conversation_id_pred'])

    # Calculate ARI
    ari = adjusted_rand_score(
        merged['conversation_id_true'],
        merged['conversation_id_pred']
    )

    return {
        'ari': ari,
        'n_messages': len(merged)  # Include number of messages for reference
    }

def extract_model_name(label_file: Path) -> str:
    """Extract model name from label file name."""
    # Expected format: labels_YYYYMMDD_modelname_groupname.csv
    parts = label_file.stem.split('_')
    if len(parts) >= 3:
        return parts[2]  # Get the model name part
    return label_file.stem  # Fallback to full stem if pattern doesn't match

def evaluate_conversation_clustering(group_dir: str) -> None:
    """Evaluate conversation clustering performance for all models in the directory."""
    group_path = Path(group_dir)
    if not group_path.exists():
        raise ValueError(f"Directory does not exist: {group_dir}")

    # Get group name from directory path
    group_name = group_path.name

    # Find ground truth file with group name
    ground_truth_path = group_path / f"GT_conversations_{group_name}.csv"
    if not ground_truth_path.exists():
        raise ValueError(f"Ground truth file not found: {ground_truth_path}")

    # Load ground truth
    logger.info(f"Loading ground truth from {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)

    # Find all label files
    label_files = list(group_path.glob("labels_*.csv"))
    if not label_files:
        raise ValueError(f"No label files found in {group_dir}")

    # Calculate metrics for each model
    results = []
    for label_file in label_files:
        model_name = extract_model_name(label_file)
        logger.info(f"Processing predictions from model: {model_name}")

        try:
            predictions = load_model_predictions(label_file)
            if predictions is None:
                continue
            metrics = calculate_ari(ground_truth, predictions)

            # Add model name and file info to metrics
            metrics['model'] = model_name
            metrics['label_file'] = label_file.name
            results.append(metrics)

        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")
            continue

    if not results:
        raise ValueError("No results could be calculated from any label file")

    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(results)

    # Reorder columns to put model first
    columns = ['model', 'label_file', 'ari', 'n_messages']
    results_df = results_df[columns]

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Save results with timestamp
    output_file = group_path / f"metrics_conversations_{group_name}_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved metrics to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate conversation clustering metrics (ARI) for different models.'
    )
    parser.add_argument(
        'group_dir',
        help='Path to directory containing label files and ground truth'
    )

    args = parser.parse_args()
    evaluate_conversation_clustering(args.group_dir)

if __name__ == "__main__":
    main()
