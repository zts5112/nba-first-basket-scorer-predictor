"""
Unified ML Data Preparation Pipeline

This script orchestrates the entire data preparation process:
1. Clean raw data
2. Tokenize players and teams
3. Engineer features
4. Create train/val/test splits

Run this script to prepare all data for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime

from src.data.clean_data import DataCleaner
from src.data.tokenize_players import PlayerTokenizer
from src.data.prepare_features import FeatureBuilder, DatasetBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_all_data(
    raw_data_path: str = "data/raw",
    output_path: str = "data/processed",
    train_end_date: str = "2024-06-01",
    val_end_date: str = "2025-06-01",
    min_player_appearances: int = 1
) -> dict:
    """
    Run the complete data preparation pipeline.

    Args:
        raw_data_path: Path to raw parquet files
        output_path: Path to save processed data
        train_end_date: End date for training data (exclusive)
        val_end_date: End date for validation data (exclusive)
        min_player_appearances: Minimum appearances for player tokenization

    Returns:
        Dictionary with summary statistics and file paths
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {'files': {}, 'stats': {}}

    # Step 1: Clean data
    logger.info("=" * 60)
    logger.info("STEP 1: Data Cleaning")
    logger.info("=" * 60)

    cleaner = DataCleaner(raw_data_path, output_path)
    cleaner.load_all_seasons()
    cleaned_df = cleaner.clean_data()

    cleaned_file = cleaner.save_cleaned_data("cleaned_games.parquet")
    results['files']['cleaned_data'] = str(cleaned_file)
    results['stats']['cleaning'] = cleaner.get_summary_stats()

    # Step 2: Tokenize players
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Player Tokenization")
    logger.info("=" * 60)

    # Re-parse starter lists for tokenization
    import json

    def parse_starters(starters):
        if starters is None:
            return []
        if isinstance(starters, np.ndarray):
            return list(starters)
        if isinstance(starters, list):
            return starters
        if isinstance(starters, str):
            try:
                return json.loads(starters)
            except:
                return []
        return []

    def extract_ids(starters):
        parsed = parse_starters(starters)
        return [p.get('id', '') for p in parsed if isinstance(p, dict)]

    def extract_names(starters):
        parsed = parse_starters(starters)
        return [p.get('name', '') for p in parsed if isinstance(p, dict)]

    cleaned_df['home_starter_id_list'] = cleaned_df['home_starters'].apply(extract_ids)
    cleaned_df['away_starter_id_list'] = cleaned_df['away_starters'].apply(extract_ids)
    cleaned_df['home_starter_names'] = cleaned_df['home_starters'].apply(extract_names)
    cleaned_df['away_starter_names'] = cleaned_df['away_starters'].apply(extract_names)

    tokenizer = PlayerTokenizer()
    tokenizer.fit(cleaned_df, min_appearances=min_player_appearances)
    tokenizer_files = tokenizer.save(output_path)

    results['files'].update({k: str(v) for k, v in tokenizer_files.items()})
    results['stats']['tokenization'] = {
        'player_vocab_size': tokenizer.vocab_size,
        'team_vocab_size': tokenizer.team_vocab_size
    }

    # Step 3: Feature Engineering
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)

    feature_builder = FeatureBuilder(tokenizer)
    features_df = feature_builder.compute_historical_features(cleaned_df)

    features_file = output_dir / "features.parquet"
    features_df.to_parquet(features_file, index=False)
    results['files']['features'] = str(features_file)

    # Step 4: Create splits
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating Train/Val/Test Splits")
    logger.info("=" * 60)

    dataset_builder = DatasetBuilder(features_df, cleaned_df)
    train_df, val_df, test_df = dataset_builder.create_temporal_splits(
        train_end_date=train_end_date,
        val_end_date=val_end_date
    )

    train_file = output_dir / "train.parquet"
    val_file = output_dir / "val.parquet"
    test_file = output_dir / "test.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)

    results['files']['train'] = str(train_file)
    results['files']['val'] = str(val_file)
    results['files']['test'] = str(test_file)

    results['stats']['splits'] = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_date_range': {
            'start': train_df['date'].min().strftime('%Y-%m-%d'),
            'end': train_df['date'].max().strftime('%Y-%m-%d')
        },
        'val_date_range': {
            'start': val_df['date'].min().strftime('%Y-%m-%d'),
            'end': val_df['date'].max().strftime('%Y-%m-%d')
        },
        'test_date_range': {
            'start': test_df['date'].min().strftime('%Y-%m-%d'),
            'end': test_df['date'].max().strftime('%Y-%m-%d')
        }
    }

    # Get feature columns info
    feature_cols = dataset_builder.get_feature_columns()
    results['stats']['features'] = {
        'total_features': len(features_df.columns),
        'feature_types': {k: len(v) for k, v in feature_cols.items()}
    }

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("\nFiles created:")
    for name, path in results['files'].items():
        logger.info(f"  {name}: {path}")

    logger.info("\nData statistics:")
    logger.info(f"  Total games: {results['stats']['cleaning']['total_games']}")
    logger.info(f"  Player vocabulary: {results['stats']['tokenization']['player_vocab_size']}")
    logger.info(f"  Team vocabulary: {results['stats']['tokenization']['team_vocab_size']}")
    logger.info(f"  Total features: {results['stats']['features']['total_features']}")
    logger.info(f"  Train/Val/Test: {results['stats']['splits']['train_size']}/"
               f"{results['stats']['splits']['val_size']}/{results['stats']['splits']['test_size']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Prepare NBA data for ML training')
    parser.add_argument('--raw-path', default='data/raw', help='Path to raw data')
    parser.add_argument('--output-path', default='data/processed', help='Output path')
    parser.add_argument('--train-end', default='2024-06-01', help='Training end date')
    parser.add_argument('--val-end', default='2025-06-01', help='Validation end date')
    parser.add_argument('--min-appearances', type=int, default=1,
                       help='Minimum player appearances for tokenization')

    args = parser.parse_args()

    results = prepare_all_data(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        train_end_date=args.train_end,
        val_end_date=args.val_end,
        min_player_appearances=args.min_appearances
    )

    return results


if __name__ == "__main__":
    main()
