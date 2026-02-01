"""
Data Cleaning Script for NBA First Basket Scorer Prediction

This script loads raw jump ball data, cleans it, and prepares it for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and standardizes raw NBA jump ball data."""

    def __init__(self, raw_data_path: str = "data/raw", output_path: str = "data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None

    def load_all_seasons(self) -> pd.DataFrame:
        """Load all season parquet files and combine them."""
        parquet_files = sorted(self.raw_data_path.glob("bball_ref_20*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No season parquet files found in {self.raw_data_path}")

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            # Extract season from filename (e.g., bball_ref_2024_25.parquet -> 2024-25)
            season = f.stem.replace("bball_ref_", "").replace("_", "-")
            df['season'] = season
            dfs.append(df)
            logger.info(f"Loaded {len(df)} games from {f.name} (season {season})")

        self.raw_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total: {len(self.raw_data)} games loaded across {len(dfs)} seasons")

        return self.raw_data

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw data:
        - Remove rows with missing jump ball data
        - Standardize column names and types
        - Parse starter lineups from JSON
        - Derive jump ball winner/loser columns
        - Handle edge cases
        """
        if self.raw_data is None:
            raise ValueError("Must load data first with load_all_seasons()")

        df = self.raw_data.copy()
        logger.info(f"Starting cleaning with {len(df)} rows")

        # 1. Remove rows with missing jump ball data
        jump_ball_cols = ['jump_ball_player1_id', 'jump_ball_player1_name',
                         'jump_ball_player2_id', 'jump_ball_player2_name',
                         'jump_ball_winning_team']

        missing_jb = df[jump_ball_cols].isnull().any(axis=1)
        df = df[~missing_jb].copy()
        logger.info(f"Removed {missing_jb.sum()} rows with missing jump ball data, {len(df)} remaining")

        # 2. Derive jump ball winner and loser
        df = self._derive_jump_ball_winner_loser(df)

        # 3. Parse starter lineups from JSON strings
        df = self._parse_starter_lineups(df)

        # 4. Standardize first score type
        df['first_score_type'] = df['first_score_type'].fillna('UNKNOWN')
        df['first_score_type'] = df['first_score_type'].str.upper().str.strip()

        # 5. Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # 6. Create derived features
        df = self._create_derived_features(df)

        # 7. Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        self.cleaned_data = df
        logger.info(f"Cleaning complete: {len(df)} rows")

        return self.cleaned_data

    def _derive_jump_ball_winner_loser(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine which player won and lost the jump ball based on possession.

        The jump ball winner is the player whose team gained possession.
        """
        # Determine which player is on which team by checking starters
        def get_jump_ball_result(row):
            winning_team = row['jump_ball_winning_team']
            p1_id = row['jump_ball_player1_id']
            p1_name = row['jump_ball_player1_name']
            p2_id = row['jump_ball_player2_id']
            p2_name = row['jump_ball_player2_name']
            home_team = row['home_team']
            away_team = row['away_team']
            home_starter_ids = str(row.get('home_starter_ids', ''))
            away_starter_ids = str(row.get('away_starter_ids', ''))

            # Check which player is on the winning team
            p1_is_home = p1_id in home_starter_ids if home_starter_ids else False
            p1_is_away = p1_id in away_starter_ids if away_starter_ids else False
            p2_is_home = p2_id in home_starter_ids if home_starter_ids else False
            p2_is_away = p2_id in away_starter_ids if away_starter_ids else False

            # Determine p1's team
            if p1_is_home:
                p1_team = home_team
            elif p1_is_away:
                p1_team = away_team
            else:
                # Fallback: check if winning team matches
                p1_team = None

            # Determine p2's team
            if p2_is_home:
                p2_team = home_team
            elif p2_is_away:
                p2_team = away_team
            else:
                p2_team = None

            # Assign winner/loser based on winning team
            if p1_team == winning_team:
                return pd.Series({
                    'jump_ball_winner_id': p1_id,
                    'jump_ball_winner_name': p1_name,
                    'jump_ball_loser_id': p2_id,
                    'jump_ball_loser_name': p2_name,
                    'jump_ball_winner_team': p1_team,
                    'jump_ball_loser_team': p2_team
                })
            elif p2_team == winning_team:
                return pd.Series({
                    'jump_ball_winner_id': p2_id,
                    'jump_ball_winner_name': p2_name,
                    'jump_ball_loser_id': p1_id,
                    'jump_ball_loser_name': p1_name,
                    'jump_ball_winner_team': p2_team,
                    'jump_ball_loser_team': p1_team
                })
            else:
                # If we can't determine, use possession_to info
                poss_id = row.get('jump_ball_possession_to_id')
                if poss_id:
                    # The player who the ball went to is on the winning team
                    # So the jump ball participant on that team won
                    if p1_team and p1_team == winning_team:
                        return pd.Series({
                            'jump_ball_winner_id': p1_id,
                            'jump_ball_winner_name': p1_name,
                            'jump_ball_loser_id': p2_id,
                            'jump_ball_loser_name': p2_name,
                            'jump_ball_winner_team': winning_team,
                            'jump_ball_loser_team': away_team if winning_team == home_team else home_team
                        })

                # Last resort: use player1 as winner if they're on winning team
                return pd.Series({
                    'jump_ball_winner_id': p1_id,
                    'jump_ball_winner_name': p1_name,
                    'jump_ball_loser_id': p2_id,
                    'jump_ball_loser_name': p2_name,
                    'jump_ball_winner_team': winning_team,
                    'jump_ball_loser_team': away_team if winning_team == home_team else home_team
                })

        logger.info("Deriving jump ball winner/loser...")
        result = df.apply(get_jump_ball_result, axis=1)
        df = pd.concat([df, result], axis=1)

        return df

    def _parse_starter_lineups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse starter lineup data into structured lists."""

        def parse_starters(starters):
            """Convert starters data (numpy array, list, or JSON string) to list of dicts."""
            if starters is None:
                return []
            # Handle numpy arrays
            if isinstance(starters, np.ndarray):
                return list(starters)
            if isinstance(starters, list):
                return starters
            # Handle string (JSON)
            if isinstance(starters, str):
                try:
                    return json.loads(starters)
                except (json.JSONDecodeError, TypeError):
                    return []
            return []

        def extract_names(starters):
            """Extract player names from starters list."""
            parsed = parse_starters(starters)
            return [p.get('name', '') for p in parsed if isinstance(p, dict)]

        def extract_ids(starters):
            """Extract player IDs from starters list."""
            parsed = parse_starters(starters)
            return [p.get('id', '') for p in parsed if isinstance(p, dict)]

        # Extract starter names as lists
        df['home_starter_names'] = df['home_starters'].apply(extract_names)
        df['away_starter_names'] = df['away_starters'].apply(extract_names)

        # Extract starter IDs as lists
        df['home_starter_id_list'] = df['home_starters'].apply(extract_ids)
        df['away_starter_id_list'] = df['away_starters'].apply(extract_ids)

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional derived features for analysis."""

        # 1. First scorer is from home team
        df['first_scorer_is_home'] = df['first_scorer_team'] == df['home_team']

        # 2. First scorer is from tip winning team
        df['first_scorer_is_tip_winner'] = df['first_scorer_team'] == df['jump_ball_winning_team']

        # 3. Jump ball winner is home player
        df['jump_ball_winner_is_home'] = df['jump_ball_winner_team'] == df['home_team']

        # 4. First scorer was jump ball participant
        df['first_scorer_was_jb_participant'] = (
            (df['first_scorer_id'] == df['jump_ball_player1_id']) |
            (df['first_scorer_id'] == df['jump_ball_player2_id'])
        )

        # 5. Day of week (games on different days might have different patterns)
        df['day_of_week'] = df['date'].dt.dayofweek

        # 6. Month (seasonal patterns)
        df['month'] = df['date'].dt.month

        # 7. First scorer position in lineup (1-5 for home, 6-10 for away, 0 if not found)
        def get_first_scorer_position(row):
            fs_id = row['first_scorer_id']
            home_ids = row['home_starter_id_list']
            away_ids = row['away_starter_id_list']

            if fs_id in home_ids:
                return home_ids.index(fs_id) + 1
            elif fs_id in away_ids:
                return away_ids.index(fs_id) + 6
            return 0  # Not found (could be from bench)

        df['first_scorer_lineup_position'] = df.apply(get_first_scorer_position, axis=1)

        return df

    def get_summary_stats(self) -> Dict:
        """Generate summary statistics for the cleaned data."""
        if self.cleaned_data is None:
            raise ValueError("Must clean data first")

        df = self.cleaned_data

        return {
            'total_games': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'seasons': df['season'].unique().tolist(),
            'unique_players': {
                'jump_ball_participants': df['jump_ball_winner_name'].nunique() + df['jump_ball_loser_name'].nunique(),
                'first_scorers': df['first_scorer_name'].nunique()
            },
            'teams': sorted(df['home_team'].unique().tolist()),
            'tip_winner_scores_first_rate': df['tip_winner_scored_first'].mean(),
            'first_score_type_distribution': df['first_score_type'].value_counts(normalize=True).to_dict(),
            'home_team_scores_first_rate': df['first_scorer_is_home'].mean(),
            'first_scorer_was_jb_participant_rate': df['first_scorer_was_jb_participant'].mean()
        }

    def save_cleaned_data(self, filename: str = "cleaned_games.parquet") -> Path:
        """Save cleaned data to parquet file."""
        if self.cleaned_data is None:
            raise ValueError("Must clean data first")

        output_file = self.output_path / filename

        # Select columns to save (exclude parsed lists which don't serialize well)
        columns_to_save = [
            'game_id', 'date', 'season', 'home_team', 'away_team',
            'jump_ball_player1_id', 'jump_ball_player1_name',
            'jump_ball_player2_id', 'jump_ball_player2_name',
            'jump_ball_winner_id', 'jump_ball_winner_name',
            'jump_ball_loser_id', 'jump_ball_loser_name',
            'jump_ball_winner_team', 'jump_ball_loser_team',
            'jump_ball_winning_team',
            'first_scorer_id', 'first_scorer_name', 'first_scorer_team',
            'first_score_type', 'first_score_description',
            'tip_winner_scored_first',
            'first_scorer_is_home', 'first_scorer_is_tip_winner',
            'jump_ball_winner_is_home', 'first_scorer_was_jb_participant',
            'day_of_week', 'month', 'first_scorer_lineup_position',
            'home_starter_ids', 'away_starter_ids',
            'home_starters', 'away_starters'
        ]

        # Only include columns that exist
        columns_to_save = [c for c in columns_to_save if c in self.cleaned_data.columns]

        self.cleaned_data[columns_to_save].to_parquet(output_file, index=False)
        logger.info(f"Saved cleaned data to {output_file}")

        return output_file


def main():
    """Main function to run data cleaning."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Data Cleaning")
    logger.info("=" * 60)

    cleaner = DataCleaner()

    # Load data
    cleaner.load_all_seasons()

    # Clean data
    cleaner.clean_data()

    # Print summary
    stats = cleaner.get_summary_stats()
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # Save cleaned data
    output_file = cleaner.save_cleaned_data()
    logger.info(f"\nCleaned data saved to: {output_file}")

    return cleaner


if __name__ == "__main__":
    main()
