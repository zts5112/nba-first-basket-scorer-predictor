"""
Feature Engineering Script for NBA First Basket Scorer Prediction (V2)

This version properly separates pre-game features from outcomes to avoid data leakage.

Key changes from V1:
- Jump ball features are stored by position (home_jb_player, away_jb_player) not outcome
- This allows predicting who wins the jump ball without leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from src.data.tokenize_players import PlayerTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBuilderV2:
    """
    Builds ML features for first basket scorer prediction.

    Key principle: All features must be knowable BEFORE the game starts.
    Outcome-dependent information (who won tip, who scored first) is stored as targets only.
    """

    def __init__(self, tokenizer: Optional[PlayerTokenizer] = None):
        self.tokenizer = tokenizer or PlayerTokenizer.load()

        # Historical statistics (will be computed incrementally)
        self.player_jb_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0
        })
        self.player_fs_stats: Dict[str, Dict] = defaultdict(lambda: {
            'first_scores': 0, 'games_started': 0
        })
        self.team_stats: Dict[str, Dict] = defaultdict(lambda: {
            'tip_wins': 0, 'tip_losses': 0,
            'scored_first_with_tip': 0, 'scored_first_without_tip': 0,
            'total_games': 0
        })
        self.h2h_stats: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
            'p1_wins': 0, 'p2_wins': 0
        })

    def compute_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for each game using only prior data.

        Returns a DataFrame with game_id and all computed features.
        """
        logger.info(f"Computing historical features for {len(df)} games...")

        # Sort by date to ensure chronological processing
        df = df.sort_values('date').reset_index(drop=True)

        features_list = []

        for idx, row in df.iterrows():
            if idx % 500 == 0:
                logger.info(f"  Processing game {idx + 1}/{len(df)}...")

            # Compute features using current historical stats
            features = self._compute_game_features(row)
            features['game_id'] = row['game_id']
            features['date'] = row['date']
            features_list.append(features)

            # Update historical stats with this game's outcome
            self._update_stats(row)

        features_df = pd.DataFrame(features_list)
        logger.info(f"Computed {len(features_df.columns)} features for {len(features_df)} games")

        return features_df

    def _compute_game_features(self, row: pd.Series) -> Dict:
        """Compute all features for a single game."""
        features = {}

        # Basic game info (known before game)
        features['home_team_token'] = self.tokenizer.encode_team(row['home_team'])
        features['away_team_token'] = self.tokenizer.encode_team(row['away_team'])
        features['day_of_week'] = row['day_of_week']
        features['month'] = row['month']

        # Identify home and away jump ball players
        # We need to determine which player is home and which is away
        jb_winner_id = row['jump_ball_winner_id']
        jb_loser_id = row['jump_ball_loser_id']
        jb_winner_team = row['jump_ball_winner_team']
        home_team = row['home_team']

        # Assign home/away based on team, not outcome
        if jb_winner_team == home_team:
            home_jb_id = jb_winner_id
            away_jb_id = jb_loser_id
            home_won_jb = True
        else:
            home_jb_id = jb_loser_id
            away_jb_id = jb_winner_id
            home_won_jb = False

        # Store jump ball player info by position (home/away), not outcome
        features['home_jb_player_token'] = self.tokenizer.encode_player(home_jb_id)
        features['away_jb_player_token'] = self.tokenizer.encode_player(away_jb_id)

        # Historical jump ball stats by position
        home_jb_stats = self.player_jb_stats[home_jb_id]
        away_jb_stats = self.player_jb_stats[away_jb_id]

        features['home_jb_player_win_rate'] = self._safe_rate(
            home_jb_stats['wins'], home_jb_stats['total'], default=0.5
        )
        features['away_jb_player_win_rate'] = self._safe_rate(
            away_jb_stats['wins'], away_jb_stats['total'], default=0.5
        )
        features['home_jb_player_total_jbs'] = home_jb_stats['total']
        features['away_jb_player_total_jbs'] = away_jb_stats['total']

        # Win rate differential (predictive feature)
        features['jb_win_rate_diff'] = features['home_jb_player_win_rate'] - features['away_jb_player_win_rate']

        # Head-to-head record
        h2h_key = tuple(sorted([home_jb_id, away_jb_id]))
        h2h = self.h2h_stats[h2h_key]
        h2h_total = h2h['p1_wins'] + h2h['p2_wins']

        if h2h_total > 0:
            if home_jb_id == h2h_key[0]:
                home_h2h_wins = h2h['p1_wins']
            else:
                home_h2h_wins = h2h['p2_wins']
            features['home_jb_h2h_wins'] = home_h2h_wins
            features['home_jb_h2h_losses'] = h2h_total - home_h2h_wins
            features['home_jb_h2h_win_rate'] = home_h2h_wins / h2h_total
        else:
            features['home_jb_h2h_wins'] = 0
            features['home_jb_h2h_losses'] = 0
            features['home_jb_h2h_win_rate'] = 0.5

        features['h2h_total_matchups'] = h2h_total

        # Predicted jump ball win probability for home player
        features['home_jb_predicted_win_prob'] = self._predict_jb_win_prob(home_jb_id, away_jb_id)

        # Starter info (known before game from lineup announcements)
        home_ids = self._parse_list(row.get('home_starter_id_list', []))
        away_ids = self._parse_list(row.get('away_starter_id_list', []))

        # Pad to 5 players each
        home_ids = (home_ids + [''] * 5)[:5]
        away_ids = (away_ids + [''] * 5)[:5]

        # Encode starters with their historical stats
        for i, pid in enumerate(home_ids):
            features[f'home_starter_{i}_token'] = self.tokenizer.encode_player(pid)
            fs_stats = self.player_fs_stats[pid]
            features[f'home_starter_{i}_fs_rate'] = self._safe_rate(
                fs_stats['first_scores'], fs_stats['games_started']
            )
            features[f'home_starter_{i}_games'] = fs_stats['games_started']

        for i, pid in enumerate(away_ids):
            features[f'away_starter_{i}_token'] = self.tokenizer.encode_player(pid)
            fs_stats = self.player_fs_stats[pid]
            features[f'away_starter_{i}_fs_rate'] = self._safe_rate(
                fs_stats['first_scores'], fs_stats['games_started']
            )
            features[f'away_starter_{i}_games'] = fs_stats['games_started']

        # Team aggregates (sum of starter first scorer rates)
        features['home_team_total_fs_rate'] = sum(
            features[f'home_starter_{i}_fs_rate'] for i in range(5)
        )
        features['away_team_total_fs_rate'] = sum(
            features[f'away_starter_{i}_fs_rate'] for i in range(5)
        )

        # Team historical stats
        home_stats = self.team_stats[row['home_team']]
        away_stats = self.team_stats[row['away_team']]

        features['home_team_tip_win_rate'] = self._safe_rate(
            home_stats['tip_wins'], home_stats['total_games'], default=0.5
        )
        features['away_team_tip_win_rate'] = self._safe_rate(
            away_stats['tip_wins'], away_stats['total_games'], default=0.5
        )
        features['home_team_fs_when_tip_win_rate'] = self._safe_rate(
            home_stats['scored_first_with_tip'], home_stats['tip_wins'], default=0.65
        )
        features['away_team_fs_when_tip_win_rate'] = self._safe_rate(
            away_stats['scored_first_with_tip'], away_stats['tip_wins'], default=0.65
        )

        # === TARGET VARIABLES (outcomes - not features for prediction) ===

        # Jump ball outcome
        features['home_won_tip'] = int(home_won_jb)

        # First scorer outcome
        first_scorer_id = row['first_scorer_id']
        first_scorer_team = row['first_scorer_team']

        features['first_scorer_is_home'] = int(first_scorer_team == home_team)
        features['tip_winner_scored_first'] = int(row['tip_winner_scored_first'])
        features['first_scorer_token'] = self.tokenizer.encode_player(first_scorer_id)
        features['first_score_type'] = row['first_score_type']

        # Which starter position is the first scorer?
        fs_position = -1  # -1 means not a starter
        for i, pid in enumerate(home_ids):
            if pid == first_scorer_id:
                fs_position = i
                break
        for i, pid in enumerate(away_ids):
            if pid == first_scorer_id:
                fs_position = i + 5
                break
        features['first_scorer_position'] = fs_position

        return features

    def _update_stats(self, row: pd.Series):
        """Update historical statistics after processing a game."""
        jb_winner_id = row['jump_ball_winner_id']
        jb_loser_id = row['jump_ball_loser_id']

        # Update jump ball stats
        self.player_jb_stats[jb_winner_id]['wins'] += 1
        self.player_jb_stats[jb_winner_id]['total'] += 1
        self.player_jb_stats[jb_loser_id]['losses'] += 1
        self.player_jb_stats[jb_loser_id]['total'] += 1

        # Update head-to-head
        h2h_key = tuple(sorted([jb_winner_id, jb_loser_id]))
        if jb_winner_id == h2h_key[0]:
            self.h2h_stats[h2h_key]['p1_wins'] += 1
        else:
            self.h2h_stats[h2h_key]['p2_wins'] += 1

        # Update first scorer stats
        first_scorer_id = row['first_scorer_id']
        self.player_fs_stats[first_scorer_id]['first_scores'] += 1

        # Update games started for all starters
        home_ids = self._parse_list(row.get('home_starter_id_list', []))
        away_ids = self._parse_list(row.get('away_starter_id_list', []))
        for pid in home_ids + away_ids:
            if pid:
                self.player_fs_stats[pid]['games_started'] += 1

        # Update team stats
        home_team = row['home_team']
        away_team = row['away_team']
        tip_winner_team = row['jump_ball_winner_team']
        first_scorer_team = row['first_scorer_team']

        self.team_stats[home_team]['total_games'] += 1
        self.team_stats[away_team]['total_games'] += 1

        if tip_winner_team == home_team:
            self.team_stats[home_team]['tip_wins'] += 1
            self.team_stats[away_team]['tip_losses'] += 1
            if first_scorer_team == home_team:
                self.team_stats[home_team]['scored_first_with_tip'] += 1
            else:
                self.team_stats[away_team]['scored_first_without_tip'] += 1
        else:
            self.team_stats[away_team]['tip_wins'] += 1
            self.team_stats[home_team]['tip_losses'] += 1
            if first_scorer_team == away_team:
                self.team_stats[away_team]['scored_first_with_tip'] += 1
            else:
                self.team_stats[home_team]['scored_first_without_tip'] += 1

    def _predict_jb_win_prob(self, home_player_id: str, away_player_id: str) -> float:
        """Predict jump ball win probability for home player using historical stats."""
        home_stats = self.player_jb_stats[home_player_id]
        away_stats = self.player_jb_stats[away_player_id]

        # Bayesian adjusted win rates (prior of 50% with strength 5)
        prior = 0.5
        prior_strength = 5

        home_adj_wr = (home_stats['wins'] + prior_strength * prior) / (
            home_stats['total'] + prior_strength
        ) if home_stats['total'] > 0 else prior

        away_adj_wr = (away_stats['wins'] + prior_strength * prior) / (
            away_stats['total'] + prior_strength
        ) if away_stats['total'] > 0 else prior

        # Log-odds combination
        if 0 < home_adj_wr < 1 and 0 < away_adj_wr < 1:
            home_logit = np.log(home_adj_wr / (1 - home_adj_wr))
            away_logit = np.log(away_adj_wr / (1 - away_adj_wr))
            diff = home_logit - away_logit
            base_prob = 1 / (1 + np.exp(-diff))
        else:
            base_prob = 0.5

        # Incorporate head-to-head if available
        h2h_key = tuple(sorted([home_player_id, away_player_id]))
        h2h = self.h2h_stats[h2h_key]
        h2h_total = h2h['p1_wins'] + h2h['p2_wins']

        if h2h_total > 0:
            if home_player_id == h2h_key[0]:
                h2h_win_rate = h2h['p1_wins'] / h2h_total
            else:
                h2h_win_rate = h2h['p2_wins'] / h2h_total

            # Weight h2h more as we have more data (max 60% weight)
            h2h_weight = min(h2h_total / 15, 0.6)
            return h2h_weight * h2h_win_rate + (1 - h2h_weight) * base_prob

        return base_prob

    def _safe_rate(self, numerator: int, denominator: int, default: float = 0.0) -> float:
        """Compute ratio safely, returning default if denominator is 0."""
        if denominator == 0:
            return default
        return numerator / denominator

    def _parse_list(self, val) -> List[str]:
        """Parse a value that might be a list or string representation of a list."""
        if isinstance(val, list):
            return val
        if isinstance(val, np.ndarray):
            return list(val)
        if isinstance(val, str):
            if val.startswith('['):
                val = val.strip('[]')
            return [s.strip().strip("'\"") for s in val.split(',') if s.strip()]
        return []


class DatasetBuilderV2:
    """Creates train/val/test splits and prepares final datasets."""

    def __init__(self, features_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        self.features_df = features_df
        self.cleaned_df = cleaned_df

    def create_temporal_splits(
        self,
        train_end_date: str = "2024-06-01",
        val_end_date: str = "2025-01-01"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/val/test splits."""
        df = self.features_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)

        train_df = df[df['date'] < train_end].copy()
        val_df = df[(df['date'] >= train_end) & (df['date'] < val_end)].copy()
        test_df = df[df['date'] >= val_end].copy()

        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Get lists of feature columns by type."""
        all_cols = self.features_df.columns.tolist()

        # Pre-game features (can be used for prediction)
        pregame_features = [
            'home_team_token', 'away_team_token', 'day_of_week', 'month',
            'home_jb_player_token', 'away_jb_player_token',
            'home_jb_player_win_rate', 'away_jb_player_win_rate',
            'home_jb_player_total_jbs', 'away_jb_player_total_jbs',
            'jb_win_rate_diff',
            'home_jb_h2h_wins', 'home_jb_h2h_losses', 'home_jb_h2h_win_rate', 'h2h_total_matchups',
            'home_jb_predicted_win_prob',
            'home_team_tip_win_rate', 'away_team_tip_win_rate',
            'home_team_fs_when_tip_win_rate', 'away_team_fs_when_tip_win_rate',
            'home_team_total_fs_rate', 'away_team_total_fs_rate',
        ]
        # Add starter features
        for i in range(5):
            pregame_features.extend([
                f'home_starter_{i}_token', f'home_starter_{i}_fs_rate', f'home_starter_{i}_games',
                f'away_starter_{i}_token', f'away_starter_{i}_fs_rate', f'away_starter_{i}_games',
            ])

        # Target variables (outcomes)
        target_features = [
            'home_won_tip', 'first_scorer_is_home', 'tip_winner_scored_first',
            'first_scorer_token', 'first_score_type', 'first_scorer_position'
        ]

        return {
            'pregame_features': [c for c in pregame_features if c in all_cols],
            'target_features': [c for c in target_features if c in all_cols],
            'id_features': ['game_id', 'date']
        }


def main():
    """Main function to run feature engineering V2."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Feature Engineering V2 (No Leakage)")
    logger.info("=" * 60)

    # Load cleaned data
    cleaned_df = pd.read_parquet("data/processed/cleaned_games.parquet")
    logger.info(f"Loaded {len(cleaned_df)} cleaned games")

    # Re-parse starter lists
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
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def extract_ids(starters):
        parsed = parse_starters(starters)
        return [p.get('id', '') for p in parsed if isinstance(p, dict)]

    cleaned_df['home_starter_id_list'] = cleaned_df['home_starters'].apply(extract_ids)
    cleaned_df['away_starter_id_list'] = cleaned_df['away_starters'].apply(extract_ids)

    # Load tokenizer
    tokenizer = PlayerTokenizer.load()

    # Build features
    builder = FeatureBuilderV2(tokenizer)
    features_df = builder.compute_historical_features(cleaned_df)

    # Save features
    output_path = Path("data/processed")
    features_df.to_parquet(output_path / "features_v2.parquet", index=False)
    logger.info(f"Saved features to {output_path / 'features_v2.parquet'}")

    # Create dataset splits
    dataset_builder = DatasetBuilderV2(features_df, cleaned_df)

    # Temporal splits
    train_df, val_df, test_df = dataset_builder.create_temporal_splits(
        train_end_date="2024-06-01",
        val_end_date="2025-06-01"
    )

    train_df.to_parquet(output_path / "train_v2.parquet", index=False)
    val_df.to_parquet(output_path / "val_v2.parquet", index=False)
    test_df.to_parquet(output_path / "test_v2.parquet", index=False)

    logger.info(f"Saved train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) splits")

    # Print feature summary
    feature_cols = dataset_builder.get_feature_columns()
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SUMMARY (V2 - No Leakage)")
    logger.info("=" * 60)
    for feature_type, cols in feature_cols.items():
        logger.info(f"\n{feature_type} ({len(cols)} columns)")

    # Print target statistics
    logger.info("\n" + "=" * 60)
    logger.info("TARGET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Home wins tip: {train_df['home_won_tip'].mean():.3f}")
    logger.info(f"Tip winner scores first: {train_df['tip_winner_scored_first'].mean():.3f}")
    logger.info(f"Home team scores first: {train_df['first_scorer_is_home'].mean():.3f}")

    return features_df, train_df, val_df, test_df


if __name__ == "__main__":
    main()
