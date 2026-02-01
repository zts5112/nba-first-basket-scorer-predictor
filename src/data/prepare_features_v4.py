"""
Feature Engineering V4 - With NBA API Player Stats

Adds player-level statistics from NBA API:
- PPG (Points Per Game)
- FG%, 3P%, FT%
- 3-point attempt rate
- Estimated usage rate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from src.data.tokenize_players import PlayerTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBuilderV4:
    """
    Enhanced feature builder with NBA API player stats.
    """

    def __init__(self, tokenizer: Optional[PlayerTokenizer] = None, lookback_games: int = 10):
        self.tokenizer = tokenizer or PlayerTokenizer.load()
        self.lookback_games = lookback_games

        # Load NBA API player stats
        self.player_api_stats = self._load_player_api_stats()

        # Historical statistics (computed incrementally)
        self.player_jb_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total': 0
        })
        self.player_fs_stats: Dict[str, Dict] = defaultdict(lambda: {
            'first_scores': 0, 'games_started': 0, 'recent_fs': deque(maxlen=lookback_games)
        })
        self.team_stats: Dict[str, Dict] = defaultdict(lambda: {
            'tip_wins': 0, 'tip_losses': 0,
            'scored_first_with_tip': 0, 'total_games': 0
        })
        self.h2h_stats: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
            'p1_wins': 0, 'p2_wins': 0
        })

    def _load_player_api_stats(self) -> Dict[str, Dict]:
        """Load player stats from NBA API cache."""
        stats_file = Path("data/processed/player_stats_lookup.json")
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            logger.info(f"Loaded API stats for {len(stats)} player-seasons")
            return stats
        else:
            logger.warning("No player API stats found. Run fetch_player_stats.py first.")
            return {}

    def _get_player_api_stats(self, bref_id: str, game_date: pd.Timestamp) -> Dict:
        """Get API stats for a player based on game date."""
        # Determine which season to use
        year = game_date.year
        month = game_date.month

        # NBA season runs Oct-June, so:
        # Oct 2024 - June 2025 = "2024-25" season
        if month >= 10:
            season = f"{year}-{(year+1) % 100:02d}"
        else:
            season = f"{year-1}-{year % 100:02d}"

        key = f"{bref_id}_{season}"
        if key in self.player_api_stats:
            return self.player_api_stats[key]

        # Try previous season
        if month >= 10:
            prev_season = f"{year-1}-{year % 100:02d}"
        else:
            prev_season = f"{year-2}-{(year-1) % 100:02d}"

        prev_key = f"{bref_id}_{prev_season}"
        if prev_key in self.player_api_stats:
            return self.player_api_stats[prev_key]

        # Return defaults
        return {
            'ppg': 0.0,
            'fg_pct': 0.0,
            'fg3_pct': 0.0,
            'ft_pct': 0.0,
            'fg3_rate': 0.0,
            'est_usage': 0.0,
            'games': 0,
            'min': 0.0
        }

    def compute_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for each game using only prior data."""
        logger.info(f"Computing V4 features for {len(df)} games...")

        df = df.sort_values('date').reset_index(drop=True)
        features_list = []

        for idx, row in df.iterrows():
            if idx % 500 == 0:
                logger.info(f"  Processing game {idx + 1}/{len(df)}...")

            features = self._compute_game_features(row)
            features['game_id'] = row['game_id']
            features['date'] = row['date']
            features_list.append(features)

            self._update_stats(row)

        features_df = pd.DataFrame(features_list)
        logger.info(f"Computed {len(features_df.columns)} features")

        return features_df

    def _compute_game_features(self, row: pd.Series) -> Dict:
        """Compute all features for a single game."""
        features = {}
        game_date = pd.to_datetime(row['date'])

        home_team = row['home_team']
        away_team = row['away_team']

        # === BASIC GAME INFO ===
        features['home_team_token'] = self.tokenizer.encode_team(home_team)
        features['away_team_token'] = self.tokenizer.encode_team(away_team)
        features['day_of_week'] = row['day_of_week']
        features['month'] = row['month']

        # === JUMP BALL FEATURES ===
        jb_winner_id = row['jump_ball_winner_id']
        jb_loser_id = row['jump_ball_loser_id']
        jb_winner_team = row['jump_ball_winner_team']

        if jb_winner_team == home_team:
            home_jb_id, away_jb_id = jb_winner_id, jb_loser_id
            home_won_jb = True
        else:
            home_jb_id, away_jb_id = jb_loser_id, jb_winner_id
            home_won_jb = False

        features['home_jb_player_token'] = self.tokenizer.encode_player(home_jb_id)
        features['away_jb_player_token'] = self.tokenizer.encode_player(away_jb_id)

        # Historical jump ball stats
        home_jb_stats = self.player_jb_stats[home_jb_id]
        away_jb_stats = self.player_jb_stats[away_jb_id]

        features['home_jb_win_rate'] = self._safe_rate(home_jb_stats['wins'], home_jb_stats['total'], 0.5)
        features['away_jb_win_rate'] = self._safe_rate(away_jb_stats['wins'], away_jb_stats['total'], 0.5)
        features['jb_win_rate_diff'] = features['home_jb_win_rate'] - features['away_jb_win_rate']
        features['home_jb_total'] = home_jb_stats['total']
        features['away_jb_total'] = away_jb_stats['total']

        # H2H
        h2h_key = tuple(sorted([home_jb_id, away_jb_id]))
        h2h = self.h2h_stats[h2h_key]
        h2h_total = h2h['p1_wins'] + h2h['p2_wins']
        if h2h_total > 0:
            home_h2h_wins = h2h['p1_wins'] if home_jb_id == h2h_key[0] else h2h['p2_wins']
            features['home_jb_h2h_rate'] = home_h2h_wins / h2h_total
        else:
            features['home_jb_h2h_rate'] = 0.5
        features['h2h_matchups'] = h2h_total

        features['home_jb_predicted_prob'] = self._predict_jb_win_prob(home_jb_id, away_jb_id)

        # === STARTER FEATURES ===
        home_ids = self._parse_list(row.get('home_starter_id_list', []))
        away_ids = self._parse_list(row.get('away_starter_id_list', []))
        home_ids = (home_ids + [''] * 5)[:5]
        away_ids = (away_ids + [''] * 5)[:5]

        features['home_jb_is_starter'] = int(home_jb_id in home_ids)
        features['away_jb_is_starter'] = int(away_jb_id in away_ids)

        # Per-starter features including NBA API stats
        for i, pid in enumerate(home_ids):
            prefix = f'home_{i}'
            features[f'{prefix}_token'] = self.tokenizer.encode_player(pid)

            # Our computed stats
            fs_stats = self.player_fs_stats[pid]
            features[f'{prefix}_fs_rate'] = self._safe_rate(fs_stats['first_scores'], fs_stats['games_started'])
            features[f'{prefix}_games'] = fs_stats['games_started']

            recent = list(fs_stats['recent_fs'])
            features[f'{prefix}_recent_fs_rate'] = np.mean(recent) if recent else 0.0

            features[f'{prefix}_is_jb'] = int(pid == home_jb_id)

            # NBA API stats
            api_stats = self._get_player_api_stats(pid, game_date)
            features[f'{prefix}_ppg'] = api_stats['ppg']
            features[f'{prefix}_fg_pct'] = api_stats['fg_pct']
            features[f'{prefix}_fg3_pct'] = api_stats['fg3_pct']
            features[f'{prefix}_fg3_rate'] = api_stats['fg3_rate']
            features[f'{prefix}_usage'] = api_stats['est_usage']

        for i, pid in enumerate(away_ids):
            prefix = f'away_{i}'
            features[f'{prefix}_token'] = self.tokenizer.encode_player(pid)

            fs_stats = self.player_fs_stats[pid]
            features[f'{prefix}_fs_rate'] = self._safe_rate(fs_stats['first_scores'], fs_stats['games_started'])
            features[f'{prefix}_games'] = fs_stats['games_started']

            recent = list(fs_stats['recent_fs'])
            features[f'{prefix}_recent_fs_rate'] = np.mean(recent) if recent else 0.0

            features[f'{prefix}_is_jb'] = int(pid == away_jb_id)

            # NBA API stats
            api_stats = self._get_player_api_stats(pid, game_date)
            features[f'{prefix}_ppg'] = api_stats['ppg']
            features[f'{prefix}_fg_pct'] = api_stats['fg_pct']
            features[f'{prefix}_fg3_pct'] = api_stats['fg3_pct']
            features[f'{prefix}_fg3_rate'] = api_stats['fg3_rate']
            features[f'{prefix}_usage'] = api_stats['est_usage']

        # Team aggregates
        features['home_total_fs_rate'] = sum(features[f'home_{i}_fs_rate'] for i in range(5))
        features['away_total_fs_rate'] = sum(features[f'away_{i}_fs_rate'] for i in range(5))
        features['home_total_ppg'] = sum(features[f'home_{i}_ppg'] for i in range(5))
        features['away_total_ppg'] = sum(features[f'away_{i}_ppg'] for i in range(5))
        features['home_avg_fg_pct'] = np.mean([features[f'home_{i}_fg_pct'] for i in range(5)])
        features['away_avg_fg_pct'] = np.mean([features[f'away_{i}_fg_pct'] for i in range(5)])
        features['home_total_usage'] = sum(features[f'home_{i}_usage'] for i in range(5))
        features['away_total_usage'] = sum(features[f'away_{i}_usage'] for i in range(5))

        # Best scorer on each team (by PPG)
        home_ppgs = [features[f'home_{i}_ppg'] for i in range(5)]
        away_ppgs = [features[f'away_{i}_ppg'] for i in range(5)]
        features['home_max_ppg'] = max(home_ppgs)
        features['away_max_ppg'] = max(away_ppgs)
        features['home_best_scorer_pos'] = home_ppgs.index(max(home_ppgs))
        features['away_best_scorer_pos'] = away_ppgs.index(max(away_ppgs))

        # JB player stats
        home_jb_api = self._get_player_api_stats(home_jb_id, game_date)
        away_jb_api = self._get_player_api_stats(away_jb_id, game_date)
        features['home_jb_ppg'] = home_jb_api['ppg']
        features['away_jb_ppg'] = away_jb_api['ppg']
        features['home_jb_fg_pct'] = home_jb_api['fg_pct']
        features['away_jb_fg_pct'] = away_jb_api['fg_pct']

        # Team historical stats
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        features['home_tip_win_rate'] = self._safe_rate(home_stats['tip_wins'], home_stats['total_games'], 0.5)
        features['away_tip_win_rate'] = self._safe_rate(away_stats['tip_wins'], away_stats['total_games'], 0.5)

        # === TARGET VARIABLES ===
        features['home_won_tip'] = int(home_won_jb)
        features['tip_winner_scored_first'] = int(row['tip_winner_scored_first'])
        features['first_scorer_is_home'] = int(row['first_scorer_team'] == home_team)
        features['first_scorer_token'] = self.tokenizer.encode_player(row['first_scorer_id'])
        features['first_score_type'] = row['first_score_type']

        # First scorer position
        first_scorer_id = row['first_scorer_id']
        fs_position = -1
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
        """Update historical stats after processing a game."""
        jb_winner_id = row['jump_ball_winner_id']
        jb_loser_id = row['jump_ball_loser_id']
        first_scorer_id = row['first_scorer_id']
        home_team = row['home_team']
        away_team = row['away_team']
        jb_winner_team = row['jump_ball_winner_team']

        # Update JB stats
        self.player_jb_stats[jb_winner_id]['wins'] += 1
        self.player_jb_stats[jb_winner_id]['total'] += 1
        self.player_jb_stats[jb_loser_id]['losses'] += 1
        self.player_jb_stats[jb_loser_id]['total'] += 1

        # Update H2H
        h2h_key = tuple(sorted([jb_winner_id, jb_loser_id]))
        if jb_winner_id == h2h_key[0]:
            self.h2h_stats[h2h_key]['p1_wins'] += 1
        else:
            self.h2h_stats[h2h_key]['p2_wins'] += 1

        # Update first scorer stats
        self.player_fs_stats[first_scorer_id]['first_scores'] += 1

        # Update games started for all starters
        home_ids = self._parse_list(row.get('home_starter_id_list', []))
        away_ids = self._parse_list(row.get('away_starter_id_list', []))

        for pid in home_ids + away_ids:
            if pid:
                self.player_fs_stats[pid]['games_started'] += 1
                scored = 1 if pid == first_scorer_id else 0
                self.player_fs_stats[pid]['recent_fs'].append(scored)

        # Update team stats
        self.team_stats[home_team]['total_games'] += 1
        self.team_stats[away_team]['total_games'] += 1

        if jb_winner_team == home_team:
            self.team_stats[home_team]['tip_wins'] += 1
        else:
            self.team_stats[away_team]['tip_wins'] += 1

    def _predict_jb_win_prob(self, home_id: str, away_id: str) -> float:
        """Predict jump ball win probability for home player."""
        home_stats = self.player_jb_stats[home_id]
        away_stats = self.player_jb_stats[away_id]

        prior, prior_strength = 0.5, 5

        home_wr = (home_stats['wins'] + prior_strength * prior) / (home_stats['total'] + prior_strength)
        away_wr = (away_stats['wins'] + prior_strength * prior) / (away_stats['total'] + prior_strength)

        if 0 < home_wr < 1 and 0 < away_wr < 1:
            home_logit = np.log(home_wr / (1 - home_wr))
            away_logit = np.log(away_wr / (1 - away_wr))
            base_prob = 1 / (1 + np.exp(-(home_logit - away_logit)))
        else:
            base_prob = 0.5

        h2h_key = tuple(sorted([home_id, away_id]))
        h2h = self.h2h_stats[h2h_key]
        h2h_total = h2h['p1_wins'] + h2h['p2_wins']

        if h2h_total > 0:
            h2h_wins = h2h['p1_wins'] if home_id == h2h_key[0] else h2h['p2_wins']
            h2h_rate = h2h_wins / h2h_total
            h2h_weight = min(h2h_total / 15, 0.6)
            return h2h_weight * h2h_rate + (1 - h2h_weight) * base_prob

        return base_prob

    def _safe_rate(self, num: int, denom: int, default: float = 0.0) -> float:
        return num / denom if denom > 0 else default

    def _parse_list(self, val) -> List[str]:
        if isinstance(val, list):
            return val
        if isinstance(val, np.ndarray):
            return list(val)
        if isinstance(val, str):
            if val.startswith('['):
                val = val.strip('[]')
            return [s.strip().strip("'\"") for s in val.split(',') if s.strip()]
        return []


def main():
    """Generate V4 features with NBA API stats."""
    logger.info("=" * 60)
    logger.info("Feature Engineering V4 - With NBA API Stats")
    logger.info("=" * 60)

    # Load cleaned data
    cleaned_df = pd.read_parquet("data/processed/cleaned_games.parquet")
    logger.info(f"Loaded {len(cleaned_df)} games")

    # Parse starter lists
    import json

    def parse_starters(starters):
        if starters is None:
            return []
        if isinstance(starters, np.ndarray):
            return list(starters)
        if isinstance(starters, list):
            return starters
        try:
            return json.loads(starters)
        except:
            return []

    def extract_ids(starters):
        parsed = parse_starters(starters)
        return [p.get('id', '') for p in parsed if isinstance(p, dict)]

    cleaned_df['home_starter_id_list'] = cleaned_df['home_starters'].apply(extract_ids)
    cleaned_df['away_starter_id_list'] = cleaned_df['away_starters'].apply(extract_ids)

    # Build features
    tokenizer = PlayerTokenizer.load()
    builder = FeatureBuilderV4(tokenizer, lookback_games=10)
    features_df = builder.compute_historical_features(cleaned_df)

    # Save
    output_path = Path("data/processed")
    features_df.to_parquet(output_path / "features_v4.parquet", index=False)
    logger.info(f"Saved to {output_path / 'features_v4.parquet'}")

    # Create splits
    features_df['date'] = pd.to_datetime(features_df['date'])
    train_df = features_df[features_df['date'] < '2024-06-01']
    val_df = features_df[(features_df['date'] >= '2024-06-01') & (features_df['date'] < '2025-06-01')]
    test_df = features_df[features_df['date'] >= '2025-06-01']

    train_df.to_parquet(output_path / "train_v4.parquet", index=False)
    val_df.to_parquet(output_path / "val_v4.parquet", index=False)
    test_df.to_parquet(output_path / "test_v4.parquet", index=False)

    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Print new API-based features
    api_features = [c for c in features_df.columns if any(x in c for x in ['ppg', 'fg_pct', 'fg3', 'usage'])]
    logger.info(f"\nNew API-based features ({len(api_features)}):")
    for f in api_features[:20]:
        logger.info(f"  {f}")

    return features_df


if __name__ == "__main__":
    main()
