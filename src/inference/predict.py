"""
Inference Pipeline for NBA First Basket Scorer Prediction

This module provides prediction functionality for upcoming games.
It loads trained models and generates predictions given:
- Home and away team
- Starting lineups
- Jump ball participants (centers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Import model classes so pickle can find them
from src.data.train_models_v4 import JumpBallModelV4, PlayerFirstScorerModelV4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Represents a player with their ID and name."""
    id: str
    name: str

    @classmethod
    def from_dict(cls, d: Dict) -> 'Player':
        return cls(id=d.get('id', ''), name=d.get('name', ''))


@dataclass
class GamePrediction:
    """Prediction result for a game."""
    home_team: str
    away_team: str
    home_jb_player: Player
    away_jb_player: Player
    home_starters: List[Player]
    away_starters: List[Player]

    # Predictions
    home_wins_tip_prob: float
    player_probabilities: List[Dict]  # List of {player, team, position, probability}

    def get_top_scorers(self, n: int = 3) -> List[Dict]:
        """Get top N most likely first scorers."""
        sorted_probs = sorted(self.player_probabilities, key=lambda x: -x['probability'])
        return sorted_probs[:n]

    def print_prediction(self):
        """Print formatted prediction."""
        print("\n" + "=" * 60)
        print(f"FIRST BASKET SCORER PREDICTION")
        print(f"{self.away_team} @ {self.home_team}")
        print("=" * 60)

        print(f"\n📊 JUMP BALL PREDICTION")
        print(f"   {self.home_jb_player.name} ({self.home_team}) vs {self.away_jb_player.name} ({self.away_team})")
        print(f"   Home wins tip: {self.home_wins_tip_prob:.1%}")

        print(f"\n🏀 FIRST SCORER PREDICTIONS (Top 5)")
        for i, pred in enumerate(self.get_top_scorers(5), 1):
            team_emoji = "🏠" if pred['team'] == 'home' else "✈️"
            team_name = self.home_team if pred['team'] == 'home' else self.away_team
            print(f"   {i}. {pred['player_name']:<25} ({team_name}) - {pred['probability']:.1%}")

        print(f"\n📋 ALL STARTER PROBABILITIES")
        print(f"\n   {self.home_team} (Home):")
        for pred in self.player_probabilities[:5]:
            print(f"      {pred['player_name']:<25}: {pred['probability']:.1%}")

        print(f"\n   {self.away_team} (Away):")
        for pred in self.player_probabilities[5:]:
            print(f"      {pred['player_name']:<25}: {pred['probability']:.1%}")


class FirstScorerPredictor:
    """
    Main predictor class for first basket scorer predictions.

    Usage:
        predictor = FirstScorerPredictor()
        prediction = predictor.predict(
            home_team="LAL",
            away_team="BOS",
            home_starters=["Anthony Davis", "LeBron James", ...],
            away_starters=["Jayson Tatum", "Jaylen Brown", ...],
            home_center="Anthony Davis",
            away_center="Al Horford"
        )
    """

    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "data/processed"
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        # Load models
        self._load_models()

        # Load tokenizer mappings
        self._load_tokenizers()

        # Load historical stats for feature computation
        self._load_historical_stats()

    def _load_models(self):
        """Load trained models."""
        self.jb_model = joblib.load(self.model_dir / "jump_ball_model_v4.joblib")
        self.player_model = joblib.load(self.model_dir / "player_first_scorer_model_v4.joblib")
        logger.info("V4 models loaded successfully")

    def _load_tokenizers(self):
        """Load player and team tokenizers."""
        with open(self.data_dir / "player_tokenizer.json") as f:
            player_data = json.load(f)

        self.player_to_token = player_data['player_to_token']
        self.player_name_to_id = player_data['player_name_to_id']
        self.player_id_to_name = player_data['player_id_to_name']
        self.player_stats = player_data.get('player_stats', {})

        with open(self.data_dir / "team_tokenizer.json") as f:
            team_data = json.load(f)

        self.team_to_token = team_data['team_to_token']
        logger.info(f"Loaded {len(self.player_to_token)} player tokens, {len(self.team_to_token)} team tokens")

        # Load NBA API player stats for V4 features
        self._load_player_api_stats()

    def _load_player_api_stats(self):
        """Load player stats from NBA API cache."""
        stats_file = self.data_dir / "player_stats_lookup.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.player_api_stats = json.load(f)
            logger.info(f"Loaded API stats for {len(self.player_api_stats)} player-seasons")
        else:
            logger.warning("No player API stats found. Some features will use defaults.")
            self.player_api_stats = {}

    def _get_player_api_stats(self, bref_id: str, season: str = "2024-25") -> Dict:
        """Get API stats for a player."""
        key = f"{bref_id}_{season}"
        if key in self.player_api_stats:
            return self.player_api_stats[key]

        # Try previous season
        parts = season.split('-')
        prev_season = f"{int(parts[0])-1}-{int(parts[1])-1:02d}"
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

    def _load_historical_stats(self):
        """Load historical statistics from training data."""
        # Load the V4 training features to get final historical stats
        train_df = pd.read_parquet(self.data_dir / "train_v4.parquet")
        val_df = pd.read_parquet(self.data_dir / "val_v4.parquet")
        test_df = pd.read_parquet(self.data_dir / "test_v4.parquet")

        # Combine all data to get most recent stats
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_df = all_df.sort_values('date')

        # Extract player statistics from the last row of features
        # These represent cumulative stats up to the most recent game
        self.player_jb_stats = {}
        self.player_fs_stats = {}
        self.team_stats = {}

        # We'll compute stats by aggregating target variables
        # For now, use simple defaults and update as we see more data
        self._compute_stats_from_data(all_df)

        logger.info("Historical statistics loaded")

    def _compute_stats_from_data(self, df: pd.DataFrame):
        """Compute player/team statistics from historical data."""
        # Jump ball stats
        self.player_jb_wins = defaultdict(int)
        self.player_jb_total = defaultdict(int)

        # First scorer stats
        self.player_fs_count = defaultdict(int)
        self.player_games = defaultdict(int)

        # Team stats
        self.team_tip_wins = defaultdict(int)
        self.team_games = defaultdict(int)

        # Process each game
        for _, row in df.iterrows():
            # JB stats - we only have home_won_tip, so we infer from player tokens
            home_won = row['home_won_tip']
            home_jb_token = row['home_jb_player_token']
            away_jb_token = row['away_jb_player_token']

            # Find player IDs from tokens
            for pid, token in self.player_to_token.items():
                if token == home_jb_token:
                    self.player_jb_total[pid] += 1
                    if home_won:
                        self.player_jb_wins[pid] += 1
                if token == away_jb_token:
                    self.player_jb_total[pid] += 1
                    if not home_won:
                        self.player_jb_wins[pid] += 1

            # First scorer stats
            fs_token = row['first_scorer_token']
            for pid, token in self.player_to_token.items():
                if token == fs_token:
                    self.player_fs_count[pid] += 1
                    break

            # Count games for starters
            for i in range(5):
                for prefix in ['home', 'away']:
                    token = row[f'{prefix}_{i}_token']
                    for pid, t in self.player_to_token.items():
                        if t == token:
                            self.player_games[pid] += 1
                            break

            # Team stats
            home_team_token = row['home_team_token']
            away_team_token = row['away_team_token']
            for team, token in self.team_to_token.items():
                if token == home_team_token:
                    self.team_games[team] += 1
                    if home_won:
                        self.team_tip_wins[team] += 1
                if token == away_team_token:
                    self.team_games[team] += 1
                    if not home_won:
                        self.team_tip_wins[team] += 1

    def _get_player_id(self, name_or_id: str) -> str:
        """Get player ID from name or ID."""
        if name_or_id in self.player_to_token:
            return name_or_id
        if name_or_id in self.player_name_to_id:
            return self.player_name_to_id[name_or_id]
        # Try fuzzy match
        name_lower = name_or_id.lower()
        for name, pid in self.player_name_to_id.items():
            if name.lower() == name_lower or name_lower in name.lower():
                return pid
        return name_or_id  # Return as-is if not found

    def _encode_player(self, player_id: str) -> int:
        """Encode player ID to token."""
        return self.player_to_token.get(player_id, 1)  # 1 = UNK

    def _encode_team(self, team: str) -> int:
        """Encode team to token."""
        return self.team_to_token.get(team.upper(), 1)

    def _get_player_jb_stats(self, player_id: str) -> Dict:
        """Get jump ball statistics for a player."""
        wins = self.player_jb_wins.get(player_id, 0)
        total = self.player_jb_total.get(player_id, 0)
        return {
            'wins': wins,
            'total': total,
            'win_rate': wins / total if total > 0 else 0.5
        }

    def _get_player_fs_stats(self, player_id: str) -> Dict:
        """Get first scorer statistics for a player."""
        fs = self.player_fs_count.get(player_id, 0)
        games = self.player_games.get(player_id, 0)
        return {
            'first_scores': fs,
            'games': games,
            'fs_rate': fs / games if games > 0 else 0.0
        }

    def _get_team_stats(self, team: str) -> Dict:
        """Get team statistics."""
        wins = self.team_tip_wins.get(team, 0)
        games = self.team_games.get(team, 0)
        return {
            'tip_wins': wins,
            'games': games,
            'tip_win_rate': wins / games if games > 0 else 0.5
        }

    def predict(
        self,
        home_team: str,
        away_team: str,
        home_starters: List[str],
        away_starters: List[str],
        home_center: str,
        away_center: str,
        day_of_week: int = 0,
        month: int = 1
    ) -> GamePrediction:
        """
        Generate prediction for a game.

        Args:
            home_team: Home team abbreviation (e.g., "LAL")
            away_team: Away team abbreviation (e.g., "BOS")
            home_starters: List of 5 home starter names
            away_starters: List of 5 away starter names
            home_center: Name of home team's jump ball player
            away_center: Name of away team's jump ball player
            day_of_week: Day of week (0=Mon, 6=Sun)
            month: Month (1-12)

        Returns:
            GamePrediction object with all predictions
        """
        # Convert names to IDs
        home_ids = [self._get_player_id(p) for p in home_starters]
        away_ids = [self._get_player_id(p) for p in away_starters]
        home_jb_id = self._get_player_id(home_center)
        away_jb_id = self._get_player_id(away_center)

        # Build feature dict
        features = self._build_features(
            home_team=home_team,
            away_team=away_team,
            home_ids=home_ids,
            away_ids=away_ids,
            home_jb_id=home_jb_id,
            away_jb_id=away_jb_id,
            day_of_week=day_of_week,
            month=month
        )

        # Create DataFrame for model input
        features_df = pd.DataFrame([features])

        # Get predictions
        home_wins_tip_prob = float(self.jb_model.predict_proba(features_df)[0])
        player_probs = self.player_model.predict_proba(features_df)[0]

        # Build player probability list
        player_probabilities = []
        for i, (player_id, prob) in enumerate(zip(home_ids + away_ids, player_probs)):
            team = 'home' if i < 5 else 'away'
            player_name = self.player_id_to_name.get(player_id, player_id)
            player_probabilities.append({
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'position': i,
                'probability': prob
            })

        # Create prediction object
        prediction = GamePrediction(
            home_team=home_team,
            away_team=away_team,
            home_jb_player=Player(id=home_jb_id, name=self.player_id_to_name.get(home_jb_id, home_center)),
            away_jb_player=Player(id=away_jb_id, name=self.player_id_to_name.get(away_jb_id, away_center)),
            home_starters=[Player(id=pid, name=self.player_id_to_name.get(pid, name))
                          for pid, name in zip(home_ids, home_starters)],
            away_starters=[Player(id=pid, name=self.player_id_to_name.get(pid, name))
                          for pid, name in zip(away_ids, away_starters)],
            home_wins_tip_prob=home_wins_tip_prob,
            player_probabilities=player_probabilities
        )

        return prediction

    def _build_features(
        self,
        home_team: str,
        away_team: str,
        home_ids: List[str],
        away_ids: List[str],
        home_jb_id: str,
        away_jb_id: str,
        day_of_week: int,
        month: int
    ) -> Dict:
        """Build feature dictionary for model input."""
        features = {}

        # Basic info
        features['home_team_token'] = self._encode_team(home_team)
        features['away_team_token'] = self._encode_team(away_team)
        features['day_of_week'] = day_of_week
        features['month'] = month

        # Jump ball features
        features['home_jb_player_token'] = self._encode_player(home_jb_id)
        features['away_jb_player_token'] = self._encode_player(away_jb_id)

        home_jb_stats = self._get_player_jb_stats(home_jb_id)
        away_jb_stats = self._get_player_jb_stats(away_jb_id)

        features['home_jb_win_rate'] = home_jb_stats['win_rate']
        features['away_jb_win_rate'] = away_jb_stats['win_rate']
        features['jb_win_rate_diff'] = home_jb_stats['win_rate'] - away_jb_stats['win_rate']
        features['home_jb_total'] = home_jb_stats['total']
        features['away_jb_total'] = away_jb_stats['total']
        features['home_jb_h2h_rate'] = 0.5  # No H2H data for new matchups
        features['h2h_matchups'] = 0

        # Predicted JB probability
        prior = 0.5
        prior_strength = 5
        home_adj = (home_jb_stats['wins'] + prior_strength * prior) / (home_jb_stats['total'] + prior_strength)
        away_adj = (away_jb_stats['wins'] + prior_strength * prior) / (away_jb_stats['total'] + prior_strength)

        if 0 < home_adj < 1 and 0 < away_adj < 1:
            home_logit = np.log(home_adj / (1 - home_adj))
            away_logit = np.log(away_adj / (1 - away_adj))
            features['home_jb_predicted_prob'] = 1 / (1 + np.exp(-(home_logit - away_logit)))
        else:
            features['home_jb_predicted_prob'] = 0.5

        # Team stats
        home_team_stats = self._get_team_stats(home_team)
        away_team_stats = self._get_team_stats(away_team)
        features['home_tip_win_rate'] = home_team_stats['tip_win_rate']
        features['away_tip_win_rate'] = away_team_stats['tip_win_rate']

        # Starter features
        home_jb_fs_stats = self._get_player_fs_stats(home_jb_id)
        away_jb_fs_stats = self._get_player_fs_stats(away_jb_id)
        features['home_jb_is_starter'] = int(home_jb_id in home_ids)
        features['away_jb_is_starter'] = int(away_jb_id in away_ids)
        features['home_jb_lineup_pos'] = home_ids.index(home_jb_id) if home_jb_id in home_ids else -1
        features['away_jb_lineup_pos'] = away_ids.index(away_jb_id) if away_jb_id in away_ids else -1
        features['home_jb_fs_rate'] = home_jb_fs_stats['fs_rate']
        features['away_jb_fs_rate'] = away_jb_fs_stats['fs_rate']

        # Per-starter features (including NBA API stats for V4)
        for i, pid in enumerate(home_ids):
            prefix = f'home_{i}'
            features[f'{prefix}_token'] = self._encode_player(pid)
            stats = self._get_player_fs_stats(pid)
            features[f'{prefix}_fs_rate'] = stats['fs_rate']
            features[f'{prefix}_games'] = stats['games']
            features[f'{prefix}_recent_fs_rate'] = stats['fs_rate']  # Approximate with overall
            features[f'{prefix}_is_jb'] = int(pid == home_jb_id)

            # V4: NBA API stats
            api_stats = self._get_player_api_stats(pid)
            features[f'{prefix}_ppg'] = api_stats['ppg']
            features[f'{prefix}_fg_pct'] = api_stats['fg_pct']
            features[f'{prefix}_fg3_pct'] = api_stats['fg3_pct']
            features[f'{prefix}_usage'] = api_stats['est_usage']

        for i, pid in enumerate(away_ids):
            prefix = f'away_{i}'
            features[f'{prefix}_token'] = self._encode_player(pid)
            stats = self._get_player_fs_stats(pid)
            features[f'{prefix}_fs_rate'] = stats['fs_rate']
            features[f'{prefix}_games'] = stats['games']
            features[f'{prefix}_recent_fs_rate'] = stats['fs_rate']
            features[f'{prefix}_is_jb'] = int(pid == away_jb_id)

            # V4: NBA API stats
            api_stats = self._get_player_api_stats(pid)
            features[f'{prefix}_ppg'] = api_stats['ppg']
            features[f'{prefix}_fg_pct'] = api_stats['fg_pct']
            features[f'{prefix}_fg3_pct'] = api_stats['fg3_pct']
            features[f'{prefix}_usage'] = api_stats['est_usage']

        # Aggregates
        features['home_total_fs_rate'] = sum(features[f'home_{i}_fs_rate'] for i in range(5))
        features['away_total_fs_rate'] = sum(features[f'away_{i}_fs_rate'] for i in range(5))

        # V4: PPG and FG% aggregates
        features['home_total_ppg'] = sum(features[f'home_{i}_ppg'] for i in range(5))
        features['away_total_ppg'] = sum(features[f'away_{i}_ppg'] for i in range(5))
        features['home_avg_fg_pct'] = np.mean([features[f'home_{i}_fg_pct'] for i in range(5)])
        features['away_avg_fg_pct'] = np.mean([features[f'away_{i}_fg_pct'] for i in range(5)])

        # V4: Best scorer by PPG
        home_ppgs = [features[f'home_{i}_ppg'] for i in range(5)]
        away_ppgs = [features[f'away_{i}_ppg'] for i in range(5)]
        features['home_max_ppg'] = max(home_ppgs) if home_ppgs else 0.0
        features['away_max_ppg'] = max(away_ppgs) if away_ppgs else 0.0

        # V4: JB player API stats
        home_jb_api = self._get_player_api_stats(home_jb_id)
        away_jb_api = self._get_player_api_stats(away_jb_id)
        features['home_jb_ppg'] = home_jb_api['ppg']
        features['away_jb_ppg'] = away_jb_api['ppg']
        features['home_jb_fg_pct'] = home_jb_api['fg_pct']
        features['away_jb_fg_pct'] = away_jb_api['fg_pct']

        return features


def main():
    """Demo the predictor with a sample game."""
    logger.info("=" * 60)
    logger.info("First Basket Scorer Predictor - Demo")
    logger.info("=" * 60)

    predictor = FirstScorerPredictor()

    # Example: Lakers vs Celtics
    prediction = predictor.predict(
        home_team="LAL",
        away_team="BOS",
        home_starters=[
            "Anthony Davis",
            "LeBron James",
            "Austin Reaves",
            "Rui Hachimura",
            "D'Angelo Russell"
        ],
        away_starters=[
            "Jayson Tatum",
            "Jaylen Brown",
            "Derrick White",
            "Jrue Holiday",
            "Al Horford"
        ],
        home_center="Anthony Davis",
        away_center="Al Horford"
    )

    prediction.print_prediction()

    return prediction


if __name__ == "__main__":
    main()
