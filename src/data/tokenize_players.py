"""
Player Tokenization Script for NBA First Basket Scorer Prediction

This script creates consistent integer encodings for players and teams,
generating lookup tables that can be used during training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlayerTokenizer:
    """
    Creates and manages player/team tokenization for ML models.

    Tokens are assigned based on frequency (more common players get lower tokens).
    Special tokens:
    - 0: [PAD] padding token
    - 1: [UNK] unknown player token
    - 2+: player tokens ordered by frequency
    """

    SPECIAL_TOKENS = {
        'PAD': 0,
        'UNK': 1,
    }

    def __init__(self):
        self.player_to_token: Dict[str, int] = {}
        self.token_to_player: Dict[int, str] = {}
        self.player_name_to_id: Dict[str, str] = {}  # Name -> Basketball Reference ID
        self.player_id_to_name: Dict[str, str] = {}  # Basketball Reference ID -> Name
        self.player_stats: Dict[str, Dict] = {}  # Player stats/metadata

        self.team_to_token: Dict[str, int] = {}
        self.token_to_team: Dict[int, str] = {}

        self._vocab_size = len(self.SPECIAL_TOKENS)
        self._team_vocab_size = len(self.SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._vocab_size

    @property
    def team_vocab_size(self) -> int:
        """Total team vocabulary size including special tokens."""
        return self._team_vocab_size

    def fit(self, df: pd.DataFrame, min_appearances: int = 1) -> 'PlayerTokenizer':
        """
        Fit the tokenizer on the dataset.

        Creates token mappings for:
        - Players (by Basketball Reference ID)
        - Teams

        Args:
            df: Cleaned games DataFrame
            min_appearances: Minimum appearances to get a token (others become UNK)
        """
        logger.info("Fitting player tokenizer...")

        # Collect all player IDs and count appearances
        player_counts = defaultdict(int)
        player_name_mapping = {}

        # Count from jump ball participants
        for col_id, col_name in [
            ('jump_ball_winner_id', 'jump_ball_winner_name'),
            ('jump_ball_loser_id', 'jump_ball_loser_name'),
        ]:
            for _, row in df.iterrows():
                pid = row[col_id]
                pname = row[col_name]
                if pd.notna(pid) and pid:
                    player_counts[pid] += 1
                    player_name_mapping[pid] = pname

        # Count from first scorers
        for _, row in df.iterrows():
            pid = row['first_scorer_id']
            pname = row['first_scorer_name']
            if pd.notna(pid) and pid:
                player_counts[pid] += 1
                player_name_mapping[pid] = pname

        # Count from starters
        for _, row in df.iterrows():
            home_ids = row.get('home_starter_id_list', [])
            away_ids = row.get('away_starter_id_list', [])
            home_names = row.get('home_starter_names', [])
            away_names = row.get('away_starter_names', [])

            if isinstance(home_ids, list):
                for pid, pname in zip(home_ids, home_names or []):
                    if pid:
                        player_counts[pid] += 1
                        if pname:
                            player_name_mapping[pid] = pname

            if isinstance(away_ids, list):
                for pid, pname in zip(away_ids, away_names or []):
                    if pid:
                        player_counts[pid] += 1
                        if pname:
                            player_name_mapping[pid] = pname

        # Sort by count (descending) to assign tokens
        sorted_players = sorted(player_counts.items(), key=lambda x: -x[1])

        # Assign tokens starting after special tokens
        token_id = len(self.SPECIAL_TOKENS)
        for player_id, count in sorted_players:
            if count >= min_appearances:
                self.player_to_token[player_id] = token_id
                self.token_to_player[token_id] = player_id
                self.player_id_to_name[player_id] = player_name_mapping.get(player_id, player_id)
                self.player_stats[player_id] = {
                    'appearances': count,
                    'name': player_name_mapping.get(player_id, player_id)
                }
                token_id += 1

        self._vocab_size = token_id

        # Create name to ID mapping (for lookup by name)
        for pid, pname in player_name_mapping.items():
            if pname:
                self.player_name_to_id[pname] = pid

        logger.info(f"Created {self._vocab_size - len(self.SPECIAL_TOKENS)} player tokens "
                   f"(min_appearances={min_appearances})")

        # Tokenize teams
        team_counts = defaultdict(int)
        for col in ['home_team', 'away_team']:
            for team in df[col]:
                if pd.notna(team):
                    team_counts[team] += 1

        sorted_teams = sorted(team_counts.items(), key=lambda x: -x[1])
        token_id = len(self.SPECIAL_TOKENS)
        for team, count in sorted_teams:
            self.team_to_token[team] = token_id
            self.token_to_team[token_id] = team
            token_id += 1

        self._team_vocab_size = token_id
        logger.info(f"Created {self._team_vocab_size - len(self.SPECIAL_TOKENS)} team tokens")

        return self

    def encode_player(self, player_id: str) -> int:
        """Encode a player ID to token. Returns UNK for unknown players."""
        if pd.isna(player_id) or not player_id:
            return self.SPECIAL_TOKENS['PAD']
        return self.player_to_token.get(player_id, self.SPECIAL_TOKENS['UNK'])

    def encode_player_by_name(self, player_name: str) -> int:
        """Encode a player by name (looks up ID first)."""
        if pd.isna(player_name) or not player_name:
            return self.SPECIAL_TOKENS['PAD']
        player_id = self.player_name_to_id.get(player_name)
        if player_id:
            return self.encode_player(player_id)
        return self.SPECIAL_TOKENS['UNK']

    def decode_player(self, token: int) -> str:
        """Decode a token to player ID."""
        if token in self.SPECIAL_TOKENS.values():
            for name, val in self.SPECIAL_TOKENS.items():
                if val == token:
                    return f"[{name}]"
        return self.token_to_player.get(token, "[UNK]")

    def encode_team(self, team: str) -> int:
        """Encode a team abbreviation to token."""
        if pd.isna(team) or not team:
            return self.SPECIAL_TOKENS['PAD']
        return self.team_to_token.get(team, self.SPECIAL_TOKENS['UNK'])

    def decode_team(self, token: int) -> str:
        """Decode a token to team abbreviation."""
        if token in self.SPECIAL_TOKENS.values():
            for name, val in self.SPECIAL_TOKENS.items():
                if val == token:
                    return f"[{name}]"
        return self.token_to_team.get(token, "[UNK]")

    def encode_lineup(self, player_ids: List[str], max_length: int = 5) -> List[int]:
        """Encode a list of player IDs, padding/truncating to max_length."""
        tokens = [self.encode_player(pid) for pid in player_ids[:max_length]]
        while len(tokens) < max_length:
            tokens.append(self.SPECIAL_TOKENS['PAD'])
        return tokens

    def save(self, output_dir: str = "data/processed") -> Dict[str, Path]:
        """Save tokenizer mappings to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files_saved = {}

        # Save player mappings
        player_data = {
            'player_to_token': self.player_to_token,
            'token_to_player': {str(k): v for k, v in self.token_to_player.items()},
            'player_id_to_name': self.player_id_to_name,
            'player_name_to_id': self.player_name_to_id,
            'player_stats': self.player_stats,
            'vocab_size': self._vocab_size,
            'special_tokens': self.SPECIAL_TOKENS
        }
        player_file = output_path / "player_tokenizer.json"
        with open(player_file, 'w') as f:
            json.dump(player_data, f, indent=2)
        files_saved['player_tokenizer'] = player_file
        logger.info(f"Saved player tokenizer to {player_file}")

        # Save team mappings
        team_data = {
            'team_to_token': self.team_to_token,
            'token_to_team': {str(k): v for k, v in self.token_to_team.items()},
            'vocab_size': self._team_vocab_size,
            'special_tokens': self.SPECIAL_TOKENS
        }
        team_file = output_path / "team_tokenizer.json"
        with open(team_file, 'w') as f:
            json.dump(team_data, f, indent=2)
        files_saved['team_tokenizer'] = team_file
        logger.info(f"Saved team tokenizer to {team_file}")

        return files_saved

    @classmethod
    def load(cls, input_dir: str = "data/processed") -> 'PlayerTokenizer':
        """Load tokenizer mappings from files."""
        input_path = Path(input_dir)
        tokenizer = cls()

        # Load player mappings
        player_file = input_path / "player_tokenizer.json"
        with open(player_file, 'r') as f:
            player_data = json.load(f)

        tokenizer.player_to_token = player_data['player_to_token']
        tokenizer.token_to_player = {int(k): v for k, v in player_data['token_to_player'].items()}
        tokenizer.player_id_to_name = player_data['player_id_to_name']
        tokenizer.player_name_to_id = player_data['player_name_to_id']
        tokenizer.player_stats = player_data.get('player_stats', {})
        tokenizer._vocab_size = player_data['vocab_size']

        # Load team mappings
        team_file = input_path / "team_tokenizer.json"
        with open(team_file, 'r') as f:
            team_data = json.load(f)

        tokenizer.team_to_token = team_data['team_to_token']
        tokenizer.token_to_team = {int(k): v for k, v in team_data['token_to_team'].items()}
        tokenizer._team_vocab_size = team_data['vocab_size']

        logger.info(f"Loaded tokenizer: {tokenizer.vocab_size} player tokens, "
                   f"{tokenizer.team_vocab_size} team tokens")

        return tokenizer

    def get_player_embedding_init(self, embedding_dim: int = 32) -> np.ndarray:
        """
        Create initial embedding weights based on player statistics.

        This provides better initialization than random weights by encoding
        frequency information.
        """
        embeddings = np.random.randn(self.vocab_size, embedding_dim) * 0.1

        # Special tokens get zero vectors
        embeddings[self.SPECIAL_TOKENS['PAD']] = 0
        embeddings[self.SPECIAL_TOKENS['UNK']] = 0

        # Scale embeddings by log frequency for known players
        for player_id, token in self.player_to_token.items():
            stats = self.player_stats.get(player_id, {})
            appearances = stats.get('appearances', 1)
            # Scale by log frequency
            scale = np.log1p(appearances) / 10.0
            embeddings[token] *= scale

        return embeddings


def main():
    """Main function to run player tokenization."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Player Tokenization")
    logger.info("=" * 60)

    # Load cleaned data
    df = pd.read_parquet("data/processed/cleaned_games.parquet")
    logger.info(f"Loaded {len(df)} cleaned games")

    # Need to re-parse starter lists since they may be stored as strings
    def parse_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [s.strip() for s in x.strip('[]').split(',') if s.strip()]
        return []

    if 'home_starter_id_list' not in df.columns:
        # Re-run the parsing
        import json
        import numpy as np

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

        def extract_names(starters):
            parsed = parse_starters(starters)
            return [p.get('name', '') for p in parsed if isinstance(p, dict)]

        df['home_starter_id_list'] = df['home_starters'].apply(extract_ids)
        df['away_starter_id_list'] = df['away_starters'].apply(extract_ids)
        df['home_starter_names'] = df['home_starters'].apply(extract_names)
        df['away_starter_names'] = df['away_starters'].apply(extract_names)

    # Create and fit tokenizer
    tokenizer = PlayerTokenizer()
    tokenizer.fit(df, min_appearances=1)

    # Save tokenizer
    tokenizer.save()

    # Print some statistics
    logger.info("\n" + "=" * 60)
    logger.info("TOKENIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total player vocabulary: {tokenizer.vocab_size}")
    logger.info(f"Total team vocabulary: {tokenizer.team_vocab_size}")

    # Show top 20 most common players
    sorted_players = sorted(
        tokenizer.player_stats.items(),
        key=lambda x: -x[1]['appearances']
    )[:20]

    logger.info("\nTop 20 most frequent players:")
    for i, (pid, stats) in enumerate(sorted_players, 1):
        token = tokenizer.player_to_token[pid]
        logger.info(f"  {i:2d}. Token {token:4d}: {stats['name']:<25} ({stats['appearances']} appearances)")

    # Show sample encodings
    logger.info("\nSample encodings:")
    sample_players = ['davisan02', 'jamesle01', 'curryst01', 'doncilu01']
    for pid in sample_players:
        token = tokenizer.encode_player(pid)
        name = tokenizer.player_id_to_name.get(pid, 'Unknown')
        logger.info(f"  {pid} ({name}): token {token}")

    return tokenizer


if __name__ == "__main__":
    main()
