"""
Fetch Player Statistics from NBA API

This script fetches player shooting stats, usage rates, and other
relevant metrics that can improve first scorer prediction.

Stats fetched:
- FG%, 3P%, FT%
- PPG (Points Per Game)
- Usage Rate
- Shot distribution (at rim, mid-range, 3PT)
- First quarter scoring stats
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

from nba_api.stats.endpoints import (
    commonallplayers,
    playercareerstats,
    playerdashboardbyyearoveryear,
    leaguedashplayerstats,
    playerdashboardbygeneralsplits
)
from nba_api.stats.static import players as nba_players

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting - NBA API can be strict
REQUEST_DELAY = 0.6  # seconds between requests


class PlayerStatsFetcher:
    """Fetches and caches player statistics from NBA API."""

    def __init__(self, cache_dir: str = "data/player_stats"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load Basketball Reference to NBA API ID mapping
        self.bref_to_nba_id: Dict[str, int] = {}
        self.nba_id_to_bref: Dict[int, str] = {}

        # Player stats cache
        self.player_stats: Dict[str, Dict] = {}

        # Load existing cache
        self._load_cache()

    def _load_cache(self):
        """Load cached player stats and ID mappings."""
        # Load ID mapping
        mapping_file = self.cache_dir / "bref_to_nba_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                data = json.load(f)
                self.bref_to_nba_id = data.get('bref_to_nba', {})
                self.nba_id_to_bref = {int(v): k for k, v in self.bref_to_nba_id.items()}
            logger.info(f"Loaded {len(self.bref_to_nba_id)} ID mappings from cache")

        # Load player stats
        stats_file = self.cache_dir / "player_stats_cache.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.player_stats = json.load(f)
            logger.info(f"Loaded stats for {len(self.player_stats)} players from cache")

    def _save_cache(self):
        """Save caches to disk."""
        # Save ID mapping
        mapping_file = self.cache_dir / "bref_to_nba_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'bref_to_nba': self.bref_to_nba_id,
                'updated': datetime.now().isoformat()
            }, f, indent=2)

        # Save player stats
        stats_file = self.cache_dir / "player_stats_cache.json"
        with open(stats_file, 'w') as f:
            json.dump(self.player_stats, f, indent=2)

        logger.info("Saved caches to disk")

    def build_id_mapping(self, bref_player_ids: List[str], bref_player_names: Dict[str, str]):
        """
        Build mapping from Basketball Reference IDs to NBA API IDs.

        Args:
            bref_player_ids: List of Basketball Reference player IDs
            bref_player_names: Dict mapping bref_id -> player name
        """
        logger.info(f"Building ID mapping for {len(bref_player_ids)} players...")

        # Get all NBA players
        all_nba_players = nba_players.get_players()

        # Create name-based lookup
        nba_by_name = {}
        for p in all_nba_players:
            full_name = p['full_name'].lower()
            nba_by_name[full_name] = p['id']
            # Also add last name, first name format
            parts = full_name.split()
            if len(parts) >= 2:
                alt_name = f"{parts[-1]}, {' '.join(parts[:-1])}"
                nba_by_name[alt_name] = p['id']

        matched = 0
        unmatched = []

        for bref_id in bref_player_ids:
            if bref_id in self.bref_to_nba_id:
                matched += 1
                continue

            name = bref_player_names.get(bref_id, '')
            if not name:
                unmatched.append(bref_id)
                continue

            # Try exact match
            name_lower = name.lower()
            if name_lower in nba_by_name:
                self.bref_to_nba_id[bref_id] = nba_by_name[name_lower]
                self.nba_id_to_bref[nba_by_name[name_lower]] = bref_id
                matched += 1
                continue

            # Try fuzzy match - remove accents and special chars
            name_simple = self._simplify_name(name_lower)
            for nba_name, nba_id in nba_by_name.items():
                if self._simplify_name(nba_name) == name_simple:
                    self.bref_to_nba_id[bref_id] = nba_id
                    self.nba_id_to_bref[nba_id] = bref_id
                    matched += 1
                    break
            else:
                unmatched.append((bref_id, name))

        logger.info(f"Matched {matched}/{len(bref_player_ids)} players")
        if unmatched[:10]:
            logger.info(f"Sample unmatched: {unmatched[:10]}")

        self._save_cache()
        return matched, unmatched

    def _simplify_name(self, name: str) -> str:
        """Remove accents and special characters from name."""
        # Common accent replacements
        replacements = {
            'ć': 'c', 'č': 'c', 'ğ': 'g', 'ş': 's', 'ö': 'o', 'ü': 'u',
            'ä': 'a', 'é': 'e', 'è': 'e', 'ñ': 'n', 'ń': 'n', 'ž': 'z',
            'đ': 'd', 'ī': 'i', 'ū': 'u', 'ā': 'a', 'ē': 'e', 'ō': 'o',
            "'": '', "'": '', "-": " ", ".": "", "jr": "", "sr": "", "ii": "", "iii": ""
        }
        result = name.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)
        # Remove extra whitespace
        result = ' '.join(result.split())
        return result

    def fetch_player_stats(self, bref_id: str, season: str = "2024-25") -> Optional[Dict]:
        """
        Fetch stats for a single player.

        Args:
            bref_id: Basketball Reference player ID
            season: NBA season (e.g., "2024-25")

        Returns:
            Dict with player stats or None if not found
        """
        cache_key = f"{bref_id}_{season}"
        if cache_key in self.player_stats:
            return self.player_stats[cache_key]

        nba_id = self.bref_to_nba_id.get(bref_id)
        if not nba_id:
            return None

        try:
            time.sleep(REQUEST_DELAY)

            # Get career stats for the season
            career = playercareerstats.PlayerCareerStats(
                player_id=nba_id,
                per_mode36='PerGame'
            )
            career_df = career.get_data_frames()[0]

            # Filter to requested season
            season_stats = career_df[career_df['SEASON_ID'] == season]

            if len(season_stats) == 0:
                # Try previous season
                prev_season = self._prev_season(season)
                season_stats = career_df[career_df['SEASON_ID'] == prev_season]

            if len(season_stats) == 0:
                return None

            row = season_stats.iloc[0]

            stats = {
                'nba_id': nba_id,
                'bref_id': bref_id,
                'season': season,
                'games': int(row.get('GP', 0)),
                'ppg': float(row.get('PTS', 0)),
                'fg_pct': float(row.get('FG_PCT', 0)),
                'fg3_pct': float(row.get('FG3_PCT', 0)),
                'ft_pct': float(row.get('FT_PCT', 0)),
                'fga': float(row.get('FGA', 0)),
                'fg3a': float(row.get('FG3A', 0)),
                'fta': float(row.get('FTA', 0)),
                'min': float(row.get('MIN', 0)),
                'ast': float(row.get('AST', 0)),
                'reb': float(row.get('REB', 0)),
            }

            # Calculate derived stats
            if stats['fga'] > 0:
                stats['fg3_rate'] = stats['fg3a'] / stats['fga']
            else:
                stats['fg3_rate'] = 0.0

            # Estimate usage (simplified)
            if stats['min'] > 0:
                stats['est_usage'] = (stats['fga'] + 0.44 * stats['fta']) / stats['min'] * 48
            else:
                stats['est_usage'] = 0.0

            self.player_stats[cache_key] = stats
            return stats

        except Exception as e:
            logger.warning(f"Error fetching stats for {bref_id}: {e}")
            return None

    def fetch_all_player_stats(self, bref_ids: List[str], season: str = "2024-25") -> Dict[str, Dict]:
        """
        Fetch stats for multiple players.

        Args:
            bref_ids: List of Basketball Reference player IDs
            season: NBA season

        Returns:
            Dict mapping bref_id -> stats
        """
        logger.info(f"Fetching stats for {len(bref_ids)} players for season {season}...")

        results = {}
        fetched = 0
        cached = 0

        for i, bref_id in enumerate(bref_ids):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(bref_ids)}")

            cache_key = f"{bref_id}_{season}"
            if cache_key in self.player_stats:
                results[bref_id] = self.player_stats[cache_key]
                cached += 1
                continue

            stats = self.fetch_player_stats(bref_id, season)
            if stats:
                results[bref_id] = stats
                fetched += 1

        logger.info(f"Fetched {fetched} new, {cached} from cache, {len(bref_ids) - fetched - cached} not found")
        self._save_cache()

        return results

    def fetch_league_stats(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Fetch league-wide player stats for a season.
        This is more efficient than fetching individual players.
        """
        logger.info(f"Fetching league stats for {season}...")

        try:
            time.sleep(REQUEST_DELAY)

            league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame'
            )
            df = league_stats.get_data_frames()[0]

            logger.info(f"Fetched stats for {len(df)} players")
            return df

        except Exception as e:
            logger.error(f"Error fetching league stats: {e}")
            return pd.DataFrame()

    def _prev_season(self, season: str) -> str:
        """Get previous season string."""
        # e.g., "2024-25" -> "2023-24"
        parts = season.split('-')
        year1 = int(parts[0]) - 1
        year2 = int(parts[1]) - 1
        return f"{year1}-{year2:02d}"

    def create_player_stats_dataset(self, output_file: str = "data/processed/player_stats.parquet"):
        """
        Create a comprehensive player stats dataset merged with our player tokenizer.
        """
        # Load player tokenizer
        with open("data/processed/player_tokenizer.json") as f:
            player_data = json.load(f)

        bref_ids = list(player_data['player_to_token'].keys())
        bref_names = player_data['player_id_to_name']

        logger.info(f"Processing {len(bref_ids)} players from tokenizer")

        # Build ID mapping first
        self.build_id_mapping(bref_ids, bref_names)

        # Fetch league-wide stats (more efficient)
        seasons = ["2024-25", "2023-24", "2022-23", "2021-22"]
        all_stats = []

        for season in seasons:
            league_df = self.fetch_league_stats(season)
            if len(league_df) > 0:
                league_df['season'] = season
                all_stats.append(league_df)
            time.sleep(REQUEST_DELAY)

        if not all_stats:
            logger.error("No stats fetched!")
            return None

        stats_df = pd.concat(all_stats, ignore_index=True)

        # Map NBA IDs to BREF IDs
        stats_df['bref_id'] = stats_df['PLAYER_ID'].map(self.nba_id_to_bref)

        # Select relevant columns
        cols = [
            'bref_id', 'PLAYER_ID', 'PLAYER_NAME', 'season',
            'GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'AST', 'REB', 'STL', 'BLK', 'TOV'
        ]
        cols = [c for c in cols if c in stats_df.columns]
        stats_df = stats_df[cols].copy()

        # Filter to players in our tokenizer
        stats_df = stats_df[stats_df['bref_id'].notna()]

        # Calculate additional metrics
        stats_df['fg3_rate'] = stats_df['FG3A'] / stats_df['FGA'].replace(0, np.nan)
        stats_df['fg3_rate'] = stats_df['fg3_rate'].fillna(0)

        # Estimate usage rate (simplified)
        stats_df['est_usage'] = (stats_df['FGA'] + 0.44 * stats_df['FTA']) / stats_df['MIN'].replace(0, np.nan) * 48
        stats_df['est_usage'] = stats_df['est_usage'].fillna(0)

        # Save
        output_path = Path(output_file)
        stats_df.to_parquet(output_path, index=False)
        logger.info(f"Saved player stats to {output_path}")

        # Also save as a lookup dict
        stats_dict = {}
        for _, row in stats_df.iterrows():
            bref_id = row['bref_id']
            season = row['season']
            key = f"{bref_id}_{season}"
            stats_dict[key] = {
                'ppg': float(row['PTS']),
                'fg_pct': float(row['FG_PCT']),
                'fg3_pct': float(row['FG3_PCT']),
                'ft_pct': float(row['FT_PCT']),
                'fg3_rate': float(row['fg3_rate']),
                'est_usage': float(row['est_usage']),
                'games': int(row['GP']),
                'min': float(row['MIN'])
            }

        lookup_file = Path("data/processed/player_stats_lookup.json")
        with open(lookup_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        logger.info(f"Saved lookup to {lookup_file}")

        return stats_df


def main():
    """Main function to fetch and save player stats."""
    logger.info("=" * 60)
    logger.info("Fetching NBA Player Statistics")
    logger.info("=" * 60)

    fetcher = PlayerStatsFetcher()
    stats_df = fetcher.create_player_stats_dataset()

    if stats_df is not None:
        logger.info(f"\nStats summary:")
        logger.info(f"  Total player-seasons: {len(stats_df)}")
        logger.info(f"  Unique players: {stats_df['bref_id'].nunique()}")
        logger.info(f"  Seasons: {stats_df['season'].unique().tolist()}")

        # Show sample
        logger.info("\nSample stats (top scorers):")
        top_scorers = stats_df.nlargest(10, 'PTS')[['PLAYER_NAME', 'season', 'PTS', 'FG_PCT', 'FG3_PCT']]
        print(top_scorers.to_string())


if __name__ == "__main__":
    main()
