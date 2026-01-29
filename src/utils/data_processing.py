"""
Data Processing and Feature Engineering

Processes raw jump ball + first scorer data into ML-ready features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JumpBallAggregator:
    """Aggregates jump ball data into player-level statistics."""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.player_stats: Optional[pd.DataFrame] = None
        self.matchup_stats: Optional[pd.DataFrame] = None
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load raw data from parquet file."""
        if filepath:
            self.raw_data = pd.read_parquet(filepath)
        else:
            # Load all parquet files in data directory
            files = list(self.data_path.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in {self.data_path}")
            
            dfs = [pd.read_parquet(f) for f in files]
            self.raw_data = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Loaded {len(self.raw_data)} games")
        return self.raw_data
    
    def compute_player_jump_ball_stats(self) -> pd.DataFrame:
        """
        Compute jump ball win/loss record for each player.
        
        Returns DataFrame with:
        - player_name
        - total_jump_balls
        - wins
        - losses  
        - win_rate
        - avg_opponent_win_rate (strength of schedule)
        """
        if self.raw_data is None:
            raise ValueError("Must load data first")
        
        # Filter to games with valid jump ball data
        jb_data = self.raw_data[
            self.raw_data['jump_ball_winner'].notna() & 
            self.raw_data['jump_ball_loser'].notna()
        ].copy()
        
        logger.info(f"Processing {len(jb_data)} games with jump ball data")
        
        # Count wins and losses for each player
        wins = jb_data.groupby('jump_ball_winner').size().reset_index(name='wins')
        wins.columns = ['player_name', 'wins']
        
        losses = jb_data.groupby('jump_ball_loser').size().reset_index(name='losses')
        losses.columns = ['player_name', 'losses']
        
        # Merge
        player_stats = wins.merge(losses, on='player_name', how='outer').fillna(0)
        player_stats['wins'] = player_stats['wins'].astype(int)
        player_stats['losses'] = player_stats['losses'].astype(int)
        player_stats['total_jump_balls'] = player_stats['wins'] + player_stats['losses']
        player_stats['win_rate'] = player_stats['wins'] / player_stats['total_jump_balls']
        
        # Compute strength of schedule (average opponent win rate)
        # First, get all opponents faced
        opponent_records = defaultdict(list)
        
        for _, row in jb_data.iterrows():
            winner = row['jump_ball_winner']
            loser = row['jump_ball_loser']
            opponent_records[winner].append(loser)
            opponent_records[loser].append(winner)
        
        # Create mapping of player to win rate
        win_rate_map = dict(zip(player_stats['player_name'], player_stats['win_rate']))
        
        # Compute average opponent win rate
        avg_opp_wr = []
        for player in player_stats['player_name']:
            opponents = opponent_records.get(player, [])
            if opponents:
                opp_win_rates = [win_rate_map.get(opp, 0.5) for opp in opponents]
                avg_opp_wr.append(np.mean(opp_win_rates))
            else:
                avg_opp_wr.append(0.5)
        
        player_stats['avg_opponent_win_rate'] = avg_opp_wr
        
        # Adjusted win rate (regression to mean based on sample size)
        # Using a simple Bayesian prior of 0.5 with strength of 10 games
        prior_strength = 10
        player_stats['adjusted_win_rate'] = (
            (player_stats['wins'] + prior_strength * 0.5) / 
            (player_stats['total_jump_balls'] + prior_strength)
        )
        
        self.player_stats = player_stats.sort_values('total_jump_balls', ascending=False)
        return self.player_stats
    
    def compute_matchup_stats(self) -> pd.DataFrame:
        """
        Compute head-to-head jump ball records.
        
        Returns DataFrame with:
        - player1, player2 (alphabetically ordered)
        - player1_wins, player2_wins
        - total_matchups
        """
        if self.raw_data is None:
            raise ValueError("Must load data first")
        
        jb_data = self.raw_data[
            self.raw_data['jump_ball_winner'].notna() & 
            self.raw_data['jump_ball_loser'].notna()
        ].copy()
        
        matchups = defaultdict(lambda: {'p1_wins': 0, 'p2_wins': 0})
        
        for _, row in jb_data.iterrows():
            winner = row['jump_ball_winner']
            loser = row['jump_ball_loser']
            
            # Order alphabetically for consistent key
            p1, p2 = sorted([winner, loser])
            key = (p1, p2)
            
            if winner == p1:
                matchups[key]['p1_wins'] += 1
            else:
                matchups[key]['p2_wins'] += 1
        
        matchup_list = []
        for (p1, p2), record in matchups.items():
            matchup_list.append({
                'player1': p1,
                'player2': p2,
                'player1_wins': record['p1_wins'],
                'player2_wins': record['p2_wins'],
                'total_matchups': record['p1_wins'] + record['p2_wins']
            })
        
        self.matchup_stats = pd.DataFrame(matchup_list)
        self.matchup_stats = self.matchup_stats.sort_values('total_matchups', ascending=False)
        
        return self.matchup_stats
    
    def get_matchup_prediction(self, player1: str, player2: str) -> Dict:
        """
        Predict jump ball winner probability for a specific matchup.
        
        Uses:
        1. Head-to-head record if available
        2. Overall win rates
        3. Bayesian adjustment for sample size
        """
        if self.player_stats is None:
            self.compute_player_jump_ball_stats()
        if self.matchup_stats is None:
            self.compute_matchup_stats()
        
        result = {
            'player1': player1,
            'player2': player2,
            'player1_win_prob': 0.5,
            'data_quality': 'none',
            'head_to_head_record': None
        }
        
        # Get individual stats
        p1_stats = self.player_stats[self.player_stats['player_name'] == player1]
        p2_stats = self.player_stats[self.player_stats['player_name'] == player2]
        
        p1_wr = p1_stats['adjusted_win_rate'].values[0] if len(p1_stats) > 0 else 0.5
        p2_wr = p2_stats['adjusted_win_rate'].values[0] if len(p2_stats) > 0 else 0.5
        
        # Base probability from individual win rates
        # Using log-odds combination
        if p1_wr > 0 and p1_wr < 1 and p2_wr > 0 and p2_wr < 1:
            p1_logit = np.log(p1_wr / (1 - p1_wr))
            p2_logit = np.log(p2_wr / (1 - p2_wr))
            diff_logit = p1_logit - p2_logit
            base_prob = 1 / (1 + np.exp(-diff_logit))
        else:
            base_prob = 0.5
        
        result['base_probability'] = base_prob
        result['player1_overall_wr'] = float(p1_wr)
        result['player2_overall_wr'] = float(p2_wr)
        
        # Check for head-to-head data
        p1_sorted, p2_sorted = sorted([player1, player2])
        h2h = self.matchup_stats[
            (self.matchup_stats['player1'] == p1_sorted) & 
            (self.matchup_stats['player2'] == p2_sorted)
        ]
        
        if len(h2h) > 0:
            h2h_row = h2h.iloc[0]
            total = h2h_row['total_matchups']
            
            if player1 == p1_sorted:
                h2h_wins = h2h_row['player1_wins']
            else:
                h2h_wins = h2h_row['player2_wins']
            
            result['head_to_head_record'] = f"{h2h_wins}-{total - h2h_wins}"
            
            # Combine base probability with head-to-head using weighted average
            # Weight head-to-head more heavily as sample size increases
            h2h_prob = h2h_wins / total
            h2h_weight = min(total / 20, 0.7)  # Max 70% weight on h2h
            
            result['player1_win_prob'] = (h2h_weight * h2h_prob + (1 - h2h_weight) * base_prob)
            result['data_quality'] = 'high' if total >= 5 else 'medium'
        else:
            result['player1_win_prob'] = base_prob
            result['data_quality'] = 'low' if len(p1_stats) > 0 and len(p2_stats) > 0 else 'none'
        
        return result


class FirstScorerAggregator:
    """Aggregates first scorer data for prediction modeling."""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load raw data."""
        if filepath:
            self.raw_data = pd.read_parquet(filepath)
        else:
            files = list(Path(self.data_path).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found in {self.data_path}")
            dfs = [pd.read_parquet(f) for f in files]
            self.raw_data = pd.concat(dfs, ignore_index=True)
        return self.raw_data
    
    def compute_first_scorer_rates(self) -> pd.DataFrame:
        """
        Compute how often each player scores first when in the starting lineup.
        
        Returns DataFrame with:
        - player_name
        - times_scored_first
        - games_started (approximated)
        - first_scorer_rate
        """
        if self.raw_data is None:
            raise ValueError("Must load data first")
        
        fs_data = self.raw_data[self.raw_data['first_scorer'].notna()].copy()
        
        # Count times each player scored first
        first_scorer_counts = fs_data.groupby('first_scorer').size().reset_index(name='times_scored_first')
        first_scorer_counts.columns = ['player_name', 'times_scored_first']
        
        # Approximate games started from jump ball participation
        # (Centers usually take jump balls, but this captures most starters)
        jb_participants = pd.concat([
            self.raw_data['jump_ball_player1'],
            self.raw_data['jump_ball_player2']
        ]).value_counts().reset_index()
        jb_participants.columns = ['player_name', 'jump_balls_participated']
        
        # Merge
        stats = first_scorer_counts.merge(jb_participants, on='player_name', how='outer').fillna(0)
        
        # For players who scored first but never took jump ball, estimate games
        # This is a rough approximation - would need box score data for accuracy
        stats['estimated_games'] = stats[['times_scored_first', 'jump_balls_participated']].max(axis=1)
        stats['estimated_games'] = stats['estimated_games'].clip(lower=1)
        
        stats['first_scorer_rate'] = stats['times_scored_first'] / stats['estimated_games']
        stats['first_scorer_rate'] = stats['first_scorer_rate'].clip(upper=1.0)
        
        return stats.sort_values('times_scored_first', ascending=False)
    
    def compute_tip_to_first_score_rate(self) -> Dict:
        """
        Compute how often the team that wins the tip scores first.
        
        This is a key baseline metric.
        """
        if self.raw_data is None:
            raise ValueError("Must load data first")
        
        valid_data = self.raw_data[
            self.raw_data['tip_winner_scored_first'].notna()
        ]
        
        tip_winner_scored = valid_data['tip_winner_scored_first'].sum()
        total_games = len(valid_data)
        
        return {
            'tip_winner_scored_first_count': int(tip_winner_scored),
            'total_games': total_games,
            'tip_winner_first_score_rate': tip_winner_scored / total_games if total_games > 0 else 0.5
        }
    
    def compute_score_type_distribution(self) -> pd.DataFrame:
        """Distribution of first score types (2PT, 3PT, FT)."""
        if self.raw_data is None:
            raise ValueError("Must load data first")
        
        valid_data = self.raw_data[self.raw_data['first_score_type'].notna()]
        
        dist = valid_data['first_score_type'].value_counts().reset_index()
        dist.columns = ['score_type', 'count']
        dist['percentage'] = dist['count'] / dist['count'].sum()
        
        return dist


class FeatureEngineer:
    """Creates ML-ready features from aggregated data."""
    
    def __init__(self, jump_ball_agg: JumpBallAggregator, first_scorer_agg: FirstScorerAggregator):
        self.jb_agg = jump_ball_agg
        self.fs_agg = first_scorer_agg
        
    def create_matchup_features(
        self, 
        home_center: str, 
        away_center: str,
        home_starters: List[str],
        away_starters: List[str]
    ) -> pd.DataFrame:
        """
        Create feature vector for a matchup.
        
        Features:
        - Jump ball win probability
        - First scorer rates for all starters
        - Team-level first possession scoring rates
        """
        features = {}
        
        # Jump ball prediction
        jb_pred = self.jb_agg.get_matchup_prediction(home_center, away_center)
        features['home_jb_win_prob'] = jb_pred['player1_win_prob']
        features['away_jb_win_prob'] = 1 - jb_pred['player1_win_prob']
        features['jb_data_quality'] = jb_pred['data_quality']
        
        # First scorer rates for starters
        fs_rates = self.fs_agg.compute_first_scorer_rates()
        fs_rate_map = dict(zip(fs_rates['player_name'], fs_rates['first_scorer_rate']))
        
        for i, player in enumerate(home_starters):
            features[f'home_starter_{i}_fs_rate'] = fs_rate_map.get(player, 0.05)
        
        for i, player in enumerate(away_starters):
            features[f'away_starter_{i}_fs_rate'] = fs_rate_map.get(player, 0.05)
        
        # Tip to first score rate
        tip_rate = self.fs_agg.compute_tip_to_first_score_rate()
        features['base_tip_winner_fs_rate'] = tip_rate['tip_winner_first_score_rate']
        
        return pd.DataFrame([features])


if __name__ == "__main__":
    # Example usage
    print("Testing data processing...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'game_id': ['001', '002', '003', '004', '005'],
        'jump_ball_winner': ['Player A', 'Player B', 'Player A', 'Player C', 'Player A'],
        'jump_ball_loser': ['Player B', 'Player A', 'Player C', 'Player A', 'Player B'],
        'jump_ball_winning_team': ['LAL', 'BOS', 'LAL', 'GSW', 'LAL'],
        'first_scorer': ['Player X', 'Player Y', 'Player X', 'Player Z', 'Player Y'],
        'first_scorer_team': ['LAL', 'BOS', 'LAL', 'GSW', 'BOS'],
        'tip_winner_scored_first': [True, True, True, False, False]
    })
    
    # Test jump ball aggregator
    jb_agg = JumpBallAggregator()
    jb_agg.raw_data = sample_data
    
    player_stats = jb_agg.compute_player_jump_ball_stats()
    print("\nPlayer Jump Ball Stats:")
    print(player_stats)
    
    matchup_stats = jb_agg.compute_matchup_stats()
    print("\nMatchup Stats:")
    print(matchup_stats)
    
    prediction = jb_agg.get_matchup_prediction('Player A', 'Player B')
    print("\nMatchup Prediction (A vs B):")
    print(prediction)
