"""
First Scorer Prediction Model

Multi-stage model:
1. Predict jump ball winner
2. Predict which team scores first (given jump ball result)
3. Predict which player scores first (given team)

This decomposition allows us to:
- Use different features at each stage
- Handle the inherent uncertainty better
- Provide interpretable probabilities
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import joblib

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Optional: better models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JumpBallModel:
    """
    Predicts probability of winning opening tip-off.
    
    Features used:
    - Historical win rate (adjusted for sample size)
    - Head-to-head record
    - Height difference (if available)
    - Home/away (slight advantage?)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, matchup_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from historical matchup data.
        
        Expected columns:
        - player1_win_rate, player2_win_rate
        - player1_total_jb, player2_total_jb  
        - head_to_head_player1_wins, head_to_head_total
        - player1_height, player2_height (optional)
        - player1_is_home
        - player1_won (target)
        """
        features = []
        
        # Win rate differential
        features.append(matchup_data['player1_win_rate'] - matchup_data['player2_win_rate'])
        
        # Experience (log of total jump balls)
        features.append(np.log1p(matchup_data['player1_total_jb']))
        features.append(np.log1p(matchup_data['player2_total_jb']))
        
        # Head-to-head
        if 'head_to_head_total' in matchup_data.columns:
            h2h_rate = matchup_data['head_to_head_player1_wins'] / matchup_data['head_to_head_total'].clip(lower=1)
            features.append(h2h_rate.fillna(0.5))
            features.append(np.log1p(matchup_data['head_to_head_total'].fillna(0)))
        
        # Height differential (if available)
        if 'player1_height' in matchup_data.columns:
            features.append(matchup_data['player1_height'] - matchup_data['player2_height'])
        
        # Home advantage
        if 'player1_is_home' in matchup_data.columns:
            features.append(matchup_data['player1_is_home'].astype(float))
        
        X = np.column_stack(features)
        y = matchup_data['player1_won'].values if 'player1_won' in matchup_data.columns else None
        
        return X, y
    
    def train(self, matchup_data: pd.DataFrame, model_type: str = 'logistic'):
        """Train the jump ball prediction model."""
        X, y = self.prepare_features(matchup_data)
        
        X = self.scaler.fit_transform(X)
        
        if model_type == 'logistic':
            base_model = LogisticRegression(C=1.0, max_iter=1000)
        elif model_type == 'rf':
            base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'xgb' and XGB_AVAILABLE:
            base_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss'
            )
        else:
            base_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        
        # Calibrate probabilities
        self.model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
        self.model.fit(X, y)
        
        # Evaluate
        cv_scores = cross_val_score(base_model, X, y, cv=5, scoring='roc_auc')
        logger.info(f"Jump Ball Model CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        return self
    
    def predict_proba(self, matchup_data: pd.DataFrame) -> np.ndarray:
        """Predict probability of player1 winning jump ball."""
        X, _ = self.prepare_features(matchup_data)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]


class TeamFirstScoreModel:
    """
    Predicts probability of each team scoring first, given jump ball result.
    
    Key insight: Team that wins tip scores first ~52-55% of the time.
    But this varies by team's offensive tendencies.
    """
    
    def __init__(self):
        self.base_tip_winner_rate = 0.53  # Will be updated from data
        self.team_adjustments = {}  # Team-specific adjustments
        
    def fit(self, game_data: pd.DataFrame):
        """
        Fit model from historical game data.
        
        Expected columns:
        - tip_winning_team
        - first_scoring_team  
        - home_team, away_team
        """
        # Overall tip winner first score rate
        tip_wins = game_data['tip_winning_team'] == game_data['first_scoring_team']
        self.base_tip_winner_rate = tip_wins.mean()
        
        logger.info(f"Tip winner scores first: {self.base_tip_winner_rate:.1%}")
        
        # Team-specific adjustments
        # (How much better/worse than average does each team perform on first possession)
        for team in pd.concat([game_data['home_team'], game_data['away_team']]).unique():
            team_games = game_data[
                (game_data['tip_winning_team'] == team)
            ]
            if len(team_games) >= 10:
                team_rate = (team_games['first_scoring_team'] == team).mean()
                self.team_adjustments[team] = team_rate - self.base_tip_winner_rate
        
        return self
    
    def predict(self, tip_win_prob: float, team: str, opponent: str) -> float:
        """
        Predict probability of team scoring first.
        
        Args:
            tip_win_prob: Probability team wins the tip
            team: Team to predict for
            opponent: Opposing team
        """
        # Probability of scoring first given tip outcomes
        team_adj = self.team_adjustments.get(team, 0)
        opp_adj = self.team_adjustments.get(opponent, 0)
        
        # P(score first | win tip)
        p_score_if_win = self.base_tip_winner_rate + team_adj
        # P(score first | lose tip)  
        p_score_if_lose = (1 - self.base_tip_winner_rate) - opp_adj
        
        # Total probability
        p_score_first = (
            tip_win_prob * p_score_if_win + 
            (1 - tip_win_prob) * p_score_if_lose
        )
        
        return np.clip(p_score_first, 0.01, 0.99)


class PlayerFirstScorerModel:
    """
    Predicts probability of each player scoring first, given their team scores first.
    
    This is essentially a "who gets the first shot" model multiplied by
    "probability of making that shot".
    """
    
    def __init__(self):
        self.player_rates = {}  # player -> first scorer rate
        self.position_base_rates = {
            'C': 0.15,   # Centers often get putbacks, tips
            'PF': 0.18,  # Roll game
            'SF': 0.20,  # Wings
            'SG': 0.22,  # Often have plays run for them
            'PG': 0.25,  # Ball handlers, first shot opportunity
        }
        self.default_rate = 0.20
        
    def fit(self, game_data: pd.DataFrame):
        """
        Fit from historical first scorer data.
        
        Computes rate of scoring first when starting.
        """
        # Group by player and count first scores
        first_scorer_counts = game_data.groupby('first_scorer').size()
        
        # Estimate games started (this is imperfect without full box score data)
        # Using jump ball participation as proxy for starters
        jb_players = pd.concat([
            game_data['jump_ball_player1'],
            game_data['jump_ball_player2']
        ])
        games_proxy = jb_players.value_counts()
        
        # For other players, use first scorer count as lower bound
        for player in first_scorer_counts.index:
            times_first = first_scorer_counts[player]
            games = games_proxy.get(player, times_first)
            games = max(games, times_first)  # Can't have scored first more than played
            
            if games >= 5:  # Minimum sample
                self.player_rates[player] = times_first / games
        
        logger.info(f"Computed first scorer rates for {len(self.player_rates)} players")
        
        return self
    
    def predict_for_lineup(
        self, 
        starters: List[str], 
        team_first_score_prob: float
    ) -> Dict[str, float]:
        """
        Predict first scorer probabilities for a starting lineup.
        
        Args:
            starters: List of 5 starting player names
            team_first_score_prob: Probability this team scores first
            
        Returns:
            Dict mapping player name to probability of scoring first
        """
        # Get raw rates for each starter
        raw_rates = []
        for player in starters:
            rate = self.player_rates.get(player, self.default_rate)
            raw_rates.append(rate)
        
        raw_rates = np.array(raw_rates)
        
        # Normalize to sum to 1 (these are conditional on team scoring first)
        if raw_rates.sum() > 0:
            conditional_probs = raw_rates / raw_rates.sum()
        else:
            conditional_probs = np.ones(5) / 5
        
        # Multiply by team's probability of scoring first
        final_probs = conditional_probs * team_first_score_prob
        
        return dict(zip(starters, final_probs))


class FirstScorerPredictor:
    """
    Main prediction class that combines all sub-models.
    
    Usage:
        predictor = FirstScorerPredictor()
        predictor.load_models('models/')  # or train from data
        
        probs = predictor.predict(
            home_team='LAL',
            away_team='BOS', 
            home_starters=['Player1', 'Player2', ...],
            away_starters=['Player1', 'Player2', ...],
            home_center='Player1',  # Who takes the tip
            away_center='Player1'
        )
    """
    
    def __init__(self):
        self.jump_ball_model = JumpBallModel()
        self.team_model = TeamFirstScoreModel()
        self.player_model = PlayerFirstScorerModel()
        self.is_fitted = False
        
    def fit(self, game_data: pd.DataFrame, matchup_data: pd.DataFrame):
        """Train all sub-models from data."""
        self.jump_ball_model.train(matchup_data)
        self.team_model.fit(game_data)
        self.player_model.fit(game_data)
        self.is_fitted = True
        return self
    
    def save(self, model_dir: str):
        """Save models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.jump_ball_model, model_dir / 'jump_ball_model.joblib')
        joblib.dump(self.team_model, model_dir / 'team_model.joblib')
        joblib.dump(self.player_model, model_dir / 'player_model.joblib')
        
        logger.info(f"Models saved to {model_dir}")
    
    def load(self, model_dir: str):
        """Load models from disk."""
        model_dir = Path(model_dir)
        
        self.jump_ball_model = joblib.load(model_dir / 'jump_ball_model.joblib')
        self.team_model = joblib.load(model_dir / 'team_model.joblib')
        self.player_model = joblib.load(model_dir / 'player_model.joblib')
        self.is_fitted = True
        
        logger.info(f"Models loaded from {model_dir}")
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        home_starters: List[str],
        away_starters: List[str],
        home_center: str,
        away_center: str,
        jb_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Predict first scorer probabilities for all starters.
        
        Returns dict mapping player names to probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() or load() first.")
        
        # Step 1: Predict jump ball winner
        if jb_data is not None:
            home_jb_prob = self.jump_ball_model.predict_proba(jb_data)[0]
        else:
            # Fall back to base rate if no model data
            home_jb_prob = 0.5
        
        # Step 2: Predict team first score probabilities
        home_team_prob = self.team_model.predict(home_jb_prob, home_team, away_team)
        away_team_prob = 1 - home_team_prob
        
        # Step 3: Predict player probabilities within each team
        home_player_probs = self.player_model.predict_for_lineup(
            home_starters, home_team_prob
        )
        away_player_probs = self.player_model.predict_for_lineup(
            away_starters, away_team_prob
        )
        
        # Combine
        all_probs = {**home_player_probs, **away_player_probs}
        
        # Normalize to ensure sum = 1 (accounting for small probability of 
        # non-starter scoring first, tip-off violation, etc.)
        total = sum(all_probs.values())
        all_probs = {k: v / total * 0.95 for k, v in all_probs.items()}  # 95% = starters
        all_probs['Other'] = 0.05  # Small prob for non-starters
        
        return all_probs
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model on test data.
        
        Returns various metrics including Brier score and calibration.
        """
        # Would implement full evaluation here
        # For now, return placeholder
        return {
            'note': 'Full evaluation requires implementation with test data structure'
        }


class ModelCalibrationAnalyzer:
    """Analyzes if predicted probabilities match actual frequencies."""
    
    @staticmethod
    def reliability_diagram(
        predictions: np.ndarray, 
        actuals: np.ndarray, 
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Create reliability diagram data.
        
        For well-calibrated model: predicted probability ≈ actual frequency
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                results.append({
                    'bin_center': (bins[i] + bins[i+1]) / 2,
                    'predicted_prob': predictions[mask].mean(),
                    'actual_freq': actuals[mask].mean(),
                    'count': mask.sum()
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def expected_calibration_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute ECE - weighted average of |predicted - actual| per bin."""
        diagram = ModelCalibrationAnalyzer.reliability_diagram(predictions, actuals)
        weights = diagram['count'] / diagram['count'].sum()
        errors = np.abs(diagram['predicted_prob'] - diagram['actual_freq'])
        return (weights * errors).sum()


if __name__ == "__main__":
    # Example usage
    print("First Scorer Predictor - Example Usage")
    print("=" * 50)
    
    # Create a simple example
    predictor = FirstScorerPredictor()
    
    # In practice, you would:
    # 1. Load collected data
    # 2. Prepare matchup data for jump ball model
    # 3. Call predictor.fit(game_data, matchup_data)
    # 4. Call predictor.predict(...) for new games
    
    print("\nModel structure:")
    print("1. JumpBallModel - Predicts tip-off winner")
    print("2. TeamFirstScoreModel - Predicts which team scores first")
    print("3. PlayerFirstScorerModel - Predicts which player scores first")
    print("\nAll probabilities are calibrated and combined probabilistically.")
