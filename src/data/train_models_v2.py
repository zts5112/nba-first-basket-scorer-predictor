"""
Model Training Script V2 for NBA First Basket Scorer Prediction

Uses the V2 features that properly separate pre-game info from outcomes.

Trains three models:
1. Jump Ball Winner Model - Predicts which player wins the tip (no leakage)
2. Team First Scorer Model - Predicts if tip-winning team scores first
3. Player First Scorer Model - Predicts which of 10 starters scores first
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import joblib
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JumpBallModelV2:
    """
    Predicts which player wins the jump ball.

    Binary classification: P(home player wins tip)
    Uses only pre-game features (no outcome leakage).
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for jump ball prediction."""
        features = []

        # Historical win rates
        features.append(df['home_jb_player_win_rate'].values)
        features.append(df['away_jb_player_win_rate'].values)
        features.append(df['jb_win_rate_diff'].values)

        # Experience (log-transformed)
        features.append(np.log1p(df['home_jb_player_total_jbs'].values))
        features.append(np.log1p(df['away_jb_player_total_jbs'].values))

        # Head-to-head
        features.append(df['home_jb_h2h_win_rate'].values)
        features.append(np.log1p(df['h2h_total_matchups'].values))

        # Predicted probability from simple model
        features.append(df['home_jb_predicted_win_prob'].values)

        # Team historical tip rates
        features.append(df['home_team_tip_win_rate'].values)
        features.append(df['away_team_tip_win_rate'].values)

        self.feature_names = [
            'home_win_rate', 'away_win_rate', 'win_rate_diff',
            'home_experience', 'away_experience',
            'h2h_win_rate', 'h2h_matchups',
            'predicted_prob',
            'home_team_tip_rate', 'away_team_tip_rate'
        ]

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the jump ball model."""
        logger.info("Training Jump Ball Model V2...")

        X_train = self.get_features(train_df)
        y_train = train_df['home_won_tip'].values

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Baseline: home wins ~51.3% of tips
        baseline = y_train.mean()

        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_pred)
        train_brier = brier_score_loss(y_train, train_pred)

        logger.info(f"  Baseline (always home): {baseline:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}, Brier: {train_brier:.4f}")

        if val_df is not None:
            X_val = self.scaler.transform(self.get_features(val_df))
            y_val = val_df['home_won_tip'].values
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            val_brier = brier_score_loss(y_val, val_pred)
            logger.info(f"  Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, Brier: {val_brier:.4f}")

        # Feature importance
        logger.info("  Feature importance:")
        for name, imp in sorted(zip(self.feature_names, self.model.feature_importances_),
                                key=lambda x: -x[1])[:5]:
            logger.info(f"    {name}: {imp:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability that home player wins jump ball."""
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)[:, 1]


class TeamFirstScorerModelV2:
    """
    Predicts if the tip-winning team scores first.

    Binary classification: P(tip winner scores first)
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for team first scorer prediction."""
        features = []

        # Team historical rates
        features.append(df['home_team_tip_win_rate'].values)
        features.append(df['away_team_tip_win_rate'].values)
        features.append(df['home_team_fs_when_tip_win_rate'].values)
        features.append(df['away_team_fs_when_tip_win_rate'].values)

        # Starter first scorer rates (sum for each team)
        home_fs_sum = df[[f'home_starter_{i}_fs_rate' for i in range(5)]].sum(axis=1).values
        away_fs_sum = df[[f'away_starter_{i}_fs_rate' for i in range(5)]].sum(axis=1).values
        features.append(home_fs_sum)
        features.append(away_fs_sum)

        # Experience of starters (average games)
        home_games = df[[f'home_starter_{i}_games' for i in range(5)]].mean(axis=1).values
        away_games = df[[f'away_starter_{i}_games' for i in range(5)]].mean(axis=1).values
        features.append(np.log1p(home_games))
        features.append(np.log1p(away_games))

        # Day/month
        features.append(df['day_of_week'].values)
        features.append(df['month'].values)

        # Jump ball features
        features.append(df['home_jb_predicted_win_prob'].values)
        features.append(df['jb_win_rate_diff'].values)

        self.feature_names = [
            'home_tip_rate', 'away_tip_rate',
            'home_fs_tip_rate', 'away_fs_tip_rate',
            'home_fs_sum', 'away_fs_sum',
            'home_exp', 'away_exp',
            'day_of_week', 'month',
            'jb_prob', 'jb_diff'
        ]

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the team first scorer model."""
        logger.info("Training Team First Scorer Model V2...")

        X_train = self.get_features(train_df)
        y_train = train_df['tip_winner_scored_first'].values

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Use simpler model to reduce overfitting
        self.model = LogisticRegression(
            C=0.1,  # Strong regularization
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Baseline: tip winner scores first ~64.8%
        baseline = y_train.mean()

        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred)

        logger.info(f"  Baseline (always tip winner): {baseline:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}, LogLoss: {train_logloss:.4f}")

        if val_df is not None:
            X_val = self.scaler.transform(self.get_features(val_df))
            y_val = val_df['tip_winner_scored_first'].values
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            val_logloss = log_loss(y_val, val_pred)
            logger.info(f"  Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability that tip winner scores first."""
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)[:, 1]


class PlayerFirstScorerModelV2:
    """
    Predicts which of the 10 starters scores first.

    Multi-class classification with 11 classes:
    - 0-4: home starters
    - 5-9: away starters
    - 10: bench player (if we include those)
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for player first scorer prediction."""
        features = []

        # Individual starter first scorer rates
        for i in range(5):
            features.append(df[f'home_starter_{i}_fs_rate'].values)
        for i in range(5):
            features.append(df[f'away_starter_{i}_fs_rate'].values)

        # Individual starter experience (log-transformed)
        for i in range(5):
            features.append(np.log1p(df[f'home_starter_{i}_games'].values))
        for i in range(5):
            features.append(np.log1p(df[f'away_starter_{i}_games'].values))

        # Team-level features
        features.append(df['home_jb_predicted_win_prob'].values)
        features.append(df['home_team_fs_when_tip_win_rate'].values)
        features.append(df['away_team_fs_when_tip_win_rate'].values)
        features.append(df['home_team_total_fs_rate'].values)
        features.append(df['away_team_total_fs_rate'].values)

        self.feature_names = (
            [f'home_{i}_fs_rate' for i in range(5)] +
            [f'away_{i}_fs_rate' for i in range(5)] +
            [f'home_{i}_exp' for i in range(5)] +
            [f'away_{i}_exp' for i in range(5)] +
            ['jb_prob', 'home_fs_tip', 'away_fs_tip', 'home_total_fs', 'away_total_fs']
        )

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the player first scorer model."""
        logger.info("Training Player First Scorer Model V2...")

        # Filter to only games where a starter scored first (position 0-9)
        train_valid = train_df[train_df['first_scorer_position'] >= 0].copy()

        X_train = self.get_features(train_valid)
        y_train = train_valid['first_scorer_position'].values

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Use regularized model
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Baseline: random guess = 10%
        baseline = 0.10

        train_pred = self.model.predict(X_train_scaled)
        train_pred_proba = self.model.predict_proba(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred_proba)

        # Top-3 accuracy
        top3_correct = sum(y in np.argsort(p)[-3:] for y, p in zip(y_train, train_pred_proba))
        train_top3_acc = top3_correct / len(y_train)

        logger.info(f"  Baseline (random): {baseline:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.4f}, Top-3: {train_top3_acc:.4f}, LogLoss: {train_logloss:.4f}")

        if val_df is not None:
            val_valid = val_df[val_df['first_scorer_position'] >= 0].copy()
            X_val = self.scaler.transform(self.get_features(val_valid))
            y_val = val_valid['first_scorer_position'].values

            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_logloss = log_loss(y_val, val_pred_proba)

            top3_correct = sum(y in np.argsort(p)[-3:] for y, p in zip(y_val, val_pred_proba))
            val_top3_acc = top3_correct / len(y_val)

            logger.info(f"  Val Accuracy: {val_acc:.4f}, Top-3: {val_top3_acc:.4f}, LogLoss: {val_logloss:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability for each of 10 starter positions."""
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)


class ModelTrainerV2:
    """Orchestrates training of all models using V2 features."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.jb_model = None
        self.team_fs_model = None
        self.player_fs_model = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test data (V2 features)."""
        train_df = pd.read_parquet(self.data_dir / "train_v2.parquet")
        val_df = pd.read_parquet(self.data_dir / "val_v2.parquet")
        test_df = pd.read_parquet(self.data_dir / "test_v2.parquet")

        logger.info(f"Loaded V2 data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def train_all(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train all three models."""
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS (V2)")
        logger.info("=" * 60)

        self.jb_model = JumpBallModelV2()
        self.jb_model.fit(train_df, val_df)

        self.team_fs_model = TeamFirstScorerModelV2()
        self.team_fs_model.fit(train_df, val_df)

        self.player_fs_model = PlayerFirstScorerModelV2()
        self.player_fs_model.fit(train_df, val_df)

        return self

    def evaluate_on_test(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate all models on test set."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 60)

        results = {}

        # Jump ball model
        jb_pred = self.jb_model.predict_proba(test_df)
        y_jb = test_df['home_won_tip'].values
        baseline_jb = y_jb.mean()
        jb_acc = accuracy_score(y_jb, (jb_pred > 0.5).astype(int))
        jb_auc = roc_auc_score(y_jb, jb_pred)
        jb_brier = brier_score_loss(y_jb, jb_pred)
        results['jump_ball'] = {
            'baseline': baseline_jb, 'accuracy': jb_acc,
            'auc': jb_auc, 'brier': jb_brier
        }
        logger.info(f"\nJump Ball Model:")
        logger.info(f"  Baseline (home wins): {baseline_jb:.4f}")
        logger.info(f"  Test Accuracy: {jb_acc:.4f}, AUC: {jb_auc:.4f}, Brier: {jb_brier:.4f}")

        # Team first scorer model
        team_pred = self.team_fs_model.predict_proba(test_df)
        y_team = test_df['tip_winner_scored_first'].values
        baseline_team = y_team.mean()
        team_acc = accuracy_score(y_team, (team_pred > 0.5).astype(int))
        team_auc = roc_auc_score(y_team, team_pred)
        team_logloss = log_loss(y_team, team_pred)
        results['team_first_scorer'] = {
            'baseline': baseline_team, 'accuracy': team_acc,
            'auc': team_auc, 'logloss': team_logloss
        }
        logger.info(f"\nTeam First Scorer Model:")
        logger.info(f"  Baseline (always tip winner): {baseline_team:.4f}")
        logger.info(f"  Test Accuracy: {team_acc:.4f}, AUC: {team_auc:.4f}, LogLoss: {team_logloss:.4f}")

        # Player first scorer model
        test_valid = test_df[test_df['first_scorer_position'] >= 0]
        player_pred = self.player_fs_model.predict_proba(test_valid)
        y_player = test_valid['first_scorer_position'].values
        player_acc = accuracy_score(y_player, np.argmax(player_pred, axis=1))
        player_logloss = log_loss(y_player, player_pred)

        top3_correct = sum(y in np.argsort(p)[-3:] for y, p in zip(y_player, player_pred))
        player_top3_acc = top3_correct / len(y_player)

        results['player_first_scorer'] = {
            'baseline': 0.10, 'accuracy': player_acc,
            'top3_accuracy': player_top3_acc, 'logloss': player_logloss
        }
        logger.info(f"\nPlayer First Scorer Model:")
        logger.info(f"  Baseline (random): 0.1000")
        logger.info(f"  Test Accuracy: {player_acc:.4f}, Top-3: {player_top3_acc:.4f}, LogLoss: {player_logloss:.4f}")

        return results

    def save_models(self):
        """Save trained models to disk."""
        joblib.dump(self.jb_model, self.model_dir / "jump_ball_model_v2.joblib")
        joblib.dump(self.team_fs_model, self.model_dir / "team_first_scorer_model_v2.joblib")
        joblib.dump(self.player_fs_model, self.model_dir / "player_first_scorer_model_v2.joblib")
        logger.info(f"\nModels saved to {self.model_dir}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Model Training V2")
    logger.info("=" * 60)

    trainer = ModelTrainerV2()

    # Load data
    train_df, val_df, test_df = trainer.load_data()

    # Train models
    trainer.train_all(train_df, val_df)

    # Evaluate on test set
    results = trainer.evaluate_on_test(test_df)

    # Save models
    trainer.save_models()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("\nModel Performance vs Baselines:")

    jb = results['jump_ball']
    logger.info(f"  Jump Ball: {jb['accuracy']:.1%} acc (vs {jb['baseline']:.1%} baseline), AUC={jb['auc']:.3f}")

    team = results['team_first_scorer']
    logger.info(f"  Team First Scorer: {team['accuracy']:.1%} acc (vs {team['baseline']:.1%} baseline)")

    player = results['player_first_scorer']
    logger.info(f"  Player First Scorer: {player['accuracy']:.1%} acc (vs 10% random)")
    logger.info(f"  Player Top-3: {player['top3_accuracy']:.1%} (vs 30% random)")

    # Calculate edge
    logger.info("\nPredictive Edge:")
    logger.info(f"  Jump Ball: +{(jb['accuracy'] - jb['baseline'])*100:.1f} percentage points")
    logger.info(f"  Player First Scorer: +{(player['accuracy'] - 0.10)*100:.1f} pp, Top-3: +{(player['top3_accuracy'] - 0.30)*100:.1f} pp")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
