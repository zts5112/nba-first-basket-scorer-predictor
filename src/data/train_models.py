"""
Model Training Script for NBA First Basket Scorer Prediction

Trains three models:
1. Jump Ball Winner Model - Predicts which player wins the tip
2. Team First Scorer Model - Predicts if tip-winning team scores first
3. Player First Scorer Model - Predicts which of 10 starters scores first

Uses gradient boosting (LightGBM if available, else sklearn).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import joblib
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    logger.info("LightGBM not available, using sklearn GradientBoosting")


class JumpBallModel:
    """
    Predicts jump ball winner probability.

    This is a binary classification: given two players, which one wins?
    We predict probability that player1 (home center) wins.
    """

    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type if HAS_LIGHTGBM or model_type != 'lightgbm' else 'gbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for jump ball prediction."""
        features = []

        # Win rate differential
        features.append(df['jb_winner_historical_win_rate'] - df['jb_loser_historical_win_rate'])

        # Experience (log-transformed)
        features.append(np.log1p(df['jb_winner_total_jbs']))
        features.append(np.log1p(df['jb_loser_total_jbs']))

        # Head-to-head
        features.append(df['jb_winner_h2h_wins'])
        features.append(df['jb_winner_h2h_losses'])
        features.append(np.log1p(df['h2h_total_matchups']))

        # Predicted probability (from historical stats)
        features.append(df['jb_winner_predicted_win_prob'])

        # Home advantage (is the winner the home player?)
        features.append(df['home_won_tip'].astype(float))

        self.feature_names = [
            'win_rate_diff', 'winner_experience', 'loser_experience',
            'h2h_wins', 'h2h_losses', 'h2h_total',
            'predicted_prob', 'home_won_tip'
        ]

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the jump ball model."""
        logger.info("Training Jump Ball Model...")

        X_train = self.get_features(train_df)
        # Target: Did the jump ball winner actually win? (always 1 by our data construction)
        # We need to reframe: predict home_won_tip
        y_train = train_df['home_won_tip'].values

        X_train = self.scaler.fit_transform(X_train)

        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # Evaluate on training data
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_pred)
        logger.info(f"  Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")

        if val_df is not None:
            X_val = self.scaler.transform(self.get_features(val_df))
            y_val = val_df['home_won_tip'].values
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"  Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability that home team wins jump ball."""
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)[:, 1]


class TeamFirstScorerModel:
    """
    Predicts if the tip-winning team scores first.

    Binary classification: P(tip winner scores first)
    """

    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type if HAS_LIGHTGBM or model_type != 'lightgbm' else 'gbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for team first scorer prediction."""
        features = []

        # Team historical rates
        features.append(df['home_team_tip_win_rate'])
        features.append(df['away_team_tip_win_rate'])
        features.append(df['home_team_fs_when_tip_win_rate'])
        features.append(df['away_team_fs_when_tip_win_rate'])

        # Which team has the tip
        features.append(df['home_won_tip'].astype(float))

        # Jump ball participant stats
        features.append(df['jb_winner_predicted_win_prob'])

        # Starter first scorer rates (sum for each team)
        home_fs_rates = df[[f'home_starter_{i}_fs_rate' for i in range(5)]].sum(axis=1)
        away_fs_rates = df[[f'away_starter_{i}_fs_rate' for i in range(5)]].sum(axis=1)
        features.append(home_fs_rates)
        features.append(away_fs_rates)

        # Experience of starters
        home_games = df[[f'home_starter_{i}_games' for i in range(5)]].mean(axis=1)
        away_games = df[[f'away_starter_{i}_games' for i in range(5)]].mean(axis=1)
        features.append(np.log1p(home_games))
        features.append(np.log1p(away_games))

        # Day/month
        features.append(df['day_of_week'])
        features.append(df['month'])

        self.feature_names = [
            'home_tip_win_rate', 'away_tip_win_rate',
            'home_fs_when_tip_rate', 'away_fs_when_tip_rate',
            'home_won_tip', 'jb_predicted_prob',
            'home_total_fs_rate', 'away_total_fs_rate',
            'home_avg_experience', 'away_avg_experience',
            'day_of_week', 'month'
        ]

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the team first scorer model."""
        logger.info("Training Team First Scorer Model...")

        X_train = self.get_features(train_df)
        y_train = train_df['tip_winner_scored_first'].values

        X_train = self.scaler.fit_transform(X_train)

        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # Baseline: always predict majority class
        baseline_acc = y_train.mean()  # ~0.648

        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred)

        logger.info(f"  Baseline (always tip winner): {baseline_acc:.4f}")
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


class PlayerFirstScorerModel:
    """
    Predicts which of the 10 starters scores first.

    Multi-class classification with 10 classes (positions 0-9).
    Position 0-4 = home starters, 5-9 = away starters.
    """

    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type if HAS_LIGHTGBM or model_type != 'lightgbm' else 'gbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for player first scorer prediction."""
        features = []

        # Individual starter first scorer rates
        for i in range(5):
            features.append(df[f'home_starter_{i}_fs_rate'])
        for i in range(5):
            features.append(df[f'away_starter_{i}_fs_rate'])

        # Individual starter experience
        for i in range(5):
            features.append(np.log1p(df[f'home_starter_{i}_games']))
        for i in range(5):
            features.append(np.log1p(df[f'away_starter_{i}_games']))

        # Team-level features
        features.append(df['home_won_tip'].astype(float))
        features.append(df['home_team_fs_when_tip_win_rate'])
        features.append(df['away_team_fs_when_tip_win_rate'])

        # Jump ball features
        features.append(df['jb_winner_predicted_win_prob'])

        self.feature_names = (
            [f'home_{i}_fs_rate' for i in range(5)] +
            [f'away_{i}_fs_rate' for i in range(5)] +
            [f'home_{i}_exp' for i in range(5)] +
            [f'away_{i}_exp' for i in range(5)] +
            ['home_won_tip', 'home_fs_tip_rate', 'away_fs_tip_rate', 'jb_prob']
        )

        X = np.column_stack(features)
        return X

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the player first scorer model."""
        logger.info("Training Player First Scorer Model...")

        # Filter to only games where a starter scored first (position 0-9)
        train_valid = train_df[train_df['first_scorer_position'] >= 0].copy()

        X_train = self.get_features(train_valid)
        y_train = train_valid['first_scorer_position'].values

        X_train = self.scaler.fit_transform(X_train)

        if self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                num_class=10,
                objective='multiclass'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # Baseline: random guess = 10%
        baseline_acc = 0.10

        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred_proba)

        # Top-3 accuracy
        top3_correct = sum(y in np.argsort(p)[-3:] for y, p in zip(y_train, train_pred_proba))
        train_top3_acc = top3_correct / len(y_train)

        logger.info(f"  Baseline (random): {baseline_acc:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.4f}, Top-3 Acc: {train_top3_acc:.4f}, LogLoss: {train_logloss:.4f}")

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

            logger.info(f"  Val Accuracy: {val_acc:.4f}, Top-3 Acc: {val_top3_acc:.4f}, LogLoss: {val_logloss:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability for each of 10 starter positions."""
        valid_df = df[df['first_scorer_position'] >= 0].copy() if 'first_scorer_position' in df.columns else df
        X = self.scaler.transform(self.get_features(valid_df))
        return self.model.predict_proba(X)


class ModelTrainer:
    """Orchestrates training of all models."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.jb_model = None
        self.team_fs_model = None
        self.player_fs_model = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test data."""
        train_df = pd.read_parquet(self.data_dir / "train.parquet")
        val_df = pd.read_parquet(self.data_dir / "val.parquet")
        test_df = pd.read_parquet(self.data_dir / "test.parquet")

        logger.info(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def train_all(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train all three models."""
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)

        # Train jump ball model
        self.jb_model = JumpBallModel()
        self.jb_model.fit(train_df, val_df)

        # Train team first scorer model
        self.team_fs_model = TeamFirstScorerModel()
        self.team_fs_model.fit(train_df, val_df)

        # Train player first scorer model
        self.player_fs_model = PlayerFirstScorerModel()
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
        jb_acc = accuracy_score(y_jb, (jb_pred > 0.5).astype(int))
        jb_auc = roc_auc_score(y_jb, jb_pred)
        results['jump_ball'] = {'accuracy': jb_acc, 'auc': jb_auc}
        logger.info(f"\nJump Ball Model:")
        logger.info(f"  Test Accuracy: {jb_acc:.4f}, AUC: {jb_auc:.4f}")

        # Team first scorer model
        team_pred = self.team_fs_model.predict_proba(test_df)
        y_team = test_df['tip_winner_scored_first'].values
        team_acc = accuracy_score(y_team, (team_pred > 0.5).astype(int))
        team_auc = roc_auc_score(y_team, team_pred)
        team_logloss = log_loss(y_team, team_pred)
        baseline = y_team.mean()
        results['team_first_scorer'] = {
            'accuracy': team_acc, 'auc': team_auc, 'logloss': team_logloss,
            'baseline': baseline
        }
        logger.info(f"\nTeam First Scorer Model:")
        logger.info(f"  Baseline (always tip winner): {baseline:.4f}")
        logger.info(f"  Test Accuracy: {team_acc:.4f}, AUC: {team_auc:.4f}, LogLoss: {team_logloss:.4f}")

        # Player first scorer model
        test_valid = test_df[test_df['first_scorer_position'] >= 0]
        player_pred = self.player_fs_model.predict_proba(test_valid)
        y_player = test_valid['first_scorer_position'].values
        player_acc = accuracy_score(y_player, np.argmax(player_pred, axis=1))
        player_logloss = log_loss(y_player, player_pred)

        # Top-3 accuracy
        top3_correct = sum(y in np.argsort(p)[-3:] for y, p in zip(y_player, player_pred))
        player_top3_acc = top3_correct / len(y_player)

        results['player_first_scorer'] = {
            'accuracy': player_acc, 'top3_accuracy': player_top3_acc,
            'logloss': player_logloss, 'baseline': 0.10
        }
        logger.info(f"\nPlayer First Scorer Model:")
        logger.info(f"  Baseline (random): 0.1000")
        logger.info(f"  Test Accuracy: {player_acc:.4f}, Top-3 Acc: {player_top3_acc:.4f}, LogLoss: {player_logloss:.4f}")

        # Position-wise accuracy
        logger.info("\n  Per-Position Accuracy:")
        for pos in range(10):
            mask = y_player == pos
            if mask.sum() > 0:
                pos_acc = (np.argmax(player_pred[mask], axis=1) == pos).mean()
                pos_label = f"Home {pos}" if pos < 5 else f"Away {pos-5}"
                logger.info(f"    {pos_label}: {pos_acc:.4f} (n={mask.sum()})")

        return results

    def save_models(self):
        """Save trained models to disk."""
        joblib.dump(self.jb_model, self.model_dir / "jump_ball_model.joblib")
        joblib.dump(self.team_fs_model, self.model_dir / "team_first_scorer_model.joblib")
        joblib.dump(self.player_fs_model, self.model_dir / "player_first_scorer_model.joblib")

        logger.info(f"\nModels saved to {self.model_dir}")

    def load_models(self):
        """Load trained models from disk."""
        self.jb_model = joblib.load(self.model_dir / "jump_ball_model.joblib")
        self.team_fs_model = joblib.load(self.model_dir / "team_first_scorer_model.joblib")
        self.player_fs_model = joblib.load(self.model_dir / "player_first_scorer_model.joblib")

        logger.info(f"Models loaded from {self.model_dir}")
        return self


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Model Training")
    logger.info("=" * 60)

    trainer = ModelTrainer()

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
    logger.info(f"  Jump Ball: {results['jump_ball']['accuracy']:.1%} accuracy (vs 50% random)")
    logger.info(f"  Team First Scorer: {results['team_first_scorer']['accuracy']:.1%} accuracy (vs {results['team_first_scorer']['baseline']:.1%} baseline)")
    logger.info(f"  Player First Scorer: {results['player_first_scorer']['accuracy']:.1%} accuracy (vs 10% random)")
    logger.info(f"  Player First Scorer Top-3: {results['player_first_scorer']['top3_accuracy']:.1%} (vs 30% random)")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
