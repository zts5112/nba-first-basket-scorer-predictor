"""
Model Training V3 - Using Enhanced Features

Trains improved models with:
1. Recent form features
2. Jump ball participant indicators
3. Position-aware features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JumpBallModelV3:
    """Predicts jump ball winner with enhanced features."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract jump ball prediction features."""
        features = [
            df['home_jb_win_rate'].values,
            df['away_jb_win_rate'].values,
            df['jb_win_rate_diff'].values,
            np.log1p(df['home_jb_total'].values),
            np.log1p(df['away_jb_total'].values),
            df['home_jb_h2h_rate'].values,
            np.log1p(df['h2h_matchups'].values),
            df['home_jb_predicted_prob'].values,
            df['home_tip_win_rate'].values,
            df['away_tip_win_rate'].values,
        ]
        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Jump Ball Model V3...")

        X_train = self.scaler.fit_transform(self.get_features(train_df))
        y_train = train_df['home_won_tip'].values

        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        self.model.fit(X_train, y_train)

        baseline = y_train.mean()
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_pred)

        logger.info(f"  Baseline: {baseline:.4f}")
        logger.info(f"  Train: Acc={train_acc:.4f}, AUC={train_auc:.4f}")

        if val_df is not None:
            X_val = self.scaler.transform(self.get_features(val_df))
            y_val = val_df['home_won_tip'].values
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"  Val: Acc={val_acc:.4f}, AUC={val_auc:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)[:, 1]


class PlayerFirstScorerModelV3:
    """
    Enhanced player first scorer model.

    Key improvements:
    - Uses recent form features
    - Accounts for jump ball participant advantage
    - Position-aware features
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix for player prediction."""
        features = []

        # Per-player features (10 starters)
        for prefix in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                      'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            # Historical first scorer rate
            features.append(df[f'{prefix}_fs_rate'].values)
            # Recent form (last 10 games)
            features.append(df[f'{prefix}_recent_fs_rate'].values)
            # Experience (log games)
            features.append(np.log1p(df[f'{prefix}_games'].values))
            # Is this player the jump ball participant?
            features.append(df[f'{prefix}_is_jb'].values.astype(float))

        # Team-level features
        features.append(df['home_jb_predicted_prob'].values)
        features.append(df['home_total_fs_rate'].values)
        features.append(df['away_total_fs_rate'].values)
        features.append(df['home_total_recent_fs'].values)
        features.append(df['away_total_recent_fs'].values)

        # Best scorer features
        features.append(df['home_max_fs_rate'].values)
        features.append(df['away_max_fs_rate'].values)

        # JB player first scorer rates
        features.append(df['home_jb_fs_rate'].values)
        features.append(df['away_jb_fs_rate'].values)

        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Player First Scorer Model V3...")

        # Filter to games where starter scored first
        train_valid = train_df[train_df['first_scorer_position'] >= 0].copy()

        X_train = self.scaler.fit_transform(self.get_features(train_valid))
        y_train = train_valid['first_scorer_position'].values

        # Use gradient boosting for better performance
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred_proba)

        # Top-3 accuracy
        top3 = sum(y in np.argsort(p)[-3:] for y, p in zip(y_train, train_pred_proba)) / len(y_train)

        logger.info(f"  Baseline: 0.1000")
        logger.info(f"  Train: Acc={train_acc:.4f}, Top3={top3:.4f}, LogLoss={train_logloss:.4f}")

        if val_df is not None:
            val_valid = val_df[val_df['first_scorer_position'] >= 0].copy()
            X_val = self.scaler.transform(self.get_features(val_valid))
            y_val = val_valid['first_scorer_position'].values

            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_logloss = log_loss(y_val, val_pred_proba)
            val_top3 = sum(y in np.argsort(p)[-3:] for y, p in zip(y_val, val_pred_proba)) / len(y_val)

            logger.info(f"  Val: Acc={val_acc:.4f}, Top3={val_top3:.4f}, LogLoss={val_logloss:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)

    def get_player_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities with player names."""
        proba = self.predict_proba(df)

        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            game_proba = proba[idx]
            for pos in range(10):
                team = 'home' if pos < 5 else 'away'
                player_idx = pos if pos < 5 else pos - 5
                results.append({
                    'game_id': row['game_id'],
                    'position': pos,
                    'team': team,
                    'player_token': row[f'{team}_{player_idx}_token'],
                    'probability': game_proba[pos]
                })

        return pd.DataFrame(results)


class ModelTrainerV3:
    """Train and evaluate V3 models."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.jb_model = None
        self.player_model = None

    def load_data(self):
        """Load V3 feature data."""
        train = pd.read_parquet(self.data_dir / "train_v3.parquet")
        val = pd.read_parquet(self.data_dir / "val_v3.parquet")
        test = pd.read_parquet(self.data_dir / "test_v3.parquet")
        logger.info(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def train_all(self, train_df, val_df):
        """Train all models."""
        logger.info("=" * 60)
        logger.info("TRAINING V3 MODELS")
        logger.info("=" * 60)

        self.jb_model = JumpBallModelV3()
        self.jb_model.fit(train_df, val_df)

        self.player_model = PlayerFirstScorerModelV3()
        self.player_model.fit(train_df, val_df)

        return self

    def evaluate(self, test_df) -> Dict:
        """Evaluate on test set."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 60)

        results = {}

        # Jump ball
        jb_pred = self.jb_model.predict_proba(test_df)
        y_jb = test_df['home_won_tip'].values
        jb_acc = accuracy_score(y_jb, (jb_pred > 0.5).astype(int))
        jb_auc = roc_auc_score(y_jb, jb_pred)
        baseline_jb = y_jb.mean()

        results['jump_ball'] = {'baseline': baseline_jb, 'accuracy': jb_acc, 'auc': jb_auc}
        logger.info(f"\nJump Ball Model:")
        logger.info(f"  Baseline: {baseline_jb:.4f}")
        logger.info(f"  Test: Acc={jb_acc:.4f}, AUC={jb_auc:.4f}")

        # Player first scorer
        test_valid = test_df[test_df['first_scorer_position'] >= 0]
        player_pred = self.player_model.predict_proba(test_valid)
        y_player = test_valid['first_scorer_position'].values

        player_acc = accuracy_score(y_player, np.argmax(player_pred, axis=1))
        player_logloss = log_loss(y_player, player_pred)
        top3 = sum(y in np.argsort(p)[-3:] for y, p in zip(y_player, player_pred)) / len(y_player)
        top5 = sum(y in np.argsort(p)[-5:] for y, p in zip(y_player, player_pred)) / len(y_player)

        results['player'] = {
            'baseline': 0.10, 'accuracy': player_acc,
            'top3': top3, 'top5': top5, 'logloss': player_logloss
        }
        logger.info(f"\nPlayer First Scorer Model:")
        logger.info(f"  Baseline: 0.1000 (random), 0.30 (top-3), 0.50 (top-5)")
        logger.info(f"  Test: Acc={player_acc:.4f}, Top3={top3:.4f}, Top5={top5:.4f}")
        logger.info(f"  LogLoss: {player_logloss:.4f}")

        # Per-position analysis
        logger.info("\n  Per-Position Accuracy:")
        for pos in range(10):
            mask = y_player == pos
            if mask.sum() > 0:
                pos_acc = (np.argmax(player_pred[mask], axis=1) == pos).mean()
                avg_prob = player_pred[mask, pos].mean()
                label = f"Home {pos}" if pos < 5 else f"Away {pos-5}"
                logger.info(f"    {label}: Acc={pos_acc:.3f}, AvgProb={avg_prob:.3f} (n={mask.sum()})")

        return results

    def save_models(self):
        """Save trained models."""
        joblib.dump(self.jb_model, self.model_dir / "jump_ball_model_v3.joblib")
        joblib.dump(self.player_model, self.model_dir / "player_first_scorer_model_v3.joblib")
        logger.info(f"\nModels saved to {self.model_dir}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Model Training V3")
    logger.info("=" * 60)

    trainer = ModelTrainerV3()
    train_df, val_df, test_df = trainer.load_data()

    trainer.train_all(train_df, val_df)
    results = trainer.evaluate(test_df)
    trainer.save_models()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY - V3 Model Performance")
    logger.info("=" * 60)

    jb = results['jump_ball']
    player = results['player']

    logger.info(f"\nJump Ball Prediction:")
    logger.info(f"  Accuracy: {jb['accuracy']:.1%} (vs {jb['baseline']:.1%} baseline)")
    logger.info(f"  Edge: +{(jb['accuracy'] - jb['baseline'])*100:.1f} percentage points")

    logger.info(f"\nPlayer First Scorer Prediction:")
    logger.info(f"  Top-1 Accuracy: {player['accuracy']:.1%} (vs 10% baseline) - Edge: +{(player['accuracy']-0.10)*100:.1f}pp")
    logger.info(f"  Top-3 Accuracy: {player['top3']:.1%} (vs 30% baseline) - Edge: +{(player['top3']-0.30)*100:.1f}pp")
    logger.info(f"  Top-5 Accuracy: {player['top5']:.1%} (vs 50% baseline) - Edge: +{(player['top5']-0.50)*100:.1f}pp")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
