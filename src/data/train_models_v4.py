"""
Model Training V4 - With NBA API Player Stats

Uses enhanced features including:
- PPG, FG%, 3P%, usage rate from NBA API
- Historical first scorer rates
- Recent form
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JumpBallModelV4:
    """Jump ball prediction model."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for jump ball prediction."""
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
            # New: API stats for JB players
            df['home_jb_ppg'].values,
            df['away_jb_ppg'].values,
            df['home_jb_fg_pct'].values,
            df['away_jb_fg_pct'].values,
        ]
        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Jump Ball Model V4...")

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


class PlayerFirstScorerModelV4:
    """Enhanced player first scorer model with API stats."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix with API stats."""
        features = []

        # Per-player features (10 starters)
        for prefix in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                      'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            # Historical first scorer rate
            features.append(df[f'{prefix}_fs_rate'].values)
            # Recent form
            features.append(df[f'{prefix}_recent_fs_rate'].values)
            # Experience
            features.append(np.log1p(df[f'{prefix}_games'].values))
            # Is JB participant
            features.append(df[f'{prefix}_is_jb'].values.astype(float))
            # NEW: API stats
            features.append(df[f'{prefix}_ppg'].values)
            features.append(df[f'{prefix}_fg_pct'].values)
            features.append(df[f'{prefix}_fg3_pct'].values)
            features.append(df[f'{prefix}_usage'].values)

        # Team aggregates
        features.append(df['home_jb_predicted_prob'].values)
        features.append(df['home_total_fs_rate'].values)
        features.append(df['away_total_fs_rate'].values)
        features.append(df['home_total_ppg'].values)
        features.append(df['away_total_ppg'].values)
        features.append(df['home_avg_fg_pct'].values)
        features.append(df['away_avg_fg_pct'].values)
        features.append(df['home_max_ppg'].values)
        features.append(df['away_max_ppg'].values)

        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Player First Scorer Model V4...")

        train_valid = train_df[train_df['first_scorer_position'] >= 0].copy()

        X_train = self.scaler.fit_transform(self.get_features(train_valid))
        y_train = train_valid['first_scorer_position'].values

        # Use simpler model to reduce overfitting
        self.model = LogisticRegression(
            C=0.5,  # Moderate regularization
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred_proba)

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


class ModelTrainerV4:
    """Train and evaluate V4 models."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.jb_model = None
        self.player_model = None

    def load_data(self):
        """Load V4 feature data."""
        train = pd.read_parquet(self.data_dir / "train_v4.parquet")
        val = pd.read_parquet(self.data_dir / "val_v4.parquet")
        test = pd.read_parquet(self.data_dir / "test_v4.parquet")
        logger.info(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def train_all(self, train_df, val_df):
        """Train all models."""
        logger.info("=" * 60)
        logger.info("TRAINING V4 MODELS (with NBA API stats)")
        logger.info("=" * 60)

        self.jb_model = JumpBallModelV4()
        self.jb_model.fit(train_df, val_df)

        self.player_model = PlayerFirstScorerModelV4()
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

        return results

    def save_models(self):
        """Save trained models."""
        joblib.dump(self.jb_model, self.model_dir / "jump_ball_model_v4.joblib")
        joblib.dump(self.player_model, self.model_dir / "player_first_scorer_model_v4.joblib")
        logger.info(f"\nModels saved to {self.model_dir}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Model Training V4")
    logger.info("=" * 60)

    trainer = ModelTrainerV4()
    train_df, val_df, test_df = trainer.load_data()

    trainer.train_all(train_df, val_df)
    results = trainer.evaluate(test_df)
    trainer.save_models()

    # Summary with comparison to V3
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY - V4 vs V3 Comparison")
    logger.info("=" * 60)

    jb = results['jump_ball']
    player = results['player']

    logger.info(f"\nJump Ball Prediction:")
    logger.info(f"  V4 Accuracy: {jb['accuracy']:.1%} (vs {jb['baseline']:.1%} baseline)")
    logger.info(f"  Edge: +{(jb['accuracy'] - jb['baseline'])*100:.1f} pp")

    logger.info(f"\nPlayer First Scorer Prediction:")
    logger.info(f"  Top-1: {player['accuracy']:.1%} (vs 10% random) - Edge: +{(player['accuracy']-0.10)*100:.1f}pp")
    logger.info(f"  Top-3: {player['top3']:.1%} (vs 30% random) - Edge: +{(player['top3']-0.30)*100:.1f}pp")
    logger.info(f"  Top-5: {player['top5']:.1%} (vs 50% random) - Edge: +{(player['top5']-0.50)*100:.1f}pp")

    # Compare to V3 baselines
    logger.info("\nImprovement over V3:")
    logger.info("  V3 Jump Ball: 64.4% -> V4 Jump Ball: {:.1%}".format(jb['accuracy']))
    logger.info("  V3 Top-3: 33.8% -> V4 Top-3: {:.1%}".format(player['top3']))
    logger.info("  V3 Top-5: 54.2% -> V4 Top-5: {:.1%}".format(player['top5']))

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
