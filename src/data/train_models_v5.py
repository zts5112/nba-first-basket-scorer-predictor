"""
Model Training V5 - XGBoost for Player First Scorer

Upgrades from V4:
- XGBoost instead of LogisticRegression for player prediction
- XGBoost for jump ball prediction
- Hyperparameter tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JumpBallModelV5:
    """Jump ball prediction model using XGBoost."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for jump ball prediction."""
        self.feature_names = [
            'home_jb_win_rate', 'away_jb_win_rate', 'jb_win_rate_diff',
            'log_home_jb_total', 'log_away_jb_total',
            'home_jb_h2h_rate', 'log_h2h_matchups',
            'home_jb_predicted_prob',
            'home_tip_win_rate', 'away_tip_win_rate',
            'home_jb_ppg', 'away_jb_ppg',
            'home_jb_fg_pct', 'away_jb_fg_pct',
            # New features
            'ppg_diff', 'fg_pct_diff',
        ]

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
            df['home_jb_ppg'].values,
            df['away_jb_ppg'].values,
            df['home_jb_fg_pct'].values,
            df['away_jb_fg_pct'].values,
            # Interaction features
            df['home_jb_ppg'].values - df['away_jb_ppg'].values,
            df['home_jb_fg_pct'].values - df['away_jb_fg_pct'].values,
        ]
        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Jump Ball Model V5 (XGBoost)...")

        X_train = self.get_features(train_df)
        y_train = train_df['home_won_tip'].values

        X_train_scaled = self.scaler.fit_transform(X_train)

        if HAS_XGBOOST:
            # Tuned to reduce overfitting
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                min_child_weight=30,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=3.0,
                gamma=0.5,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=20, random_state=42
            )

        self.model.fit(X_train_scaled, y_train)

        baseline = y_train.mean()
        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
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


class PlayerFirstScorerModelV5:
    """Player first scorer model using XGBoost multiclass."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix with API stats."""
        features = []
        feature_names = []

        # Per-player features (10 starters)
        for prefix in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                      'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            # Historical first scorer rate
            features.append(df[f'{prefix}_fs_rate'].values)
            feature_names.append(f'{prefix}_fs_rate')

            # Recent form
            features.append(df[f'{prefix}_recent_fs_rate'].values)
            feature_names.append(f'{prefix}_recent_fs_rate')

            # Experience (log transformed)
            features.append(np.log1p(df[f'{prefix}_games'].values))
            feature_names.append(f'{prefix}_log_games')

            # Is JB participant
            features.append(df[f'{prefix}_is_jb'].values.astype(float))
            feature_names.append(f'{prefix}_is_jb')

            # API stats
            features.append(df[f'{prefix}_ppg'].values)
            feature_names.append(f'{prefix}_ppg')

            features.append(df[f'{prefix}_fg_pct'].values)
            feature_names.append(f'{prefix}_fg_pct')

            features.append(df[f'{prefix}_fg3_pct'].values)
            feature_names.append(f'{prefix}_fg3_pct')

            features.append(df[f'{prefix}_usage'].values)
            feature_names.append(f'{prefix}_usage')

            # NEW: Composite scoring potential
            ppg = df[f'{prefix}_ppg'].values
            fg_pct = df[f'{prefix}_fg_pct'].values
            usage = df[f'{prefix}_usage'].values
            features.append(ppg * fg_pct / 100)  # Points weighted by efficiency
            feature_names.append(f'{prefix}_ppg_x_fgpct')

            features.append(ppg * usage / 100)  # Points weighted by usage
            feature_names.append(f'{prefix}_ppg_x_usage')

        # Team aggregates
        features.append(df['home_jb_predicted_prob'].values)
        feature_names.append('home_jb_predicted_prob')

        features.append(df['home_total_fs_rate'].values)
        feature_names.append('home_total_fs_rate')

        features.append(df['away_total_fs_rate'].values)
        feature_names.append('away_total_fs_rate')

        features.append(df['home_total_ppg'].values)
        feature_names.append('home_total_ppg')

        features.append(df['away_total_ppg'].values)
        feature_names.append('away_total_ppg')

        features.append(df['home_avg_fg_pct'].values)
        feature_names.append('home_avg_fg_pct')

        features.append(df['away_avg_fg_pct'].values)
        feature_names.append('away_avg_fg_pct')

        features.append(df['home_max_ppg'].values)
        feature_names.append('home_max_ppg')

        features.append(df['away_max_ppg'].values)
        feature_names.append('away_max_ppg')

        # NEW: Team-level aggregates
        features.append(df['home_total_ppg'].values - df['away_total_ppg'].values)
        feature_names.append('ppg_diff')

        features.append(df['home_avg_fg_pct'].values - df['away_avg_fg_pct'].values)
        feature_names.append('fg_pct_diff')

        self.feature_names = feature_names
        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the model."""
        logger.info("Training Player First Scorer Model V5 (XGBoost)...")

        train_valid = train_df[train_df['first_scorer_position'] >= 0].copy()

        X_train = self.get_features(train_valid)
        y_train = train_valid['first_scorer_position'].values

        X_train_scaled = self.scaler.fit_transform(X_train)

        if HAS_XGBOOST:
            # Tuned to reduce overfitting
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=3,           # Reduced from 5
                learning_rate=0.05,    # Increased from 0.03
                min_child_weight=50,   # Increased from 5 (more regularization)
                subsample=0.7,         # Reduced from 0.8
                colsample_bytree=0.6,  # Reduced from 0.8
                reg_alpha=2.0,         # Increased from 0.5
                reg_lambda=5.0,        # Increased from 2.0
                gamma=1.0,             # Added min loss reduction
                random_state=42,
                eval_metric='mlogloss',
                num_class=10,
                objective='multi:softprob'
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("Using sklearn GradientBoosting (XGBoost not available)")
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=10, random_state=42
            )

        self.model.fit(X_train_scaled, y_train)

        train_pred = self.model.predict(X_train_scaled)
        train_pred_proba = self.model.predict_proba(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_logloss = log_loss(y_train, train_pred_proba)

        top3 = sum(y in np.argsort(p)[-3:] for y, p in zip(y_train, train_pred_proba)) / len(y_train)
        top5 = sum(y in np.argsort(p)[-5:] for y, p in zip(y_train, train_pred_proba)) / len(y_train)

        logger.info(f"  Baseline: 0.1000 (random)")
        logger.info(f"  Train: Acc={train_acc:.4f}, Top3={top3:.4f}, Top5={top5:.4f}, LogLoss={train_logloss:.4f}")

        if val_df is not None:
            val_valid = val_df[val_df['first_scorer_position'] >= 0].copy()
            X_val = self.scaler.transform(self.get_features(val_valid))
            y_val = val_valid['first_scorer_position'].values

            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_logloss = log_loss(y_val, val_pred_proba)
            val_top3 = sum(y in np.argsort(p)[-3:] for y, p in zip(y_val, val_pred_proba)) / len(y_val)
            val_top5 = sum(y in np.argsort(p)[-5:] for y, p in zip(y_val, val_pred_proba)) / len(y_val)

            logger.info(f"  Val: Acc={val_acc:.4f}, Top3={val_top3:.4f}, Top5={val_top5:.4f}, LogLoss={val_logloss:.4f}")

        # Feature importance
        if HAS_XGBOOST and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_features = sorted(zip(self.feature_names, importances), key=lambda x: -x[1])[:15]
            logger.info("\n  Top 15 Features:")
            for name, imp in top_features:
                logger.info(f"    {name}: {imp:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self.get_features(df))
        return self.model.predict_proba(X)


class ModelTrainerV5:
    """Train and evaluate V5 models."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.jb_model = None
        self.player_model = None

    def load_data(self):
        """Load V4 feature data (same features, different model)."""
        train = pd.read_parquet(self.data_dir / "train_v4.parquet")
        val = pd.read_parquet(self.data_dir / "val_v4.parquet")
        test = pd.read_parquet(self.data_dir / "test_v4.parquet")
        logger.info(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def train_all(self, train_df, val_df):
        """Train all models."""
        logger.info("=" * 60)
        logger.info("TRAINING V5 MODELS (XGBoost)")
        logger.info("=" * 60)

        self.jb_model = JumpBallModelV5()
        self.jb_model.fit(train_df, val_df)

        self.player_model = PlayerFirstScorerModelV5()
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
        joblib.dump(self.jb_model, self.model_dir / "jump_ball_model_v5.joblib")
        joblib.dump(self.player_model, self.model_dir / "player_first_scorer_model_v5.joblib")
        logger.info(f"\nModels saved to {self.model_dir}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("NBA First Basket Scorer - Model Training V5 (XGBoost)")
    logger.info("=" * 60)

    if not HAS_XGBOOST:
        logger.error("XGBoost not installed! Run: pip install xgboost")
        return None, None

    trainer = ModelTrainerV5()
    train_df, val_df, test_df = trainer.load_data()

    trainer.train_all(train_df, val_df)
    results = trainer.evaluate(test_df)
    trainer.save_models()

    # Summary with comparison to V4
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY - V5 (XGBoost) vs V4 (LogReg) Comparison")
    logger.info("=" * 60)

    jb = results['jump_ball']
    player = results['player']

    logger.info(f"\nJump Ball Prediction:")
    logger.info(f"  V5 Accuracy: {jb['accuracy']:.1%} (vs {jb['baseline']:.1%} baseline)")
    logger.info(f"  V4 Accuracy: 64.7%")
    logger.info(f"  Change: {(jb['accuracy'] - 0.647)*100:+.1f} pp")

    logger.info(f"\nPlayer First Scorer Prediction:")
    logger.info(f"  Top-1: {player['accuracy']:.1%} (V4: 12.5%) - Change: {(player['accuracy']-0.125)*100:+.1f}pp")
    logger.info(f"  Top-3: {player['top3']:.1%} (V4: 34.3%) - Change: {(player['top3']-0.343)*100:+.1f}pp")
    logger.info(f"  Top-5: {player['top5']:.1%} (V4: 53.6%) - Change: {(player['top5']-0.536)*100:+.1f}pp")

    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
