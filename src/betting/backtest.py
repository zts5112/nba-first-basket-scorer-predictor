"""
Historical Backtesting for Betting Strategy

Uses our trained V4 models to simulate betting on historical games.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

from src.betting.strategy import (
    BettingStrategy, BettingSimulator, BettingOdds,
    generate_synthetic_odds
)
from src.data.train_models_v4 import JumpBallModelV4, PlayerFirstScorerModelV4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalBacktester:
    """
    Backtest betting strategy on historical games using trained models.
    """

    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "data/processed",
        vig: float = 0.08  # 8% vig (realistic for props)
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.vig = vig

        # Load models
        self.jb_model = joblib.load(self.model_dir / "jump_ball_model_v4.joblib")
        self.player_model = joblib.load(self.model_dir / "player_first_scorer_model_v4.joblib")
        logger.info("V4 models loaded")

        # Load player tokenizer for ID->name mapping
        with open(self.data_dir / "player_tokenizer.json") as f:
            player_data = json.load(f)
        self.token_to_player = {v: k for k, v in player_data['player_to_token'].items()}
        self.player_id_to_name = player_data['player_id_to_name']

    def load_test_data(self) -> pd.DataFrame:
        """Load test set for backtesting."""
        test_df = pd.read_parquet(self.data_dir / "test_v4.parquet")
        logger.info(f"Loaded {len(test_df)} test games")
        return test_df

    def get_player_id_from_token(self, token: int) -> str:
        """Convert token back to player ID."""
        return self.token_to_player.get(token, f"UNK_{token}")

    def get_player_name(self, player_id: str) -> str:
        """Get player name from ID."""
        return self.player_id_to_name.get(player_id, player_id)

    def simulate_market_odds(
        self,
        model_probs: np.ndarray,
        player_ids: List[str],
        market_efficiency: float = 0.7
    ) -> List[BettingOdds]:
        """
        Generate realistic market odds with inefficiency.

        The market uses a blend of:
        - "Naive" probability (uniform + some noise towards favorites)
        - Partially informed probability

        market_efficiency: 0.0 = pure noise, 1.0 = perfectly efficient
        """
        odds_list = []
        n_players = len(model_probs)

        if n_players == 0:
            return []

        # Create "naive market" view - favors players with higher indices
        # (stars tend to be earlier in lineup) plus uniform noise
        naive_probs = np.array([0.15, 0.14, 0.11, 0.08, 0.07,  # Home
                                 0.14, 0.13, 0.08, 0.06, 0.04])  # Away

        # Add some random noise
        noise = np.random.normal(0, 0.02, n_players)
        naive_probs = naive_probs + noise
        naive_probs = np.maximum(naive_probs, 0.02)
        naive_probs = naive_probs / naive_probs.sum()

        # Blend naive with model probs based on efficiency
        # Higher efficiency = market is closer to true probs
        market_probs = market_efficiency * model_probs + (1 - market_efficiency) * naive_probs
        market_probs = market_probs / market_probs.sum()

        for i, (prob, pid) in enumerate(zip(market_probs, player_ids)):
            if prob <= 0.01:
                continue

            # Apply vig (sportsbook's edge)
            implied_prob = prob * (1 + self.vig)
            implied_prob = min(implied_prob, 0.95)

            decimal_odds = 1 / implied_prob

            odds = BettingOdds.from_decimal(
                player_id=pid,
                player_name=self.get_player_name(pid),
                decimal_odds=decimal_odds
            )
            odds_list.append(odds)

        return odds_list

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        strategy: BettingStrategy,
        use_random_odds: bool = False
    ) -> BettingSimulator:
        """
        Run backtest on test data.

        Args:
            test_df: Test dataset with features and outcomes
            strategy: Betting strategy to use
            use_random_odds: If True, generate random odds (baseline comparison)

        Returns:
            BettingSimulator with results
        """
        # Create fresh strategy with same parameters but reset bankroll
        fresh_strategy = BettingStrategy(
            min_edge=strategy.min_edge,
            min_prob=strategy.min_prob,
            kelly_fraction=strategy.kelly_fraction,
            max_bet_pct=strategy.max_bet_pct,
            bankroll=strategy.bankroll
        )
        simulator = BettingSimulator(fresh_strategy, fresh_strategy.bankroll)

        # Filter to games with valid first scorer
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        logger.info(f"Backtesting on {len(valid_games)} games with valid first scorer")

        for idx, row in valid_games.iterrows():
            game_id = row['game_id']

            # Get model predictions
            game_df = pd.DataFrame([row])
            player_probs = self.player_model.predict_proba(game_df)[0]

            # Get player IDs for the 10 starters
            player_ids = []
            for i in range(5):
                home_token = row[f'home_{i}_token']
                player_ids.append(self.get_player_id_from_token(home_token))
            for i in range(5):
                away_token = row[f'away_{i}_token']
                player_ids.append(self.get_player_id_from_token(away_token))

            # Get actual first scorer
            first_scorer_pos = int(row['first_scorer_position'])
            actual_first_scorer = player_ids[first_scorer_pos] if 0 <= first_scorer_pos < 10 else None

            if actual_first_scorer is None:
                continue

            # Generate market odds
            if use_random_odds:
                # Baseline: use model probs as "true" but compare to random guessing
                # This simulates having no model edge
                random_model_probs = np.random.dirichlet(np.ones(10))
                market_odds = self.simulate_market_odds(player_probs, player_ids)
                model_probs_dict = {pid: prob for pid, prob in zip(player_ids, random_model_probs)}
            else:
                # Normal: our model vs market
                market_odds = self.simulate_market_odds(player_probs, player_ids)
                model_probs_dict = {pid: prob for pid, prob in zip(player_ids, player_probs)}

            # Simulate betting
            simulator.simulate_game(
                game_id=game_id,
                model_probs=model_probs_dict,
                market_odds=market_odds,
                actual_first_scorer=actual_first_scorer
            )

        return simulator

    def run_edge_analysis(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze edge by comparing model predictions to actual outcomes.
        """
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()

        results = []
        for idx, row in valid_games.iterrows():
            game_df = pd.DataFrame([row])
            player_probs = self.player_model.predict_proba(game_df)[0]

            first_scorer_pos = int(row['first_scorer_position'])

            # Get probability assigned to actual first scorer
            if 0 <= first_scorer_pos < 10:
                prob_on_winner = player_probs[first_scorer_pos]
                max_prob = max(player_probs)
                predicted_pos = np.argmax(player_probs)
                correct = (predicted_pos == first_scorer_pos)

                # Top-N accuracy
                top3 = first_scorer_pos in np.argsort(player_probs)[-3:]
                top5 = first_scorer_pos in np.argsort(player_probs)[-5:]

                results.append({
                    'game_id': row['game_id'],
                    'prob_on_winner': prob_on_winner,
                    'max_prob': max_prob,
                    'correct': correct,
                    'top3': top3,
                    'top5': top5,
                    'first_scorer_pos': first_scorer_pos,
                    'predicted_pos': predicted_pos
                })

        results_df = pd.DataFrame(results)

        logger.info("\n=== EDGE ANALYSIS ===")
        logger.info(f"Games analyzed: {len(results_df)}")
        logger.info(f"Top-1 Accuracy: {results_df['correct'].mean():.1%}")
        logger.info(f"Top-3 Accuracy: {results_df['top3'].mean():.1%}")
        logger.info(f"Top-5 Accuracy: {results_df['top5'].mean():.1%}")
        logger.info(f"Avg probability on winner: {results_df['prob_on_winner'].mean():.1%}")
        logger.info(f"Baseline (random): 10%")

        return results_df


def main():
    """Run historical backtest."""
    logger.info("=" * 60)
    logger.info("HISTORICAL BETTING BACKTEST")
    logger.info("=" * 60)

    backtester = HistoricalBacktester(vig=0.08)
    test_df = backtester.load_test_data()

    # Run edge analysis first
    edge_df = backtester.run_edge_analysis(test_df)

    print("\n" + "=" * 60)
    print("BETTING STRATEGY BACKTEST")
    print("=" * 60)
    print("\nAssumptions:")
    print("  - 8% vig (sportsbook margin)")
    print("  - Market efficiency: 70% (market captures 70% of true info)")
    print("  - 720 test games from 2025-26 season")

    # Different strategy configurations
    strategies = [
        ("Conservative", BettingStrategy(min_edge=0.04, kelly_fraction=0.15, max_bet_pct=0.03, bankroll=1000)),
        ("Moderate", BettingStrategy(min_edge=0.03, kelly_fraction=0.20, max_bet_pct=0.04, bankroll=1000)),
        ("Aggressive", BettingStrategy(min_edge=0.02, kelly_fraction=0.25, max_bet_pct=0.05, bankroll=1000)),
    ]

    print("\n" + "-" * 60)
    print("RESULTS BY STRATEGY")
    print("-" * 60)

    for name, strategy in strategies:
        print(f"\n{name} Strategy (min edge: {strategy.min_edge:.0%}, Kelly: {strategy.kelly_fraction:.0%})")

        simulator = backtester.run_backtest(test_df, strategy)
        summary = simulator.get_summary()

        if summary['total_bets'] > 0:
            print(f"  Bets Placed: {summary['total_bets']}")
            print(f"  Win Rate: {summary['win_rate']:.1%} (expected: ~11%)")
            print(f"  Total Wagered: ${summary['total_wagered']:.2f}")
            print(f"  Net Profit: ${summary['total_profit']:+.2f}")
            print(f"  ROI: {summary['roi']:+.1%}")
            print(f"  Bankroll: ${summary['initial_bankroll']:.0f} -> ${summary['final_bankroll']:.2f}")
        else:
            print("  No bets placed with these criteria")

    # Flat betting comparison
    print("\n" + "-" * 60)
    print("FLAT BETTING COMPARISON")
    print("-" * 60)
    print("\nWhat if we flat bet $10 on our top pick every game?")

    flat_bet = 10.0
    wins = 0
    profit = 0.0

    valid_games = test_df[test_df['first_scorer_position'] >= 0]
    for idx, row in valid_games.iterrows():
        game_df = pd.DataFrame([row])
        player_probs = backtester.player_model.predict_proba(game_df)[0]
        predicted_pos = np.argmax(player_probs)
        actual_pos = int(row['first_scorer_position'])

        # Assume average odds of +800 for top pick (typical first scorer line)
        decimal_odds = 9.0

        if predicted_pos == actual_pos:
            wins += 1
            profit += flat_bet * (decimal_odds - 1)
        else:
            profit -= flat_bet

    n_games = len(valid_games)
    total_wagered = n_games * flat_bet
    roi = profit / total_wagered if total_wagered > 0 else 0

    print(f"  Games: {n_games}")
    print(f"  Wins: {wins} ({wins/n_games:.1%})")
    print(f"  Total Wagered: ${total_wagered:.0f}")
    print(f"  Net Profit: ${profit:+.2f}")
    print(f"  ROI: {roi:+.1%}")

    return backtester, edge_df


if __name__ == "__main__":
    backtester, edge_df = main()
