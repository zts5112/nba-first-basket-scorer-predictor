"""
Betting Strategy Module for NBA First Basket Scorer

This module provides tools for:
1. Converting model probabilities to betting decisions
2. Kelly criterion for optimal bet sizing
3. Expected value (EV) calculations
4. Backtesting on historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BettingOdds:
    """Represents betting odds for a player."""
    player_id: str
    player_name: str
    american_odds: int  # e.g., +450, -110
    decimal_odds: float  # e.g., 5.50, 1.91
    implied_prob: float  # e.g., 0.18 (18%)

    @classmethod
    def from_american(cls, player_id: str, player_name: str, american_odds: int) -> 'BettingOdds':
        """Create from American odds."""
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        implied_prob = 1 / decimal_odds
        return cls(player_id, player_name, american_odds, decimal_odds, implied_prob)

    @classmethod
    def from_decimal(cls, player_id: str, player_name: str, decimal_odds: float) -> 'BettingOdds':
        """Create from decimal odds."""
        if decimal_odds >= 2.0:
            american_odds = int((decimal_odds - 1) * 100)
        else:
            american_odds = int(-100 / (decimal_odds - 1))

        implied_prob = 1 / decimal_odds
        return cls(player_id, player_name, american_odds, decimal_odds, implied_prob)


@dataclass
class BetRecommendation:
    """A recommended bet with sizing."""
    player_id: str
    player_name: str
    model_prob: float       # Our model's probability
    implied_prob: float     # Sportsbook's implied probability
    edge: float             # model_prob - implied_prob
    decimal_odds: float
    kelly_fraction: float   # Optimal bet as fraction of bankroll
    recommended_bet: float  # Actual recommended bet (with Kelly fraction applied)
    expected_value: float   # EV per dollar bet

    def __str__(self):
        return (
            f"{self.player_name}: {self.model_prob:.1%} model vs {self.implied_prob:.1%} implied | "
            f"Edge: {self.edge:+.1%} | EV: ${self.expected_value:.3f}/$ | "
            f"Kelly: {self.kelly_fraction:.1%}"
        )


class BettingStrategy:
    """
    Betting strategy optimizer using model predictions.

    Key concepts:
    1. Edge = Model Probability - Implied Probability
    2. Expected Value (EV) = (prob * payout) - (1 - prob)
    3. Kelly Criterion = edge / (odds - 1) for optimal growth
    """

    def __init__(
        self,
        min_edge: float = 0.02,        # Minimum edge to bet (2%)
        min_prob: float = 0.05,        # Minimum model probability (5%)
        kelly_fraction: float = 0.25,  # Fraction of Kelly to use (conservative)
        max_bet_pct: float = 0.05,     # Max bet as % of bankroll
        bankroll: float = 1000.0       # Starting bankroll
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.bankroll = bankroll

    def calculate_edge(self, model_prob: float, implied_prob: float) -> float:
        """Calculate betting edge."""
        return model_prob - implied_prob

    def calculate_ev(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calculate expected value per dollar bet.

        EV = (prob * win_amount) - (1 - prob) * stake
        For $1 bet: EV = prob * (odds - 1) - (1 - prob)
        """
        win_amount = decimal_odds - 1
        ev = (model_prob * win_amount) - (1 - model_prob)
        return ev

    def kelly_criterion(self, model_prob: float, decimal_odds: float) -> float:
        """
        Calculate Kelly criterion bet fraction.

        Kelly = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = net odds (decimal_odds - 1)

        Returns the fraction of bankroll to bet for optimal growth.
        """
        if decimal_odds <= 1:
            return 0.0

        p = model_prob
        q = 1 - p
        b = decimal_odds - 1

        kelly = (p * b - q) / b

        # Kelly can be negative if EV is negative
        return max(0, kelly)

    def analyze_bet(
        self,
        model_prob: float,
        odds: BettingOdds
    ) -> Optional[BetRecommendation]:
        """
        Analyze a potential bet and return recommendation.

        Returns None if bet doesn't meet minimum criteria.
        """
        edge = self.calculate_edge(model_prob, odds.implied_prob)
        ev = self.calculate_ev(model_prob, odds.decimal_odds)
        kelly = self.kelly_criterion(model_prob, odds.decimal_odds)

        # Apply fractional Kelly for safety
        adjusted_kelly = kelly * self.kelly_fraction

        # Cap at max bet percentage
        adjusted_kelly = min(adjusted_kelly, self.max_bet_pct)

        # Check minimum criteria
        if edge < self.min_edge:
            return None
        if model_prob < self.min_prob:
            return None
        if ev <= 0:
            return None

        recommended_bet = adjusted_kelly * self.bankroll

        return BetRecommendation(
            player_id=odds.player_id,
            player_name=odds.player_name,
            model_prob=model_prob,
            implied_prob=odds.implied_prob,
            edge=edge,
            decimal_odds=odds.decimal_odds,
            kelly_fraction=adjusted_kelly,
            recommended_bet=recommended_bet,
            expected_value=ev
        )

    def find_value_bets(
        self,
        model_probs: Dict[str, float],  # player_id -> probability
        market_odds: List[BettingOdds]
    ) -> List[BetRecommendation]:
        """
        Find all value bets from model predictions vs market odds.

        Returns list of recommendations sorted by expected value.
        """
        recommendations = []

        for odds in market_odds:
            if odds.player_id not in model_probs:
                continue

            model_prob = model_probs[odds.player_id]
            rec = self.analyze_bet(model_prob, odds)

            if rec is not None:
                recommendations.append(rec)

        # Sort by EV (best first)
        recommendations.sort(key=lambda x: -x.expected_value)

        return recommendations

    def simulate_bet(
        self,
        bet: BetRecommendation,
        won: bool
    ) -> Tuple[float, float]:
        """
        Simulate a bet outcome.

        Returns (profit, new_bankroll)
        """
        if won:
            profit = bet.recommended_bet * (bet.decimal_odds - 1)
        else:
            profit = -bet.recommended_bet

        self.bankroll += profit
        return profit, self.bankroll


class BettingSimulator:
    """
    Backtesting simulator for betting strategies.
    """

    def __init__(
        self,
        strategy: BettingStrategy,
        initial_bankroll: float = 1000.0
    ):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.strategy.bankroll = initial_bankroll

        # Track results
        self.bets: List[Dict] = []
        self.bankroll_history: List[float] = [initial_bankroll]

    def simulate_game(
        self,
        game_id: str,
        model_probs: Dict[str, float],
        market_odds: List[BettingOdds],
        actual_first_scorer: str
    ) -> List[Dict]:
        """
        Simulate betting on a single game.

        Args:
            game_id: Unique game identifier
            model_probs: Dict of player_id -> model probability
            market_odds: List of betting odds for each player
            actual_first_scorer: player_id of actual first scorer

        Returns:
            List of bet results
        """
        # Find value bets
        recommendations = self.strategy.find_value_bets(model_probs, market_odds)

        results = []
        for rec in recommendations:
            won = (rec.player_id == actual_first_scorer)
            profit, new_bankroll = self.strategy.simulate_bet(rec, won)

            result = {
                'game_id': game_id,
                'player_id': rec.player_id,
                'player_name': rec.player_name,
                'model_prob': rec.model_prob,
                'implied_prob': rec.implied_prob,
                'edge': rec.edge,
                'decimal_odds': rec.decimal_odds,
                'bet_amount': rec.recommended_bet,
                'won': won,
                'profit': profit,
                'bankroll': new_bankroll,
                'ev': rec.expected_value
            }
            results.append(result)
            self.bets.append(result)

        self.bankroll_history.append(self.strategy.bankroll)
        return results

    def get_summary(self) -> Dict:
        """Get summary statistics of simulation."""
        if not self.bets:
            return {'total_bets': 0}

        df = pd.DataFrame(self.bets)

        total_bets = len(df)
        wins = df['won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets > 0 else 0

        total_wagered = df['bet_amount'].sum()
        total_profit = df['profit'].sum()
        roi = total_profit / total_wagered if total_wagered > 0 else 0

        avg_edge = df['edge'].mean()
        avg_ev = df['ev'].mean()

        final_bankroll = self.strategy.bankroll
        total_return = (final_bankroll - self.initial_bankroll) / self.initial_bankroll

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'avg_edge': avg_edge,
            'avg_ev': avg_ev,
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_return': total_return
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("BETTING SIMULATION SUMMARY")
        print("=" * 60)

        if summary['total_bets'] == 0:
            print("No bets placed.")
            return

        print(f"\nBet Statistics:")
        print(f"  Total Bets: {summary['total_bets']}")
        print(f"  Wins: {summary['wins']} | Losses: {summary['losses']}")
        print(f"  Win Rate: {summary['win_rate']:.1%}")

        print(f"\nFinancial Results:")
        print(f"  Total Wagered: ${summary['total_wagered']:.2f}")
        print(f"  Total Profit: ${summary['total_profit']:+.2f}")
        print(f"  ROI: {summary['roi']:+.1%}")

        print(f"\nEdge Analysis:")
        print(f"  Avg Edge: {summary['avg_edge']:+.1%}")
        print(f"  Avg EV: ${summary['avg_ev']:.3f} per $1 bet")

        print(f"\nBankroll:")
        print(f"  Initial: ${summary['initial_bankroll']:.2f}")
        print(f"  Final: ${summary['final_bankroll']:.2f}")
        print(f"  Return: {summary['total_return']:+.1%}")


def generate_synthetic_odds(
    player_probs: Dict[str, float],
    player_names: Dict[str, str],
    vig: float = 0.10  # 10% vig/juice
) -> List[BettingOdds]:
    """
    Generate synthetic market odds from true probabilities.

    Adds vig to simulate sportsbook margins.
    """
    odds_list = []

    # Total probability with vig
    total_prob = sum(player_probs.values())
    vig_multiplier = (1 + vig)

    for player_id, true_prob in player_probs.items():
        # Sportsbook implies higher probability (vig)
        implied_prob = true_prob * vig_multiplier / total_prob
        implied_prob = min(implied_prob, 0.95)  # Cap at 95%

        decimal_odds = 1 / implied_prob

        odds = BettingOdds.from_decimal(
            player_id=player_id,
            player_name=player_names.get(player_id, player_id),
            decimal_odds=decimal_odds
        )
        odds_list.append(odds)

    return odds_list


def demo_betting_strategy():
    """Demonstrate the betting strategy with sample data."""
    print("=" * 60)
    print("BETTING STRATEGY DEMO")
    print("=" * 60)

    # Sample model predictions (from our V4 model)
    model_probs = {
        'tatumja01': 0.174,   # Jayson Tatum
        'brownja02': 0.146,   # Jaylen Brown
        'horforal01': 0.134,  # Al Horford
        'jamesle01': 0.134,   # LeBron James
        'davisan02': 0.123,   # Anthony Davis
        'reaveau01': 0.091,   # Austin Reaves
        'russeda01': 0.054,   # D'Angelo Russell
        'whitede01': 0.052,   # Derrick White
        'hachiru01': 0.045,   # Rui Hachimura
        'holidjr01': 0.045,   # Jrue Holiday
    }

    player_names = {
        'tatumja01': 'Jayson Tatum',
        'brownja02': 'Jaylen Brown',
        'horforal01': 'Al Horford',
        'jamesle01': 'LeBron James',
        'davisan02': 'Anthony Davis',
        'reaveau01': 'Austin Reaves',
        'russeda01': "D'Angelo Russell",
        'whitede01': 'Derrick White',
        'hachiru01': 'Rui Hachimura',
        'holidjr01': 'Jrue Holiday',
    }

    # Sample market odds (typical first basket scorer lines)
    market_odds = [
        BettingOdds.from_american('tatumja01', 'Jayson Tatum', +500),
        BettingOdds.from_american('brownja02', 'Jaylen Brown', +550),
        BettingOdds.from_american('horforal01', 'Al Horford', +900),
        BettingOdds.from_american('jamesle01', 'LeBron James', +550),
        BettingOdds.from_american('davisan02', 'Anthony Davis', +600),
        BettingOdds.from_american('reaveau01', 'Austin Reaves', +1000),
        BettingOdds.from_american('russeda01', "D'Angelo Russell", +1200),
        BettingOdds.from_american('whitede01', 'Derrick White', +1400),
        BettingOdds.from_american('hachiru01', 'Rui Hachimura', +1600),
        BettingOdds.from_american('holidjr01', 'Jrue Holiday', +1800),
    ]

    print("\n1. MARKET ODDS vs MODEL PREDICTIONS")
    print("-" * 60)
    print(f"{'Player':<20} {'Market':<10} {'Implied':<10} {'Model':<10} {'Edge':<10}")
    print("-" * 60)

    for odds in market_odds:
        model_prob = model_probs.get(odds.player_id, 0)
        edge = model_prob - odds.implied_prob
        print(f"{odds.player_name:<20} {odds.american_odds:>+5}     "
              f"{odds.implied_prob:>6.1%}     {model_prob:>6.1%}     {edge:>+6.1%}")

    # Create strategy
    strategy = BettingStrategy(
        min_edge=0.02,        # 2% minimum edge
        kelly_fraction=0.25,  # Quarter Kelly
        max_bet_pct=0.05,     # 5% max bet
        bankroll=1000.0
    )

    print("\n2. VALUE BETS FOUND")
    print("-" * 60)

    recommendations = strategy.find_value_bets(model_probs, market_odds)

    if not recommendations:
        print("No value bets found with current criteria.")
    else:
        for rec in recommendations:
            print(f"\n{rec.player_name}")
            print(f"  Model: {rec.model_prob:.1%} vs Market: {rec.implied_prob:.1%}")
            print(f"  Edge: {rec.edge:+.1%} | EV: ${rec.expected_value:.3f} per $1")
            print(f"  Kelly: {rec.kelly_fraction:.2%} of bankroll")
            print(f"  Recommended Bet: ${rec.recommended_bet:.2f}")

    print("\n3. STRATEGY PARAMETERS")
    print("-" * 60)
    print(f"  Minimum Edge: {strategy.min_edge:.0%}")
    print(f"  Minimum Probability: {strategy.min_prob:.0%}")
    print(f"  Kelly Fraction: {strategy.kelly_fraction:.0%} (quarter Kelly)")
    print(f"  Max Bet: {strategy.max_bet_pct:.0%} of bankroll")
    print(f"  Bankroll: ${strategy.bankroll:.2f}")

    return recommendations


if __name__ == "__main__":
    demo_betting_strategy()
