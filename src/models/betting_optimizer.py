"""
Betting Optimization Module

Implements:
1. Kelly Criterion for optimal bet sizing
2. Multi-outcome bet optimization
3. Expected Value calculation
4. Bankroll management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, linprog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BettingLine:
    """Represents a betting line for a player scoring first."""
    player: str
    american_odds: int  # e.g., +450, -110
    decimal_odds: float = None  # Calculated from american
    implied_prob: float = None  # Bookmaker's implied probability
    
    def __post_init__(self):
        if self.decimal_odds is None:
            self.decimal_odds = self._american_to_decimal(self.american_odds)
        if self.implied_prob is None:
            self.implied_prob = 1 / self.decimal_odds
    
    @staticmethod
    def _american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))
    
    @staticmethod
    def _decimal_to_american(decimal: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))


@dataclass 
class BetOpportunity:
    """A potential bet with model probability and betting line."""
    player: str
    model_prob: float  # Our predicted probability
    line: BettingLine
    
    @property
    def expected_value(self) -> float:
        """Expected value per $1 bet."""
        # EV = (prob * payout) - (1-prob) * stake
        # For $1 stake: EV = prob * (decimal - 1) - (1 - prob)
        return self.model_prob * (self.line.decimal_odds - 1) - (1 - self.model_prob)
    
    @property
    def edge(self) -> float:
        """Our edge over the bookmaker."""
        return self.model_prob - self.line.implied_prob
    
    @property
    def kelly_fraction(self) -> float:
        """Full Kelly bet fraction."""
        # Kelly = (bp - q) / b
        # where b = decimal odds - 1, p = prob of winning, q = 1 - p
        b = self.line.decimal_odds - 1
        p = self.model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        return max(0, kelly)  # Never negative (no bet)


class KellyCriterion:
    """
    Kelly Criterion implementation for bet sizing.
    
    Full Kelly maximizes long-term growth but has high variance.
    Fractional Kelly (e.g., 0.25x) reduces variance at cost of growth rate.
    """
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Args:
            kelly_fraction: Multiplier for Kelly (0.25 = quarter Kelly, common conservative choice)
        """
        self.kelly_fraction = kelly_fraction
    
    def optimal_bet_size(
        self, 
        prob: float, 
        decimal_odds: float,
        bankroll: float
    ) -> float:
        """
        Calculate optimal bet size for a single bet.
        
        Args:
            prob: Our probability estimate of winning
            decimal_odds: Decimal odds offered
            bankroll: Current bankroll
            
        Returns:
            Optimal bet amount in dollars
        """
        b = decimal_odds - 1
        p = prob
        q = 1 - p
        
        # Full Kelly
        full_kelly = (b * p - q) / b
        
        if full_kelly <= 0:
            return 0  # No edge, no bet
        
        # Apply fraction and bankroll
        bet_size = bankroll * full_kelly * self.kelly_fraction
        
        return max(0, bet_size)
    
    def size_multiple_bets(
        self,
        opportunities: List[BetOpportunity],
        bankroll: float,
        max_total_bet: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Size bets for multiple opportunities.
        
        Note: For correlated bets (like first scorer where only one wins),
        simple Kelly doesn't apply directly. This uses independent Kelly
        then scales down - more sophisticated methods exist.
        """
        bet_sizes = {}
        
        for opp in opportunities:
            if opp.expected_value > 0:
                size = self.optimal_bet_size(
                    opp.model_prob, 
                    opp.line.decimal_odds,
                    bankroll
                )
                bet_sizes[opp.player] = size
        
        # Scale down if total exceeds limit or bankroll
        total_bet = sum(bet_sizes.values())
        max_allowed = max_total_bet or bankroll * 0.3  # Default max 30% of bankroll
        
        if total_bet > max_allowed:
            scale = max_allowed / total_bet
            bet_sizes = {k: v * scale for k, v in bet_sizes.items()}
        
        return bet_sizes


class MultiOutcomeBetOptimizer:
    """
    Optimizer for mutually exclusive outcomes (only one player scores first).
    
    This is more sophisticated than independent Kelly because the bets
    are correlated - if Player A scores first, all other bets lose.
    """
    
    def __init__(self, max_bet_fraction: float = 0.30, min_edge: float = 0.02):
        """
        Args:
            max_bet_fraction: Maximum fraction of bankroll to bet total
            min_edge: Minimum edge required to bet (default 2%)
        """
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
    
    def find_value_bets(
        self,
        model_probs: Dict[str, float],
        betting_lines: Dict[str, BettingLine]
    ) -> List[BetOpportunity]:
        """
        Find bets with positive expected value.
        
        Args:
            model_probs: Dict of player -> our probability
            betting_lines: Dict of player -> BettingLine
            
        Returns:
            List of BetOpportunity with positive EV
        """
        opportunities = []
        
        for player, prob in model_probs.items():
            if player in betting_lines:
                line = betting_lines[player]
                opp = BetOpportunity(player, prob, line)
                
                if opp.edge >= self.min_edge and opp.expected_value > 0:
                    opportunities.append(opp)
        
        return sorted(opportunities, key=lambda x: x.expected_value, reverse=True)
    
    def optimize_bet_allocation(
        self,
        opportunities: List[BetOpportunity],
        bankroll: float
    ) -> Dict[str, float]:
        """
        Optimize bet allocation across mutually exclusive outcomes.
        
        Uses constrained optimization to maximize expected log growth
        (equivalent to Kelly for multiple correlated bets).
        
        This accounts for the fact that only one bet can win.
        """
        if not opportunities:
            return {}
        
        n = len(opportunities)
        
        # Get probabilities and payouts
        probs = np.array([opp.model_prob for opp in opportunities])
        payouts = np.array([opp.line.decimal_odds for opp in opportunities])
        
        # Probability that none of our bets win
        prob_none = 1 - probs.sum()
        
        def neg_expected_log_growth(bet_fracs):
            """
            Negative of expected log growth (we minimize this).
            
            Growth = sum(prob_i * log(1 + bet_i * (payout_i - 1) - sum(other bets)))
                   + prob_none * log(1 - sum(bets))
            
            Simplified: assume we lose all other bets when one wins.
            """
            total_bet = bet_fracs.sum()
            
            if total_bet >= 1:
                return 1e10  # Invalid
            
            growth = 0
            
            # For each possible winner
            for i in range(n):
                # If player i wins: gain payout, lose all other bets
                profit = bet_fracs[i] * (payouts[i] - 1) - (total_bet - bet_fracs[i])
                if 1 + profit > 0:
                    growth += probs[i] * np.log(1 + profit)
                else:
                    return 1e10  # Would go bust
            
            # If none of our picks win
            if prob_none > 0:
                if 1 - total_bet > 0:
                    growth += prob_none * np.log(1 - total_bet)
                else:
                    return 1e10
            
            return -growth  # Negative because we minimize
        
        # Initial guess: proportional to edge
        edges = np.array([opp.edge for opp in opportunities])
        edges = np.maximum(edges, 0)
        if edges.sum() > 0:
            x0 = edges / edges.sum() * 0.1  # Start conservative
        else:
            x0 = np.ones(n) * 0.02
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.max_bet_fraction - x.sum()},  # Total bet limit
        ]
        
        # Bounds: each bet between 0 and max_fraction
        bounds = [(0, self.max_bet_fraction) for _ in range(n)]
        
        # Optimize
        result = minimize(
            neg_expected_log_growth,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            bet_fracs = result.x
        else:
            logger.warning("Optimization failed, using simple allocation")
            bet_fracs = x0
        
        # Convert to dollar amounts
        bet_sizes = {
            opp.player: max(0, frac * bankroll)
            for opp, frac in zip(opportunities, bet_fracs)
        }
        
        return bet_sizes
    
    def generate_bet_recommendations(
        self,
        model_probs: Dict[str, float],
        betting_lines: Dict[str, BettingLine],
        bankroll: float
    ) -> pd.DataFrame:
        """
        Generate full bet recommendations with analysis.
        
        Returns DataFrame with all analysis for each potential bet.
        """
        opportunities = self.find_value_bets(model_probs, betting_lines)
        
        if not opportunities:
            return pd.DataFrame(columns=[
                'player', 'model_prob', 'implied_prob', 'edge', 
                'decimal_odds', 'american_odds', 'ev_per_dollar', 
                'recommended_bet', 'potential_profit'
            ])
        
        bet_sizes = self.optimize_bet_allocation(opportunities, bankroll)
        
        records = []
        for opp in opportunities:
            bet_amount = bet_sizes.get(opp.player, 0)
            records.append({
                'player': opp.player,
                'model_prob': opp.model_prob,
                'implied_prob': opp.line.implied_prob,
                'edge': opp.edge,
                'decimal_odds': opp.line.decimal_odds,
                'american_odds': opp.line.american_odds,
                'ev_per_dollar': opp.expected_value,
                'kelly_fraction': opp.kelly_fraction,
                'recommended_bet': bet_amount,
                'potential_profit': bet_amount * (opp.line.decimal_odds - 1),
                'expected_profit': bet_amount * opp.expected_value
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('expected_profit', ascending=False)
        
        return df


class BettingSimulator:
    """Simulates betting strategy performance over historical data."""
    
    def __init__(self, initial_bankroll: float = 1000):
        self.initial_bankroll = initial_bankroll
        
    def simulate(
        self,
        predictions: List[Dict[str, float]],  # List of game predictions
        actuals: List[str],  # List of actual first scorers
        lines: List[Dict[str, BettingLine]],  # Lines for each game
        optimizer: MultiOutcomeBetOptimizer
    ) -> pd.DataFrame:
        """
        Simulate betting strategy over historical games.
        
        Returns DataFrame with bankroll over time.
        """
        bankroll = self.initial_bankroll
        history = [{'game': 0, 'bankroll': bankroll, 'bet_amount': 0, 'profit': 0}]
        
        for i, (preds, actual, game_lines) in enumerate(zip(predictions, actuals, lines)):
            if not game_lines:
                continue
            
            # Get recommendations
            recs = optimizer.generate_bet_recommendations(preds, game_lines, bankroll)
            
            if recs.empty:
                history.append({
                    'game': i + 1,
                    'bankroll': bankroll,
                    'bet_amount': 0,
                    'profit': 0,
                    'winner': actual
                })
                continue
            
            # Calculate profit/loss
            total_bet = recs['recommended_bet'].sum()
            
            if actual in recs['player'].values:
                # We bet on the winner
                winner_row = recs[recs['player'] == actual].iloc[0]
                profit = winner_row['recommended_bet'] * (winner_row['decimal_odds'] - 1)
                # Subtract other bets that lost
                other_bets = total_bet - winner_row['recommended_bet']
                profit -= other_bets
            else:
                # We didn't bet on the winner, lose all bets
                profit = -total_bet
            
            bankroll += profit
            
            history.append({
                'game': i + 1,
                'bankroll': bankroll,
                'bet_amount': total_bet,
                'profit': profit,
                'winner': actual
            })
        
        return pd.DataFrame(history)
    
    def calculate_metrics(self, history: pd.DataFrame) -> Dict:
        """Calculate performance metrics from simulation."""
        returns = history['bankroll'].pct_change().dropna()
        
        bets = history[history['bet_amount'] > 0]
        wins = bets[bets['profit'] > 0]
        
        return {
            'final_bankroll': history['bankroll'].iloc[-1],
            'total_return': (history['bankroll'].iloc[-1] / self.initial_bankroll) - 1,
            'max_drawdown': (history['bankroll'] / history['bankroll'].cummax() - 1).min(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'total_bets': len(bets),
            'winning_bets': len(wins),
            'win_rate': len(wins) / len(bets) if len(bets) > 0 else 0,
            'avg_profit_per_bet': bets['profit'].mean() if len(bets) > 0 else 0,
        }


# Utility functions

def parse_odds_from_sportsbook(odds_str: str) -> int:
    """Parse odds string like '+450' or '-110' to int."""
    odds_str = odds_str.strip().replace(' ', '')
    if odds_str.startswith('+'):
        return int(odds_str[1:])
    elif odds_str.startswith('-'):
        return int(odds_str)
    else:
        return int(odds_str)


def calculate_total_vig(lines: List[BettingLine]) -> float:
    """
    Calculate total vig (bookmaker's edge) from betting lines.
    
    Sum of implied probabilities > 1 = vig
    """
    total_implied = sum(line.implied_prob for line in lines)
    return total_implied - 1


if __name__ == "__main__":
    # Example usage
    print("Betting Optimizer - Example")
    print("=" * 50)
    
    # Example betting lines (typical first scorer market)
    lines = {
        'LeBron James': BettingLine('LeBron James', +400),
        'Anthony Davis': BettingLine('Anthony Davis', +500),
        'Austin Reaves': BettingLine('Austin Reaves', +700),
        'D\'Angelo Russell': BettingLine('D\'Angelo Russell', +650),
        'Rui Hachimura': BettingLine('Rui Hachimura', +900),
        'Jayson Tatum': BettingLine('Jayson Tatum', +350),
        'Jaylen Brown': BettingLine('Jaylen Brown', +450),
        'Derrick White': BettingLine('Derrick White', +600),
        'Jrue Holiday': BettingLine('Jrue Holiday', +800),
        'Al Horford': BettingLine('Al Horford', +1100),
    }
    
    # Example model predictions (our probabilities)
    model_probs = {
        'LeBron James': 0.15,
        'Anthony Davis': 0.18,
        'Austin Reaves': 0.08,
        'D\'Angelo Russell': 0.07,
        'Rui Hachimura': 0.06,
        'Jayson Tatum': 0.20,
        'Jaylen Brown': 0.12,
        'Derrick White': 0.07,
        'Jrue Holiday': 0.04,
        'Al Horford': 0.03,
    }
    
    print("\nBetting Lines:")
    for player, line in lines.items():
        print(f"  {player}: {line.american_odds:+d} ({line.implied_prob:.1%})")
    
    print(f"\nTotal Vig: {calculate_total_vig(list(lines.values())):.1%}")
    
    # Find value bets
    optimizer = MultiOutcomeBetOptimizer(max_bet_fraction=0.25, min_edge=0.02)
    recommendations = optimizer.generate_bet_recommendations(
        model_probs, lines, bankroll=1000
    )
    
    print("\nValue Bets Found:")
    print(recommendations[['player', 'model_prob', 'implied_prob', 'edge', 
                          'ev_per_dollar', 'recommended_bet']].to_string(index=False))
    
    print(f"\nTotal Recommended Bet: ${recommendations['recommended_bet'].sum():.2f}")
    print(f"Expected Profit: ${recommendations['expected_profit'].sum():.2f}")
