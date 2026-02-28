"""
Alternative Betting Strategies for NBA First Basket Scorer

Based on historical data analysis, these strategies exploit different patterns
beyond the main model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BetRecommendation:
    """A betting recommendation."""
    strategy: str
    player: Optional[str]
    team: Optional[str]
    bet_type: str
    odds: int
    implied_prob: float
    model_prob: float
    edge: float
    ev: float
    confidence: str
    reasoning: str


class AlternativeStrategies:
    """
    Alternative betting strategies based on pattern analysis.

    Key findings from historical data:
    1. Tip winner's team scores first 64.9% of the time
    2. Home team scores first 54.2% of the time
    3. Highest PPG player scores first 13.9% (vs 10% baseline)
    4. Top 2 PPG players score first 26.7% (vs 20% baseline)
    5. Contrarian longshots can be +EV at high enough odds
    """

    def __init__(self):
        # Historical probabilities from data
        self.tip_winner_team_fs_rate = 0.649
        self.home_team_fs_rate = 0.542
        self.highest_ppg_fs_rate = 0.139
        self.top2_ppg_fs_rate = 0.267
        self.model_top1_accuracy = 0.142
        self.model_top3_accuracy = 0.390
        self.model_top5_accuracy = 0.611

    def analyze_all_strategies(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_ppg: Dict[str, float],
        home_team: str,
        away_team: str,
        tip_winner_team: Optional[str] = None,
        budget: float = 30.0
    ) -> List[BetRecommendation]:
        """
        Analyze all alternative strategies and return recommendations.
        """
        recommendations = []

        # Strategy 1: Value Bet (model edge)
        value_recs = self._strategy_value_bets(model_probs, market_odds)
        recommendations.extend(value_recs)

        # Strategy 2: Efficiency Play (PPG * FG%)
        eff_rec = self._strategy_efficiency_play(model_probs, market_odds, player_ppg, home_team, away_team)
        if eff_rec:
            recommendations.append(eff_rec)

        # Strategy 3: Contrarian Longshot
        contrarian_rec = self._strategy_contrarian(model_probs, market_odds)
        if contrarian_rec:
            recommendations.append(contrarian_rec)

        # Strategy 4: Second Choice (when close)
        second_rec = self._strategy_second_choice(model_probs, market_odds)
        if second_rec:
            recommendations.append(second_rec)

        # Strategy 5: Team First Scorer (if available)
        if tip_winner_team:
            team_rec = self._strategy_tip_winner_team(tip_winner_team, home_team)
            if team_rec:
                recommendations.append(team_rec)

        return recommendations

    def _strategy_value_bets(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int]
    ) -> List[BetRecommendation]:
        """Find standard value bets where model > market."""
        recs = []

        for player, model_prob in model_probs.items():
            if player not in market_odds:
                continue

            american_odds = market_odds[player]
            if american_odds > 0:
                decimal_odds = 1 + (american_odds / 100)
            else:
                decimal_odds = 1 + (100 / abs(american_odds))

            implied_prob = 1 / decimal_odds
            edge = model_prob - implied_prob
            ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)

            if edge > 0.02 and ev > 0:
                confidence = "HIGH" if edge > 0.04 else "MEDIUM"
                recs.append(BetRecommendation(
                    strategy="VALUE_BET",
                    player=player,
                    team=None,
                    bet_type="First Basket Scorer",
                    odds=american_odds,
                    implied_prob=implied_prob,
                    model_prob=model_prob,
                    edge=edge,
                    ev=ev,
                    confidence=confidence,
                    reasoning=f"Model gives {model_prob:.1%} vs market {implied_prob:.1%}"
                ))

        return sorted(recs, key=lambda x: -x.ev)

    def _strategy_efficiency_play(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_ppg: Dict[str, float],
        home_team: str,
        away_team: str
    ) -> Optional[BetRecommendation]:
        """Bet on highest efficiency (PPG) player if undervalued."""
        if not player_ppg:
            return None

        # Find highest PPG player
        best_player = max(player_ppg.items(), key=lambda x: x[1])
        player, ppg = best_player

        if player not in market_odds:
            return None

        american_odds = market_odds[player]
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        implied_prob = 1 / decimal_odds

        # Historical rate for highest PPG is ~13.9%
        est_prob = self.highest_ppg_fs_rate
        edge = est_prob - implied_prob
        ev = (est_prob * (decimal_odds - 1)) - (1 - est_prob)

        if edge > 0.01:
            return BetRecommendation(
                strategy="EFFICIENCY_PLAY",
                player=player,
                team=None,
                bet_type="First Basket Scorer",
                odds=american_odds,
                implied_prob=implied_prob,
                model_prob=est_prob,
                edge=edge,
                ev=ev,
                confidence="MEDIUM",
                reasoning=f"Highest PPG ({ppg:.1f}) historically scores first 13.9%"
            )

        return None

    def _strategy_contrarian(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int]
    ) -> Optional[BetRecommendation]:
        """Bet longest odds player if odds are extreme enough."""
        # Find the longest odds player
        longest_player = None
        longest_odds = 0

        for player, odds in market_odds.items():
            if odds > longest_odds:
                longest_odds = odds
                longest_player = player

        if not longest_player or longest_odds < 1500:
            return None

        # Lowest model probability player hits ~7.9% of time
        est_prob = 0.079

        decimal_odds = 1 + (longest_odds / 100)
        implied_prob = 1 / decimal_odds
        edge = est_prob - implied_prob
        ev = (est_prob * (decimal_odds - 1)) - (1 - est_prob)

        if ev > 0:
            return BetRecommendation(
                strategy="CONTRARIAN_LONGSHOT",
                player=longest_player,
                team=None,
                bet_type="First Basket Scorer",
                odds=longest_odds,
                implied_prob=implied_prob,
                model_prob=est_prob,
                edge=edge,
                ev=ev,
                confidence="LOW",
                reasoning=f"Longshots at +{longest_odds} can be +EV; high variance"
            )

        return None

    def _strategy_second_choice(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int]
    ) -> Optional[BetRecommendation]:
        """Bet 2nd choice when gap with 1st is small."""
        sorted_players = sorted(model_probs.items(), key=lambda x: -x[1])

        if len(sorted_players) < 2:
            return None

        top1_player, top1_prob = sorted_players[0]
        top2_player, top2_prob = sorted_players[1]

        # Only if gap is < 3%
        if top1_prob - top2_prob > 0.03:
            return None

        if top2_player not in market_odds:
            return None

        american_odds = market_odds[top2_player]
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        implied_prob = 1 / decimal_odds
        edge = top2_prob - implied_prob
        ev = (top2_prob * (decimal_odds - 1)) - (1 - top2_prob)

        if edge > 0.01:
            return BetRecommendation(
                strategy="SECOND_CHOICE",
                player=top2_player,
                team=None,
                bet_type="First Basket Scorer",
                odds=american_odds,
                implied_prob=implied_prob,
                model_prob=top2_prob,
                edge=edge,
                ev=ev,
                confidence="MEDIUM",
                reasoning=f"Close to #1 ({top1_prob:.1%} vs {top2_prob:.1%}) but better odds"
            )

        return None

    def _strategy_tip_winner_team(
        self,
        tip_winner_team: str,
        home_team: str
    ) -> Optional[BetRecommendation]:
        """Bet on tip winner's team scoring first (if prop available)."""
        # Tip winner team scores first 64.9% historically
        est_prob = self.tip_winner_team_fs_rate

        # Assume typical odds for "which team scores first" is around -110
        # (varies by book)
        american_odds = -110
        decimal_odds = 1 + (100 / 110)
        implied_prob = 1 / decimal_odds

        edge = est_prob - implied_prob
        ev = (est_prob * (decimal_odds - 1)) - (1 - est_prob)

        return BetRecommendation(
            strategy="TIP_WINNER_TEAM",
            player=None,
            team=tip_winner_team,
            bet_type="Team First Scorer",
            odds=american_odds,
            implied_prob=implied_prob,
            model_prob=est_prob,
            edge=edge,
            ev=ev,
            confidence="HIGH",
            reasoning=f"Tip winner team scores first 64.9% vs ~52% implied"
        )


def print_strategy_analysis(
    model_probs: Dict[str, float],
    market_odds: Dict[str, int],
    player_ppg: Dict[str, float],
    home_team: str,
    away_team: str,
    home_jb_win_prob: float,
    budget: float = 30.0
):
    """Print comprehensive betting strategy analysis."""

    strategies = AlternativeStrategies()

    # Determine likely tip winner
    tip_winner = home_team if home_jb_win_prob > 0.5 else away_team

    recommendations = strategies.analyze_all_strategies(
        model_probs=model_probs,
        market_odds=market_odds,
        player_ppg=player_ppg,
        home_team=home_team,
        away_team=away_team,
        tip_winner_team=tip_winner,
        budget=budget
    )

    print("=" * 70)
    print("COMPREHENSIVE BETTING STRATEGY ANALYSIS")
    print("=" * 70)

    # Group by strategy type
    strategy_groups = {}
    for rec in recommendations:
        if rec.strategy not in strategy_groups:
            strategy_groups[rec.strategy] = []
        strategy_groups[rec.strategy].append(rec)

    for strategy, recs in strategy_groups.items():
        print(f"\n{strategy.replace('_', ' ')}")
        print("-" * 50)

        for rec in recs:
            if rec.player:
                print(f"\n  {rec.player}")
            elif rec.team:
                print(f"\n  {rec.team} (Team)")

            print(f"    Odds: {rec.odds:+d}")
            print(f"    Model: {rec.model_prob:.1%} vs Market: {rec.implied_prob:.1%}")
            print(f"    Edge: {rec.edge:+.1%} | EV: ${rec.ev:.3f}/$1")
            print(f"    Confidence: {rec.confidence}")
            print(f"    Reasoning: {rec.reasoning}")

    # Allocation suggestion
    print("\n" + "=" * 70)
    print(f"SUGGESTED ALLOCATION (${budget:.0f} BUDGET)")
    print("=" * 70)

    # Filter to positive EV bets
    positive_ev = [r for r in recommendations if r.ev > 0]

    if not positive_ev:
        print("\nNo positive EV bets found. Consider passing on this game.")
        return

    # Allocate by edge
    total_edge = sum(r.edge for r in positive_ev if r.edge > 0)

    print()
    total_allocated = 0
    for rec in sorted(positive_ev, key=lambda x: -x.ev)[:4]:  # Top 4
        if rec.edge > 0:
            allocation = (rec.edge / total_edge) * budget
            allocation = min(allocation, budget * 0.5)  # Cap at 50%
            total_allocated += allocation

            target = rec.player if rec.player else rec.team
            print(f"  ${allocation:.2f} on {target} ({rec.odds:+d}) - {rec.strategy}")

    remaining = budget - total_allocated
    if remaining > 0:
        print(f"  ${remaining:.2f} reserve (for live betting or hedge)")

    return recommendations


if __name__ == "__main__":
    # Demo with sample data
    model_probs = {
        'Jaylen Brown': 0.177,
        'Bobby Portis': 0.155,
        'Myles Turner': 0.108,
        'Derrick White': 0.094,
        'Payton Pritchard': 0.093,
        'Neemias Queta': 0.092,
        'Kyle Kuzma': 0.092,
        'AJ Green': 0.067,
        'Ryan Rollins': 0.065,
        'Sam Hauser': 0.057,
    }

    market_odds = {
        'Jaylen Brown': +390,
        'Payton Pritchard': +600,
        'Derrick White': +650,
        'Neemias Queta': +950,
        'Sam Hauser': +1000,
        'Ryan Rollins': +750,
        'Bobby Portis': +800,
        'Kyle Kuzma': +950,
        'AJ Green': +1000,
        'Myles Turner': +1000,
    }

    player_ppg = {
        'Jaylen Brown': 24.5,
        'Bobby Portis': 14.2,
        'Kyle Kuzma': 15.8,
        'Derrick White': 12.1,
        'Myles Turner': 13.5,
        'Payton Pritchard': 11.2,
        'Neemias Queta': 5.4,
        'Ryan Rollins': 8.1,
        'AJ Green': 7.2,
        'Sam Hauser': 8.5,
    }

    print_strategy_analysis(
        model_probs=model_probs,
        market_odds=market_odds,
        player_ppg=player_ppg,
        home_team='BOS',
        away_team='MIL',
        home_jb_win_prob=0.601,
        budget=30.0
    )
