"""
Alternative Betting Strategies for NBA First Basket Scorer

Based on historical data analysis, these strategies exploit different patterns
beyond the main model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


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
    sub_bets: Optional[List[Dict]] = field(default_factory=lambda: None)


def _american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def _implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    return 1 / _american_to_decimal(american_odds)


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
        self.first_basket_2pt_rate = 0.654
        self.league_avg_fg_pct = 0.470

    def analyze_all_strategies(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_ppg: Dict[str, float],
        home_team: str,
        away_team: str,
        tip_winner_team: Optional[str] = None,
        budget: float = 30.0,
        player_teams: Optional[Dict[str, str]] = None,
        player_fg_pct: Optional[Dict[str, float]] = None,
        player_fg3_rate: Optional[Dict[str, float]] = None,
        home_tip_prob: Optional[float] = None,
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

        # Strategy 6: Team First Scorer (player-level)
        if player_teams and home_tip_prob is not None:
            team_fs_recs = self._strategy_team_first_scorer(
                model_probs, market_odds, player_teams, home_team, away_team, home_tip_prob
            )
            recommendations.extend(team_fs_recs)

        # Strategy 7: Multi-Player Hedge
        if player_teams and tip_winner_team:
            hedge_rec = self._strategy_multi_player_hedge(
                model_probs, market_odds, player_teams, tip_winner_team
            )
            if hedge_rec:
                recommendations.append(hedge_rec)

        # Strategy 8: Score Type Filter
        if player_fg_pct and player_fg3_rate:
            score_recs = self._strategy_score_type_value(
                model_probs, market_odds, player_fg_pct, player_fg3_rate
            )
            recommendations.extend(score_recs)

        # Strategy 9: Correlated Parlay (informational)
        if player_teams and tip_winner_team:
            parlay_rec = self._strategy_correlated_parlay(
                model_probs, market_odds, player_teams, tip_winner_team
            )
            if parlay_rec:
                recommendations.append(parlay_rec)

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
            decimal_odds = _american_to_decimal(american_odds)
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

    def _estimate_team_scores_first_prob(self, home_tip_prob: float) -> float:
        """Estimate P(home team scores first) from jump ball prediction."""
        # If home wins tip, their team scores first 64.9% of the time
        # If away wins tip, home team scores first (1 - 64.9%) = 35.1%
        return home_tip_prob * self.tip_winner_team_fs_rate + (1 - home_tip_prob) * (1 - self.tip_winner_team_fs_rate)

    def _strategy_team_first_scorer(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_teams: Dict[str, str],
        home_team: str,
        away_team: str,
        home_tip_prob: float,
    ) -> List[BetRecommendation]:
        """
        Find value bets on 'Player to score their team's first basket'.

        Team-first-scorer probability is higher than game-first-scorer because
        even if the other team scores first, your player can still win.
        P(team first) ~= P(game first) / P(player's team scores first)
        """
        recs = []
        p_home_first = self._estimate_team_scores_first_prob(home_tip_prob)

        for player, game_prob in model_probs.items():
            if player not in market_odds:
                continue

            team = player_teams.get(player)
            if not team:
                continue

            # Compute team-first-scorer probability
            if team == home_team:
                p_team_first = p_home_first
            else:
                p_team_first = 1 - p_home_first

            # Avoid division by very small numbers
            if p_team_first < 0.2:
                continue

            team_fs_prob = min(game_prob / p_team_first, 0.50)

            # Compare against market odds (use game first scorer odds as proxy
            # since team first scorer odds are typically slightly shorter)
            american_odds = market_odds[player]
            decimal_odds = _american_to_decimal(american_odds)
            # Team first scorer odds are typically 60-70% of game first scorer odds
            # because the probability is higher. Approximate the team FS market.
            team_fs_decimal = 1 + (decimal_odds - 1) * 0.55
            team_fs_implied = 1 / team_fs_decimal

            edge = team_fs_prob - team_fs_implied
            ev = (team_fs_prob * (team_fs_decimal - 1)) - (1 - team_fs_prob)

            if edge > 0.03 and ev > 0:
                confidence = "HIGH" if edge > 0.06 else "MEDIUM"
                # Show approximate team FS American odds
                if team_fs_decimal >= 2:
                    team_fs_american = int((team_fs_decimal - 1) * 100)
                else:
                    team_fs_american = int(-100 / (team_fs_decimal - 1))

                recs.append(BetRecommendation(
                    strategy="TEAM_FIRST_SCORER",
                    player=player,
                    team=team,
                    bet_type="Team First Basket",
                    odds=team_fs_american,
                    implied_prob=team_fs_implied,
                    model_prob=team_fs_prob,
                    edge=edge,
                    ev=ev,
                    confidence=confidence,
                    reasoning=f"Team FS prob {team_fs_prob:.1%} (game FS {game_prob:.1%} / team first {p_team_first:.1%})"
                ))

        return sorted(recs, key=lambda x: -x.ev)[:3]

    def _strategy_multi_player_hedge(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_teams: Dict[str, str],
        tip_winner_team: str,
    ) -> Optional[BetRecommendation]:
        """
        Spread bets across top 2-3 players from tip-winning team.
        Any one hitting = profit if individual odds are high enough.
        """
        # Get players from the tip-winning team sorted by model probability
        team_players = [
            (player, prob) for player, prob in model_probs.items()
            if player_teams.get(player) == tip_winner_team and player in market_odds
        ]
        team_players.sort(key=lambda x: -x[1])

        if len(team_players) < 2:
            return None

        # Take top 3 (or 2 if only 2 available)
        n_bets = min(3, len(team_players))
        selected = team_players[:n_bets]

        # Check profitability: each player's decimal odds must be > n_bets + 1
        # so any single win covers all stakes
        sub_bets = []
        combined_prob = 0.0
        all_profitable = True

        for player, prob in selected:
            american_odds = market_odds[player]
            decimal_odds = _american_to_decimal(american_odds)
            if decimal_odds < n_bets + 1:
                all_profitable = False
            combined_prob += prob
            sub_bets.append({
                'player': player,
                'odds': american_odds,
                'model_prob': prob,
                'decimal_odds': decimal_odds,
            })

        if not all_profitable:
            return None

        # Calculate EV: for each player that could win, profit = (decimal_odds - 1) * stake - (n_bets - 1) * stake
        # Net per win = (decimal_odds - n_bets) * stake
        # Expected profit = sum(prob_i * (decimal_odds_i - n_bets)) - (1 - combined_prob) * n_bets
        # All per unit stake
        total_ev = 0.0
        for bet in sub_bets:
            total_ev += bet['model_prob'] * (bet['decimal_odds'] - n_bets)
        total_ev -= (1 - combined_prob) * n_bets
        ev_per_dollar = total_ev / n_bets

        # Worst-case win: the lowest-odds player hits
        min_decimal = min(b['decimal_odds'] for b in sub_bets)
        worst_case_profit_per_stake = min_decimal - n_bets

        edge = combined_prob - (n_bets / (min_decimal + n_bets - 1))

        if ev_per_dollar > 0 and worst_case_profit_per_stake > 0:
            player_names = [b['player'] for b in sub_bets]
            confidence = "HIGH" if combined_prob > 0.30 else "MEDIUM"

            return BetRecommendation(
                strategy="MULTI_PLAYER_HEDGE",
                player=" + ".join(player_names),
                team=tip_winner_team,
                bet_type="First Basket Scorer (x{})".format(n_bets),
                odds=min(b['odds'] for b in sub_bets),
                implied_prob=1.0 - combined_prob,
                model_prob=combined_prob,
                edge=edge,
                ev=ev_per_dollar,
                confidence=confidence,
                reasoning=f"{n_bets} bets on {tip_winner_team}: combined {combined_prob:.1%} hit rate, any win profits",
                sub_bets=sub_bets,
            )

        return None

    def _strategy_score_type_value(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_fg_pct: Dict[str, float],
        player_fg3_rate: Dict[str, float],
    ) -> List[BetRecommendation]:
        """
        Adjust model probabilities to favor close-range scorers (2PT bias).
        85% of first baskets are 2PT shots; high FG% / low 3PT rate = edge.
        """
        if not player_fg_pct:
            return []

        # Compute adjusted probabilities
        adjusted = {}
        for player, prob in model_probs.items():
            fg_pct = player_fg_pct.get(player, self.league_avg_fg_pct)
            fg3_rate = player_fg3_rate.get(player, 0.30)
            close_range_factor = (fg_pct / self.league_avg_fg_pct) * (1 - 0.3 * fg3_rate)
            adjusted[player] = prob * close_range_factor

        # Renormalize
        total = sum(adjusted.values())
        if total <= 0:
            return []
        adjusted = {p: v / total for p, v in adjusted.items()}

        recs = []
        for player, adj_prob in adjusted.items():
            if player not in market_odds:
                continue

            original_prob = model_probs[player]
            # Only recommend if the adjustment meaningfully increases probability
            if adj_prob <= original_prob * 1.05:
                continue

            american_odds = market_odds[player]
            decimal_odds = _american_to_decimal(american_odds)
            implied_prob = 1 / decimal_odds
            edge = adj_prob - implied_prob
            ev = (adj_prob * (decimal_odds - 1)) - (1 - adj_prob)

            if edge > 0.02 and ev > 0:
                fg_pct = player_fg_pct.get(player, 0)
                fg3_rate = player_fg3_rate.get(player, 0)
                recs.append(BetRecommendation(
                    strategy="SCORE_TYPE_FILTER",
                    player=player,
                    team=None,
                    bet_type="First Basket Scorer",
                    odds=american_odds,
                    implied_prob=implied_prob,
                    model_prob=adj_prob,
                    edge=edge,
                    ev=ev,
                    confidence="MEDIUM",
                    reasoning=f"2PT-adjusted prob {adj_prob:.1%} (FG% {fg_pct:.0%}, 3PT rate {fg3_rate:.0%})"
                ))

        return sorted(recs, key=lambda x: -x.ev)[:2]

    def _strategy_correlated_parlay(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, int],
        player_teams: Dict[str, str],
        tip_winner_team: str,
    ) -> Optional[BetRecommendation]:
        """
        Parlay 'Team scores first' + 'Player scores first basket'.
        Perfectly correlated: if player scores first, team scored first.
        True prob = P(player scores first). Most books block this.
        """
        # Find best player from tip-winning team
        best_player = None
        best_prob = 0.0
        for player, prob in model_probs.items():
            if player_teams.get(player) == tip_winner_team and player in market_odds:
                if prob > best_prob:
                    best_prob = prob
                    best_player = player

        if not best_player or best_prob < 0.10:
            return None

        # Parlay payout = team_odds * player_odds
        # Assume team first scorer odds = -110 (decimal 1.909)
        team_decimal = 1.909
        player_american = market_odds[best_player]
        player_decimal = _american_to_decimal(player_american)
        parlay_decimal = team_decimal * player_decimal

        # True probability is just P(player scores game first)
        true_prob = best_prob
        # Independent parlay implied probability = P(team) * P(player)
        parlay_implied = (1 / team_decimal) * (1 / player_decimal)

        edge = true_prob - parlay_implied
        ev = (true_prob * (parlay_decimal - 1)) - (1 - true_prob)

        if edge > 0.02 and ev > 0:
            parlay_american = int((parlay_decimal - 1) * 100) if parlay_decimal >= 2 else int(-100 / (parlay_decimal - 1))
            return BetRecommendation(
                strategy="CORRELATED_PARLAY",
                player=best_player,
                team=tip_winner_team,
                bet_type="Parlay: Team 1st + Player 1st",
                odds=parlay_american,
                implied_prob=parlay_implied,
                model_prob=true_prob,
                edge=edge,
                ev=ev,
                confidence="HIGH",
                reasoning=f"Correlated parlay: true prob {true_prob:.1%} vs implied {parlay_implied:.1%} (most books block)"
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
    budget: float = 30.0,
    player_teams: Optional[Dict[str, str]] = None,
    player_fg_pct: Optional[Dict[str, float]] = None,
    player_fg3_rate: Optional[Dict[str, float]] = None,
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
        budget=budget,
        player_teams=player_teams,
        player_fg_pct=player_fg_pct,
        player_fg3_rate=player_fg3_rate,
        home_tip_prob=home_jb_win_prob,
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

            print(f"    Bet Type: {rec.bet_type}")
            print(f"    Odds: {rec.odds:+d}")
            print(f"    Model: {rec.model_prob:.1%} vs Market: {rec.implied_prob:.1%}")
            print(f"    Edge: {rec.edge:+.1%} | EV: ${rec.ev:.3f}/$1")
            print(f"    Confidence: {rec.confidence}")
            print(f"    Reasoning: {rec.reasoning}")

            if rec.sub_bets:
                print(f"    Breakdown:")
                for sb in rec.sub_bets:
                    print(f"      - {sb['player']} ({sb['odds']:+d}) model: {sb['model_prob']:.1%}")

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
