"""
Historical Backtesting for Betting Strategies

Uses V6 XGBoost models to simulate betting on historical games.
Supports backtesting individual strategies and comparison reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.betting.strategy import (
    BettingStrategy, BettingSimulator, BettingOdds,
    generate_synthetic_odds
)
from src.data.train_models_v6 import JumpBallModelV6, PlayerFirstScorerModelV6
from src.betting.alternative_strategies import _american_to_decimal, _implied_probability

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

        # Load V6 models
        self.jb_model = joblib.load(self.model_dir / "jump_ball_model_v6.joblib")
        self.player_model = joblib.load(self.model_dir / "player_first_scorer_model_v6.joblib")
        logger.info("V6 models loaded")

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
        """
        odds_list = []
        n_players = len(model_probs)

        if n_players == 0:
            return []

        naive_probs = np.array([0.15, 0.14, 0.11, 0.08, 0.07,
                                 0.14, 0.13, 0.08, 0.06, 0.04])

        noise = np.random.normal(0, 0.02, n_players)
        naive_probs = naive_probs + noise
        naive_probs = np.maximum(naive_probs, 0.02)
        naive_probs = naive_probs / naive_probs.sum()

        market_probs = market_efficiency * model_probs + (1 - market_efficiency) * naive_probs
        market_probs = market_probs / market_probs.sum()

        for i, (prob, pid) in enumerate(zip(market_probs, player_ids)):
            if prob <= 0.01:
                continue

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

    def simulate_market_american_odds(
        self,
        model_probs: np.ndarray,
        player_names: List[str],
        market_efficiency: float = 0.7
    ) -> Dict[str, int]:
        """Generate simulated market odds as Dict[player_name, american_odds]."""
        n_players = len(model_probs)
        if n_players == 0:
            return {}

        naive_probs = np.array([0.15, 0.14, 0.11, 0.08, 0.07,
                                 0.14, 0.13, 0.08, 0.06, 0.04])

        noise = np.random.normal(0, 0.02, n_players)
        naive_probs = naive_probs + noise
        naive_probs = np.maximum(naive_probs, 0.02)
        naive_probs = naive_probs / naive_probs.sum()

        market_probs = market_efficiency * model_probs + (1 - market_efficiency) * naive_probs
        market_probs = market_probs / market_probs.sum()

        odds_dict = {}
        for prob, name in zip(market_probs, player_names):
            if prob <= 0.01:
                continue
            implied = prob * (1 + self.vig)
            implied = min(implied, 0.95)
            decimal_odds = 1 / implied

            if decimal_odds >= 2.0:
                american = int((decimal_odds - 1) * 100)
            else:
                american = int(-100 / (decimal_odds - 1))
            odds_dict[name] = american

        return odds_dict

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        strategy: BettingStrategy,
        use_random_odds: bool = False
    ) -> BettingSimulator:
        """Run backtest on test data."""
        fresh_strategy = BettingStrategy(
            min_edge=strategy.min_edge,
            min_prob=strategy.min_prob,
            kelly_fraction=strategy.kelly_fraction,
            max_bet_pct=strategy.max_bet_pct,
            bankroll=strategy.bankroll
        )
        simulator = BettingSimulator(fresh_strategy, fresh_strategy.bankroll)

        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        logger.info(f"Backtesting on {len(valid_games)} games with valid first scorer")

        for idx, row in valid_games.iterrows():
            game_id = row['game_id']
            game_df = pd.DataFrame([row])
            player_probs = self.player_model.predict_proba(game_df)[0]

            player_ids = []
            for i in range(5):
                home_token = row[f'home_{i}_token']
                player_ids.append(self.get_player_id_from_token(home_token))
            for i in range(5):
                away_token = row[f'away_{i}_token']
                player_ids.append(self.get_player_id_from_token(away_token))

            first_scorer_pos = int(row['first_scorer_position'])
            actual_first_scorer = player_ids[first_scorer_pos] if 0 <= first_scorer_pos < 10 else None

            if actual_first_scorer is None:
                continue

            if use_random_odds:
                random_model_probs = np.random.dirichlet(np.ones(10))
                market_odds = self.simulate_market_odds(player_probs, player_ids)
                model_probs_dict = {pid: prob for pid, prob in zip(player_ids, random_model_probs)}
            else:
                market_odds = self.simulate_market_odds(player_probs, player_ids)
                model_probs_dict = {pid: prob for pid, prob in zip(player_ids, player_probs)}

            simulator.simulate_game(
                game_id=game_id,
                model_probs=model_probs_dict,
                market_odds=market_odds,
                actual_first_scorer=actual_first_scorer
            )

        return simulator

    def run_edge_analysis(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze edge by comparing model predictions to actual outcomes."""
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()

        results = []
        for idx, row in valid_games.iterrows():
            game_df = pd.DataFrame([row])
            player_probs = self.player_model.predict_proba(game_df)[0]

            first_scorer_pos = int(row['first_scorer_position'])

            if 0 <= first_scorer_pos < 10:
                prob_on_winner = player_probs[first_scorer_pos]
                max_prob = max(player_probs)
                predicted_pos = np.argmax(player_probs)
                correct = (predicted_pos == first_scorer_pos)

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


class StrategyBacktester(HistoricalBacktester):
    """Backtest advanced betting strategies on historical data."""

    TIP_WINNER_TEAM_FS_RATE = 0.649
    LEAGUE_AVG_FG_PCT = 0.470

    def _get_game_data(self, row):
        """Extract common game data from a row."""
        game_df = pd.DataFrame([row])
        player_probs = self.player_model.predict_proba(game_df)[0]
        jb_probs = self.jb_model.predict_proba(game_df)[0]

        # P(home wins tip) - jb_probs may be scalar or array
        if hasattr(jb_probs, '__len__') and len(jb_probs) > 1:
            home_tip_prob = jb_probs[1]  # class 1 = home wins
        else:
            home_tip_prob = float(jb_probs)

        player_names = []
        player_teams = {}
        player_fg_pct = {}
        player_fg3_rate = {}
        player_ppg = {}

        for i in range(5):
            token = row[f'home_{i}_token']
            pid = self.get_player_id_from_token(token)
            name = self.get_player_name(pid)
            player_names.append(name)
            player_teams[name] = 'home'
            player_fg_pct[name] = row.get(f'home_{i}_fg_pct', self.LEAGUE_AVG_FG_PCT)
            player_fg3_rate[name] = row.get(f'home_{i}_fg3_rate', 0.30)
            player_ppg[name] = row.get(f'home_{i}_ppg', 10.0)

        for i in range(5):
            token = row[f'away_{i}_token']
            pid = self.get_player_id_from_token(token)
            name = self.get_player_name(pid)
            player_names.append(name)
            player_teams[name] = 'away'
            player_fg_pct[name] = row.get(f'away_{i}_fg_pct', self.LEAGUE_AVG_FG_PCT)
            player_fg3_rate[name] = row.get(f'away_{i}_fg3_rate', 0.30)
            player_ppg[name] = row.get(f'away_{i}_ppg', 10.0)

        first_scorer_pos = int(row['first_scorer_position'])
        actual_first_scorer = player_names[first_scorer_pos] if 0 <= first_scorer_pos < 10 else None
        first_scorer_is_home = bool(row.get('first_scorer_is_home', first_scorer_pos < 5))

        model_probs = {name: prob for name, prob in zip(player_names, player_probs)}

        return {
            'player_names': player_names,
            'player_probs': player_probs,
            'model_probs': model_probs,
            'home_tip_prob': home_tip_prob,
            'player_teams': player_teams,
            'player_fg_pct': player_fg_pct,
            'player_fg3_rate': player_fg3_rate,
            'player_ppg': player_ppg,
            'first_scorer_pos': first_scorer_pos,
            'actual_first_scorer': actual_first_scorer,
            'first_scorer_is_home': first_scorer_is_home,
        }

    def backtest_value_bet(self, test_df: pd.DataFrame, min_edge: float = 0.02) -> Dict:
        """Backtest basic value bet strategy (baseline)."""
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        wins = 0
        losses = 0
        total_wagered = 0.0
        total_profit = 0.0
        stake = 10.0

        for idx, row in valid_games.iterrows():
            gd = self._get_game_data(row)
            if gd['actual_first_scorer'] is None:
                continue

            market_odds = self.simulate_market_american_odds(
                gd['player_probs'], gd['player_names']
            )

            # Find best value bet
            best_player = None
            best_ev = 0

            for player, model_prob in gd['model_probs'].items():
                if player not in market_odds:
                    continue
                american = market_odds[player]
                decimal_odds = _american_to_decimal(american)
                implied = 1 / decimal_odds
                edge = model_prob - implied
                ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)

                if edge > min_edge and ev > best_ev:
                    best_ev = ev
                    best_player = player

            if best_player:
                total_wagered += stake
                american = market_odds[best_player]
                decimal_odds = _american_to_decimal(american)

                if best_player == gd['actual_first_scorer']:
                    wins += 1
                    total_profit += stake * (decimal_odds - 1)
                else:
                    losses += 1
                    total_profit -= stake

        n_bets = wins + losses
        return {
            'strategy': 'VALUE_BET',
            'games': len(valid_games),
            'bets': n_bets,
            'wins': wins,
            'win_rate': wins / n_bets if n_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0,
        }

    def backtest_team_first_scorer(self, test_df: pd.DataFrame) -> Dict:
        """
        Backtest team-first-scorer strategy.

        Limitation: we only know the game first scorer, who is also the
        team-first scorer for the team that scored first. We can only
        evaluate bets on players from that team.
        """
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        wins = 0
        losses = 0
        total_wagered = 0.0
        total_profit = 0.0
        stake = 10.0
        games_evaluated = 0

        for idx, row in valid_games.iterrows():
            gd = self._get_game_data(row)
            if gd['actual_first_scorer'] is None:
                continue

            p_home_first = (
                gd['home_tip_prob'] * self.TIP_WINNER_TEAM_FS_RATE
                + (1 - gd['home_tip_prob']) * (1 - self.TIP_WINNER_TEAM_FS_RATE)
            )

            # Generate game-first-scorer market odds
            game_market_odds = self.simulate_market_american_odds(
                gd['player_probs'], gd['player_names']
            )

            # Compute team-first-scorer probabilities
            best_player = None
            best_ev = 0.0

            for player, game_prob in gd['model_probs'].items():
                team = gd['player_teams'][player]
                is_home = (team == 'home')

                # Only evaluate for the team that actually scored first
                if is_home != gd['first_scorer_is_home']:
                    continue

                p_team_first = p_home_first if is_home else (1 - p_home_first)
                if p_team_first < 0.15:
                    continue

                team_fs_prob = min(game_prob / p_team_first, 0.50)

                if player not in game_market_odds:
                    continue

                # Approximate team FS odds (shorter than game FS odds)
                game_decimal = _american_to_decimal(game_market_odds[player])
                team_fs_decimal = 1 + (game_decimal - 1) * 0.55
                team_fs_implied = 1 / team_fs_decimal

                edge = team_fs_prob - team_fs_implied
                ev = (team_fs_prob * (team_fs_decimal - 1)) - (1 - team_fs_prob)

                if edge > 0.03 and ev > best_ev:
                    best_ev = ev
                    best_player = player

            if best_player:
                games_evaluated += 1
                total_wagered += stake

                # Winner = the actual first scorer (who is the team first scorer
                # for the team that scored first)
                if best_player == gd['actual_first_scorer']:
                    # Compute payout using team FS odds
                    game_decimal = _american_to_decimal(game_market_odds[best_player])
                    team_fs_decimal = 1 + (game_decimal - 1) * 0.55
                    wins += 1
                    total_profit += stake * (team_fs_decimal - 1)
                else:
                    losses += 1
                    total_profit -= stake

        n_bets = wins + losses
        return {
            'strategy': 'TEAM_FIRST_SCORER',
            'games': games_evaluated,
            'bets': n_bets,
            'wins': wins,
            'win_rate': wins / n_bets if n_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0,
            'note': 'Only evaluated on team that scored first (~50% of games)',
        }

    def backtest_multi_player_hedge(self, test_df: pd.DataFrame, n_picks: int = 3) -> Dict:
        """
        Backtest multi-player hedge: bet top N players from tip-winner team.
        Win if ANY of them score the game's first basket.
        """
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        wins = 0
        losses = 0
        total_wagered = 0.0
        total_profit = 0.0
        stake_per_bet = 10.0
        games_bet = 0

        for idx, row in valid_games.iterrows():
            gd = self._get_game_data(row)
            if gd['actual_first_scorer'] is None:
                continue

            # Determine tip winner team
            tip_winner = 'home' if gd['home_tip_prob'] > 0.5 else 'away'

            # Get players from tip-winning team with market odds
            market_odds = self.simulate_market_american_odds(
                gd['player_probs'], gd['player_names']
            )

            team_players = [
                (name, gd['model_probs'][name])
                for name in gd['player_names']
                if gd['player_teams'][name] == tip_winner and name in market_odds
            ]
            team_players.sort(key=lambda x: -x[1])

            if len(team_players) < n_picks:
                continue

            selected = team_players[:n_picks]

            # Check profitability: each decimal odds must be > n_picks + 1
            all_profitable = True
            for name, prob in selected:
                decimal_odds = _american_to_decimal(market_odds[name])
                if decimal_odds < n_picks + 1:
                    all_profitable = False
                    break

            if not all_profitable:
                continue

            games_bet += 1
            total_stake = stake_per_bet * n_picks
            total_wagered += total_stake

            # Check if any selected player was the actual first scorer
            winner = None
            for name, prob in selected:
                if name == gd['actual_first_scorer']:
                    winner = name
                    break

            if winner:
                wins += 1
                decimal_odds = _american_to_decimal(market_odds[winner])
                payout = stake_per_bet * decimal_odds
                total_profit += payout - total_stake
            else:
                losses += 1
                total_profit -= total_stake

        n_bets = wins + losses
        return {
            'strategy': f'MULTI_PLAYER_HEDGE (top {n_picks})',
            'games': games_bet,
            'bets': n_bets,
            'wins': wins,
            'win_rate': wins / n_bets if n_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0,
        }

    def backtest_score_type_filter(self, test_df: pd.DataFrame, min_edge: float = 0.02) -> Dict:
        """
        Backtest score-type-adjusted value bets.
        Boosts players with high FG% and low 3PT rate.
        """
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        wins = 0
        losses = 0
        total_wagered = 0.0
        total_profit = 0.0
        stake = 10.0

        for idx, row in valid_games.iterrows():
            gd = self._get_game_data(row)
            if gd['actual_first_scorer'] is None:
                continue

            market_odds = self.simulate_market_american_odds(
                gd['player_probs'], gd['player_names']
            )

            # Compute score-type-adjusted probabilities
            adjusted = {}
            for name, prob in gd['model_probs'].items():
                fg_pct = gd['player_fg_pct'].get(name, self.LEAGUE_AVG_FG_PCT)
                fg3_rate = gd['player_fg3_rate'].get(name, 0.30)
                factor = (fg_pct / self.LEAGUE_AVG_FG_PCT) * (1 - 0.3 * fg3_rate)
                adjusted[name] = prob * factor

            total = sum(adjusted.values())
            if total <= 0:
                continue
            adjusted = {n: v / total for n, v in adjusted.items()}

            # Find best adjusted value bet
            best_player = None
            best_ev = 0.0

            for name, adj_prob in adjusted.items():
                original_prob = gd['model_probs'][name]
                # Only if adjustment meaningfully increases probability
                if adj_prob <= original_prob * 1.05:
                    continue
                if name not in market_odds:
                    continue

                decimal_odds = _american_to_decimal(market_odds[name])
                implied = 1 / decimal_odds
                edge = adj_prob - implied
                ev = (adj_prob * (decimal_odds - 1)) - (1 - adj_prob)

                if edge > min_edge and ev > best_ev:
                    best_ev = ev
                    best_player = name

            if best_player:
                total_wagered += stake
                decimal_odds = _american_to_decimal(market_odds[best_player])

                if best_player == gd['actual_first_scorer']:
                    wins += 1
                    total_profit += stake * (decimal_odds - 1)
                else:
                    losses += 1
                    total_profit -= stake

        n_bets = wins + losses
        return {
            'strategy': 'SCORE_TYPE_FILTER',
            'games': len(valid_games),
            'bets': n_bets,
            'wins': wins,
            'win_rate': wins / n_bets if n_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0,
        }

    def backtest_correlated_parlay(self, test_df: pd.DataFrame) -> Dict:
        """
        Backtest correlated parlay: Team scores first + Player scores first.
        Wins if the player scores the game's first basket.
        """
        valid_games = test_df[test_df['first_scorer_position'] >= 0].copy()
        wins = 0
        losses = 0
        total_wagered = 0.0
        total_profit = 0.0
        stake = 10.0

        for idx, row in valid_games.iterrows():
            gd = self._get_game_data(row)
            if gd['actual_first_scorer'] is None:
                continue

            tip_winner = 'home' if gd['home_tip_prob'] > 0.5 else 'away'

            market_odds = self.simulate_market_american_odds(
                gd['player_probs'], gd['player_names']
            )

            # Find best player from tip-winning team
            best_player = None
            best_prob = 0.0

            for name, prob in gd['model_probs'].items():
                if gd['player_teams'][name] == tip_winner and name in market_odds and prob > best_prob:
                    best_prob = prob
                    best_player = name

            if not best_player or best_prob < 0.10:
                continue

            # Parlay: team first (-110) * player first scorer
            team_decimal = 1.909  # -110
            player_decimal = _american_to_decimal(market_odds[best_player])
            parlay_decimal = team_decimal * player_decimal

            # True prob = P(player scores first)
            true_prob = best_prob
            parlay_implied = (1 / team_decimal) * (1 / player_decimal)

            edge = true_prob - parlay_implied
            ev = (true_prob * (parlay_decimal - 1)) - (1 - true_prob)

            if edge > 0.02 and ev > 0:
                total_wagered += stake

                if best_player == gd['actual_first_scorer']:
                    wins += 1
                    total_profit += stake * (parlay_decimal - 1)
                else:
                    losses += 1
                    total_profit -= stake

        n_bets = wins + losses
        return {
            'strategy': 'CORRELATED_PARLAY',
            'games': len(valid_games),
            'bets': n_bets,
            'wins': wins,
            'win_rate': wins / n_bets if n_bets > 0 else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': total_profit / total_wagered if total_wagered > 0 else 0,
            'note': 'Most sportsbooks block correlated parlays',
        }

    def backtest_all(self, test_df: pd.DataFrame) -> List[Dict]:
        """Run all strategy backtests and return results."""
        np.random.seed(42)  # Reproducible market odds

        results = []
        results.append(self.backtest_value_bet(test_df))

        np.random.seed(42)
        results.append(self.backtest_team_first_scorer(test_df))

        np.random.seed(42)
        results.append(self.backtest_multi_player_hedge(test_df, n_picks=3))

        np.random.seed(42)
        results.append(self.backtest_multi_player_hedge(test_df, n_picks=2))

        np.random.seed(42)
        results.append(self.backtest_score_type_filter(test_df))

        np.random.seed(42)
        results.append(self.backtest_correlated_parlay(test_df))

        return results

    def print_comparison_report(self, results: List[Dict]):
        """Print formatted comparison of all strategies."""
        print("\n" + "=" * 90)
        print("STRATEGY BACKTEST COMPARISON (2025-26 season test set)")
        print("=" * 90)
        print(f"\nAssumptions: {self.vig*100:.0f}% vig, 70% market efficiency, $10 flat bets")
        print()

        header = f"{'Strategy':<30} {'Games':>6} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Wagered':>10} {'Profit':>10} {'ROI':>8}"
        print(header)
        print("-" * 90)

        for r in results:
            line = (
                f"{r['strategy']:<30} "
                f"{r['games']:>6} "
                f"{r['bets']:>6} "
                f"{r['wins']:>6} "
                f"{r['win_rate']:>6.1%} "
                f"${r['total_wagered']:>9.0f} "
                f"${r['total_profit']:>+9.0f} "
                f"{r['roi']:>+7.1%}"
            )
            print(line)

        print("-" * 90)
        print("\nNotes:")
        for r in results:
            if 'note' in r:
                print(f"  - {r['strategy']}: {r['note']}")


def main():
    """Run historical backtest with all strategies."""
    logger.info("=" * 60)
    logger.info("HISTORICAL BETTING BACKTEST (V6 Models)")
    logger.info("=" * 60)

    backtester = StrategyBacktester(vig=0.08)
    test_df = backtester.load_test_data()

    # Run edge analysis first
    edge_df = backtester.run_edge_analysis(test_df)

    # Run all strategy backtests
    results = backtester.backtest_all(test_df)
    backtester.print_comparison_report(results)

    return backtester, edge_df


if __name__ == "__main__":
    backtester, edge_df = main()
