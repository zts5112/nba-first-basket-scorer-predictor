#!/usr/bin/env python3
"""
NBA First Scorer Prediction Pipeline

Usage:
    python main.py collect --seasons 2021-22 2022-23 2023-24 2024-25
    python main.py process
    python main.py train
    python main.py predict --home LAL --away BOS
    python main.py odds --home NYK --away MIL --show-browser
    python main.py ui
"""

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'


def cmd_collect(args):
    """Collect data from NBA API or Basketball Reference."""
    logger.info(f"Collecting data for seasons: {args.seasons}")
    logger.info(f"Source: {args.source}")

    if args.source == 'bball_ref':
        max_games = args.max_games if args.test else None

        if args.use_selenium:
            # Use Selenium with real Chrome browser
            from scrapers.bball_ref_selenium import SeleniumBasketballReferenceScraper

            logger.info("Using Selenium (real Chrome browser) for scraping")

            with SeleniumBasketballReferenceScraper(
                data_dir=str(RAW_DATA_DIR),
                headless=not args.show_browser,
                delay_between_requests=3.0,
                checkpoint_interval=25
            ) as fetcher:
                if args.test:
                    logger.info(f"Running in test mode - max {max_games} games")
                    df = fetcher.fetch_season_data(args.seasons[0], max_games=max_games)
                else:
                    df = fetcher.fetch_multiple_seasons(args.seasons)
        else:
            # Use requests-based scraper
            from scrapers.bball_ref_scraper import BasketballReferenceScraper

            fetcher = BasketballReferenceScraper(
                data_dir=str(RAW_DATA_DIR),
                max_workers=args.workers,
                requests_per_second=0.2,
                use_browser_cookies=args.use_cookies
            )

            if args.test:
                logger.info(f"Running in test mode - max {max_games} games")
                df = fetcher.fetch_season_data(args.seasons[0], max_games=max_games)
            else:
                df = fetcher.fetch_multiple_seasons(args.seasons)

        logger.info(f"Collection complete: {len(df)} games")
    else:
        from scrapers.nba_api_fetcher import NBADataFetcher

        fetcher = NBADataFetcher(data_dir=str(RAW_DATA_DIR))

        if args.test:
            logger.info("Running in test mode - fetching limited data")
            df = fetcher.fetch_season_data(args.seasons[0])
        else:
            df = fetcher.fetch_multiple_seasons(args.seasons)

        logger.info(f"Collection complete: {len(df)} games")

    return df


def cmd_process(args):
    """Process raw data into features."""
    from data.clean_data import DataCleaner
    from utils.data_processing import JumpBallAggregator, FirstScorerAggregator

    logger.info("Processing raw data...")

    # Step 1: Clean data
    logger.info("Step 1: Cleaning raw data...")
    cleaner = DataCleaner(raw_data_path=str(RAW_DATA_DIR), output_path=str(PROCESSED_DATA_DIR))
    cleaner.load_all_seasons()
    cleaner.clean_data()
    cleaner.save_cleaned_data()
    stats = cleaner.get_summary_stats()
    logger.info(f"Cleaned {stats['total_games']} games across {len(stats['seasons'])} seasons")

    # Column mapping from cleaned data to what aggregators expect
    col_renames = {
        'jump_ball_winner_name': 'jump_ball_winner',
        'jump_ball_loser_name': 'jump_ball_loser',
        'first_scorer_name': 'first_scorer',
        'jump_ball_player1_name': 'jump_ball_player1',
        'jump_ball_player2_name': 'jump_ball_player2',
    }

    # Step 2: Aggregate jump ball stats using cleaned data
    logger.info("Step 2: Aggregating jump ball stats...")
    jb_agg = JumpBallAggregator()
    jb_agg.raw_data = cleaner.cleaned_data.rename(columns=col_renames)

    player_stats = jb_agg.compute_player_jump_ball_stats()
    player_stats.to_parquet(PROCESSED_DATA_DIR / 'player_jump_ball_stats.parquet')
    logger.info(f"Saved jump ball stats for {len(player_stats)} players")

    matchup_stats = jb_agg.compute_matchup_stats()
    matchup_stats.to_parquet(PROCESSED_DATA_DIR / 'matchup_stats.parquet')
    logger.info(f"Saved {len(matchup_stats)} head-to-head matchups")

    # Step 3: Aggregate first scorer stats
    logger.info("Step 3: Aggregating first scorer stats...")
    fs_agg = FirstScorerAggregator()
    fs_agg.raw_data = cleaner.cleaned_data.rename(columns=col_renames)

    fs_rates = fs_agg.compute_first_scorer_rates()
    fs_rates.to_parquet(PROCESSED_DATA_DIR / 'first_scorer_rates.parquet')
    logger.info(f"Saved first scorer rates for {len(fs_rates)} players")

    # Tip to first score analysis
    tip_stats = fs_agg.compute_tip_to_first_score_rate()
    logger.info(f"Tip winner scores first: {tip_stats['tip_winner_first_score_rate']:.1%}")

    return player_stats, matchup_stats, fs_rates


def cmd_train(args):
    """Train prediction models."""
    from models.first_scorer_model import FirstScorerPredictor
    import pandas as pd
    
    logger.info("Training models...")
    
    # Load processed data
    game_data = pd.read_parquet(RAW_DATA_DIR / 'all_seasons_combined.parquet')
    
    # For jump ball model, need to create matchup training data
    # This is a simplified version - full implementation would be more sophisticated
    matchup_data = create_matchup_training_data(game_data)
    
    predictor = FirstScorerPredictor()
    predictor.fit(game_data, matchup_data)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(str(MODEL_DIR))
    
    logger.info(f"Models saved to {MODEL_DIR}")


def create_matchup_training_data(game_data):
    """Create training data for jump ball model from game data."""
    import pandas as pd
    
    records = []
    
    for _, game in game_data.iterrows():
        if pd.isna(game.get('jump_ball_winner')) or pd.isna(game.get('jump_ball_loser')):
            continue
        
        winner = game['jump_ball_winner']
        loser = game['jump_ball_loser']
        
        # Order alphabetically for consistency
        if winner < loser:
            p1, p2 = winner, loser
            p1_won = 1
        else:
            p1, p2 = loser, winner
            p1_won = 0
        
        records.append({
            'player1': p1,
            'player2': p2,
            'player1_won': p1_won,
            'player1_win_rate': 0.5,  # Will be computed properly with full data
            'player2_win_rate': 0.5,
            'player1_total_jb': 10,
            'player2_total_jb': 10,
        })
    
    return pd.DataFrame(records)


def cmd_predict(args):
    """Generate predictions for a game."""
    from models.first_scorer_model import FirstScorerPredictor
    from data.train_models_v6 import JumpBallModelV6, PlayerFirstScorerModelV6
    import __main__
    __main__.JumpBallModelV6 = JumpBallModelV6
    __main__.PlayerFirstScorerModelV6 = PlayerFirstScorerModelV6

    logger.info(f"Generating predictions for {args.home} vs {args.away}")

    predictor = FirstScorerPredictor()
    predictor.load(str(MODEL_DIR))
    
    # In production, would fetch actual lineups
    # For now, placeholder
    logger.info("Note: Using placeholder lineups. Integrate lineup API for production.")
    
    predictions = predictor.predict(
        home_team=args.home,
        away_team=args.away,
        home_starters=args.home_starters or ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        away_starters=args.away_starters or ['Player6', 'Player7', 'Player8', 'Player9', 'Player10'],
        home_center=args.home_center or 'Player1',
        away_center=args.away_center or 'Player6'
    )
    
    print("\nFirst Scorer Probabilities:")
    print("-" * 40)
    for player, prob in sorted(predictions.items(), key=lambda x: -x[1]):
        print(f"{player:25s} {prob:6.1%}")
    
    return predictions


def cmd_ui(args):
    """Launch Streamlit UI."""
    import subprocess
    
    ui_path = PROJECT_ROOT / 'ui' / 'app.py'
    logger.info(f"Launching UI from {ui_path}")
    
    subprocess.run(['streamlit', 'run', str(ui_path)])


def cmd_analyze(args):
    """Run analysis on collected data to assess predictability."""
    import pandas as pd
    from utils.data_processing import JumpBallAggregator, FirstScorerAggregator
    
    logger.info("Running predictability analysis...")
    
    # Load data
    try:
        game_data = pd.read_parquet(RAW_DATA_DIR / 'all_seasons_combined.parquet')
    except FileNotFoundError:
        logger.error("No data found. Run 'python main.py collect' first.")
        return
    
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total games: {len(game_data)}")
    
    # Jump ball analysis
    jb_valid = game_data[game_data['jump_ball_winner'].notna()]
    print(f"Games with jump ball data: {len(jb_valid)}")
    
    # First scorer analysis  
    fs_valid = game_data[game_data['first_scorer'].notna()]
    print(f"Games with first scorer data: {len(fs_valid)}")
    
    # Tip to first score
    tip_fs = game_data[game_data['tip_winner_scored_first'].notna()]
    if len(tip_fs) > 0:
        tip_rate = tip_fs['tip_winner_scored_first'].mean()
        print(f"\nTip winner scores first: {tip_rate:.1%} ({len(tip_fs)} games)")
        print(f"  -> Baseline edge from winning tip: {(tip_rate - 0.5)*100:.1f}%")
    
    # Jump ball predictability
    print("\n" + "=" * 60)
    print("JUMP BALL PREDICTABILITY")
    print("=" * 60)
    
    jb_agg = JumpBallAggregator()
    jb_agg.raw_data = game_data
    player_stats = jb_agg.compute_player_jump_ball_stats()
    
    # Players with enough data
    experienced = player_stats[player_stats['total_jump_balls'] >= 20]
    print(f"Players with 20+ jump balls: {len(experienced)}")
    
    # Win rate distribution
    print(f"\nWin rate distribution (players with 20+ JBs):")
    print(f"  Min: {experienced['win_rate'].min():.1%}")
    print(f"  25%: {experienced['win_rate'].quantile(0.25):.1%}")
    print(f"  50%: {experienced['win_rate'].quantile(0.50):.1%}")
    print(f"  75%: {experienced['win_rate'].quantile(0.75):.1%}")
    print(f"  Max: {experienced['win_rate'].max():.1%}")
    
    # Top jump ball winners
    print(f"\nTop 10 jump ball winners (min 20 attempts):")
    top_jb = experienced.nlargest(10, 'win_rate')[['player_name', 'wins', 'losses', 'win_rate']]
    for _, row in top_jb.iterrows():
        print(f"  {row['player_name']:25s} {row['wins']:3.0f}-{row['losses']:3.0f} ({row['win_rate']:.1%})")
    
    # First scorer analysis
    print("\n" + "=" * 60)
    print("FIRST SCORER PREDICTABILITY")
    print("=" * 60)
    
    fs_agg = FirstScorerAggregator()
    fs_agg.raw_data = game_data
    fs_rates = fs_agg.compute_first_scorer_rates()
    
    # Score type distribution
    score_dist = fs_agg.compute_score_type_distribution()
    print("\nFirst score type distribution:")
    for _, row in score_dist.iterrows():
        print(f"  {row['score_type']:5s}: {row['percentage']:.1%}")
    
    # Top first scorers
    top_fs = fs_rates.nlargest(10, 'times_scored_first')
    print(f"\nMost frequent first scorers:")
    for _, row in top_fs.iterrows():
        print(f"  {row['player_name']:25s} {row['times_scored_first']:3.0f} times ({row['first_scorer_rate']:.1%})")
    
    # Entropy analysis (how predictable is first scorer?)
    print("\n" + "=" * 60)
    print("PREDICTABILITY ASSESSMENT")
    print("=" * 60)
    
    # If first scorer was random among 10 starters, each has 10% chance
    # Entropy of uniform: -10 * 0.1 * log(0.1) = 2.30
    # Lower entropy = more predictable
    
    fs_probs = fs_rates['first_scorer_rate'].values
    fs_probs = fs_probs[fs_probs > 0]
    fs_probs = fs_probs / fs_probs.sum()  # Normalize
    
    import numpy as np
    entropy = -np.sum(fs_probs * np.log(fs_probs + 1e-10))
    uniform_entropy = np.log(10)  # 10 starters
    
    print(f"First scorer entropy: {entropy:.2f}")
    print(f"Uniform (random) entropy: {uniform_entropy:.2f}")
    print(f"Predictability gain: {(1 - entropy/uniform_entropy)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if tip_rate and tip_rate > 0.52:
        print("✅ Jump ball winner has meaningful edge for first score")
    else:
        print("⚠️  Jump ball edge is marginal")
    
    if experienced['win_rate'].std() > 0.1:
        print("✅ Jump ball outcomes show player skill differentiation")
    else:
        print("⚠️  Jump ball outcomes appear largely random")
    
    print("\nNext steps:")
    print("1. Train models: python main.py train")
    print("2. Backtest on held-out data")
    print("3. Compare model predictions to betting lines")


def cmd_odds(args):
    """Fetch first basket scorer odds."""
    import os

    logger.info("Fetching first basket scorer odds...")

    # Determine API key
    api_key = getattr(args, 'api_key', None) or os.environ.get('ODDS_API_KEY', '')

    if not api_key:
        print("\nNo API key provided. To use The Odds API (recommended):")
        print("  1. Get a free key at https://the-odds-api.com/")
        print("  2. Run: export ODDS_API_KEY=your_key_here")
        print("  3. Or: python main.py odds --api-key your_key_here")
        print("\n  Free tier: 500 credits/month (more than enough for daily use)")
        return

    from scrapers.odds_api import OddsAPIFetcher

    bookmaker = getattr(args, 'bookmaker', 'fanduel')
    fetcher = OddsAPIFetcher(
        api_key=api_key,
        bookmaker=bookmaker,
        cache_dir=str(DATA_DIR / 'odds_cache'),
    )

    if getattr(args, 'no_cache', False):
        fetcher.cache_ttl = 0

    if args.home and args.away:
        # Fetch odds for a specific game
        odds = fetcher.get_odds_for_game(args.home, args.away)
        if not odds:
            logger.error(f"Could not find first basket scorer odds for {args.away} @ {args.home}")
            return

        _print_odds_table(args.away, args.home, odds)

        # Optionally run predictions + betting analysis
        if args.predict:
            if not (args.home_starters and args.away_starters and args.home_center and args.away_center):
                logger.error("--predict requires --home-starters, --away-starters, --home-center, --away-center")
                return
            _run_prediction_with_odds(args, odds)
    else:
        # Fetch odds for all today's games
        all_odds = fetcher.get_all_games_odds()
        if not all_odds:
            logger.error("No NBA games or odds found")
            return

        for game_key, odds in all_odds.items():
            parts = game_key.split(" @ ")
            away = parts[0] if len(parts) == 2 else game_key
            home = parts[1] if len(parts) == 2 else ""
            _print_odds_table(away, home, odds)
            print()


def _print_odds_table(away: str, home: str, odds: dict):
    """Pretty-print fetched odds."""
    print(f"\nFirst Basket Scorer Odds: {away} @ {home}")
    print("-" * 50)
    for player, american in sorted(odds.items(), key=lambda x: x[1]):
        print(f"  {player:<30} {american:+d}")
    print(f"\n  ({len(odds)} players)")


def _run_prediction_with_odds(args, market_odds: dict):
    """Run prediction pipeline with live odds."""
    # Models were pickled from __main__, so pickle looks for the classes there.
    # Import them and inject into the current __main__ module.
    from data.train_models_v6 import JumpBallModelV6, PlayerFirstScorerModelV6
    import __main__
    __main__.JumpBallModelV6 = JumpBallModelV6
    __main__.PlayerFirstScorerModelV6 = PlayerFirstScorerModelV6

    from inference.predict import FirstScorerPredictor
    from betting.alternative_strategies import print_strategy_analysis
    from scrapers.player_name_matcher import PlayerNameMatcher

    logger.info("Running prediction model...")

    predictor = FirstScorerPredictor()
    prediction = predictor.predict(
        home_team=args.home,
        away_team=args.away,
        home_starters=args.home_starters,
        away_starters=args.away_starters,
        home_center=args.home_center,
        away_center=args.away_center,
    )
    prediction.print_prediction()

    # Match FanDuel names to model names
    matcher = PlayerNameMatcher()
    matched_odds, unmatched = matcher.match_odds_dict(market_odds)

    if unmatched:
        print(f"\nNote: Could not match {len(unmatched)} FanDuel names: {unmatched}")

    # Build model_probs dict
    model_probs = {
        p['player_name']: p['probability']
        for p in prediction.player_probabilities
    }

    # Build player_ppg from predictor's stats
    player_ppg = {}
    for p in prediction.player_probabilities:
        name = p['player_name']
        stats = predictor._get_player_api_stats(name)
        if stats and stats.get('ppg', 0) > 0:
            player_ppg[name] = stats['ppg']

    # Run strategy analysis
    budget = getattr(args, 'budget', 30.0)
    print_strategy_analysis(
        model_probs=model_probs,
        market_odds=matched_odds,
        player_ppg=player_ppg,
        home_team=args.home,
        away_team=args.away,
        home_jb_win_prob=prediction.home_wins_tip_prob,
        budget=budget,
    )


def main():
    parser = argparse.ArgumentParser(description='NBA First Scorer Prediction Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from NBA API or Basketball Reference')
    collect_parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2021-22', '2022-23', '2023-24', '2024-25'],
        help='Seasons to collect (e.g., 2023-24)'
    )
    collect_parser.add_argument(
        '--source',
        choices=['nba_api', 'bball_ref'],
        default='bball_ref',
        help='Data source (default: bball_ref)'
    )
    collect_parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel workers for scraping (max 5, lower is safer)'
    )
    collect_parser.add_argument(
        '--max-games',
        type=int,
        default=10,
        help='Max games to scrape in test mode'
    )
    collect_parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with limited data'
    )
    collect_parser.add_argument(
        '--use-cookies',
        action='store_true',
        help='Use browser cookies to bypass rate limits (requires visiting basketball-reference.com first)'
    )
    collect_parser.add_argument(
        '--use-selenium',
        action='store_true',
        help='Use Selenium with real Chrome browser (recommended for bypassing rate limits)'
    )
    collect_parser.add_argument(
        '--show-browser',
        action='store_true',
        help='Show the Chrome browser window (only with --use-selenium, useful for debugging)'
    )

    # Process command
    subparsers.add_parser('process', help='Process raw data into features')
    
    # Train command
    subparsers.add_parser('train', help='Train prediction models')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--home', required=True, help='Home team abbreviation')
    predict_parser.add_argument('--away', required=True, help='Away team abbreviation')
    predict_parser.add_argument('--home-starters', nargs=5, help='Home starting lineup')
    predict_parser.add_argument('--away-starters', nargs=5, help='Away starting lineup')
    predict_parser.add_argument('--home-center', help='Home player taking tip')
    predict_parser.add_argument('--away-center', help='Away player taking tip')
    
    # UI command
    subparsers.add_parser('ui', help='Launch Streamlit UI')
    
    # Odds command
    odds_parser = subparsers.add_parser('odds', help='Fetch first basket scorer odds from The Odds API')
    odds_parser.add_argument('--home', help='Home team abbreviation (e.g., NYK)')
    odds_parser.add_argument('--away', help='Away team abbreviation (e.g., MIL)')
    odds_parser.add_argument('--api-key', help='The Odds API key (or set ODDS_API_KEY env var)')
    odds_parser.add_argument('--bookmaker', default='fanduel',
                             help='Sportsbook to get odds from (default: fanduel)')
    odds_parser.add_argument('--no-cache', action='store_true',
                             help='Force fresh fetch, ignore cached odds')
    odds_parser.add_argument('--predict', action='store_true',
                             help='Also run prediction model and show value bets')
    odds_parser.add_argument('--home-starters', nargs=5,
                             help='Home starting lineup (5 names, required with --predict)')
    odds_parser.add_argument('--away-starters', nargs=5,
                             help='Away starting lineup (5 names, required with --predict)')
    odds_parser.add_argument('--home-center', help='Home jump ball player (required with --predict)')
    odds_parser.add_argument('--away-center', help='Away jump ball player (required with --predict)')
    odds_parser.add_argument('--budget', type=float, default=30.0,
                             help='Betting budget for recommendations (default: $30)')

    # Analyze command
    subparsers.add_parser('analyze', help='Analyze data for predictability')
    
    args = parser.parse_args()
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dispatch command
    if args.command == 'collect':
        cmd_collect(args)
    elif args.command == 'process':
        cmd_process(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'ui':
        cmd_ui(args)
    elif args.command == 'odds':
        cmd_odds(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
