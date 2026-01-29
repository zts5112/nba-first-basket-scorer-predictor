#!/usr/bin/env python3
"""
NBA First Scorer Prediction Pipeline

Usage:
    python main.py collect --seasons 2021-22 2022-23 2023-24 2024-25
    python main.py process
    python main.py train
    python main.py predict --home LAL --away BOS
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
        from scrapers.bball_ref_scraper import BasketballReferenceScraper

        fetcher = BasketballReferenceScraper(
            data_dir=str(RAW_DATA_DIR),
            max_workers=args.workers,
            requests_per_second=0.5
        )

        max_games = args.max_games if args.test else None

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
    from utils.data_processing import JumpBallAggregator, FirstScorerAggregator
    
    logger.info("Processing raw data...")
    
    # Jump ball stats
    jb_agg = JumpBallAggregator(data_path=str(RAW_DATA_DIR))
    jb_agg.load_data()
    
    player_stats = jb_agg.compute_player_jump_ball_stats()
    player_stats.to_parquet(PROCESSED_DATA_DIR / 'player_jump_ball_stats.parquet')
    logger.info(f"Saved jump ball stats for {len(player_stats)} players")
    
    matchup_stats = jb_agg.compute_matchup_stats()
    matchup_stats.to_parquet(PROCESSED_DATA_DIR / 'matchup_stats.parquet')
    logger.info(f"Saved {len(matchup_stats)} head-to-head matchups")
    
    # First scorer stats
    fs_agg = FirstScorerAggregator(data_path=str(RAW_DATA_DIR))
    fs_agg.load_data()
    
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
        default=3,
        help='Number of parallel workers for scraping (max 5)'
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
    elif args.command == 'analyze':
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
