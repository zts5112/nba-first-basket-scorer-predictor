# NBA First Scorer Predictor 🏀

Predict which player will score first in NBA games using jump ball data, player tendencies, and machine learning. Includes betting optimization with Kelly Criterion.

## Project Overview

This project attempts to gain an edge in the "first scorer" betting market by:

1. **Modeling jump ball outcomes** - Predict who wins the opening tip based on historical matchups
2. **Modeling team first-possession scoring** - Teams that win the tip score first ~53% of the time
3. **Modeling individual first scorer rates** - Some players are more likely to take/make the first shot
4. **Optimizing bet sizing** - Kelly Criterion for correlated outcomes

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/nba-first-scorer.git
cd nba-first-scorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Collect data (this takes a while - ~1 request/second due to rate limiting)
python main.py collect --seasons 2023-24 2024-25

# 2. Analyze the data to assess predictability
python main.py analyze

# 3. Process into features
python main.py process

# 4. Train models
python main.py train

# 5. Launch the UI
python main.py ui
```

## Project Structure

```
nba_first_scorer/
├── main.py                     # CLI orchestration
├── requirements.txt            # Dependencies
├── README.md
├── data/
│   ├── raw/                    # Raw game data from NBA API
│   └── processed/              # Processed features
├── models/                     # Trained model files
├── src/
│   ├── scrapers/
│   │   └── nba_api_fetcher.py  # NBA data collection
│   ├── models/
│   │   ├── first_scorer_model.py   # Prediction models
│   │   └── betting_optimizer.py    # Kelly criterion & optimization
│   └── utils/
│       └── data_processing.py      # Feature engineering
└── ui/
    └── app.py                  # Streamlit interface
```

## Data Sources

- **Primary**: `nba_api` - Python wrapper for stats.nba.com
  - Play-by-play data contains jump ball and first basket events
  - ~0.6 second delay between requests to respect rate limits

## Key Insights

From initial analysis of NBA data:

- **Tip winner advantage**: Team that wins the tip scores first ~53% of the time (vs 50% baseline)
- **Jump ball skill**: Win rates range from ~35% to ~70% for players with significant sample sizes
- **Position matters**: Centers take most tips, but PGs often score first due to ball handling
- **First score types**: ~70% 2PT, ~25% 3PT, ~5% FT

## Model Architecture

### Stage 1: Jump Ball Model
- Features: Historical win rates, head-to-head record, height/reach (if available)
- Output: Probability of home center winning tip

### Stage 2: Team First Score Model  
- Features: Tip win probability, team offensive tendencies
- Output: Probability each team scores first

### Stage 3: Player First Scorer Model
- Features: Individual first scorer rates, position, team context
- Output: Probability each starter scores first

### Betting Optimizer
- Multi-outcome Kelly Criterion for mutually exclusive events
- Accounts for correlation (only one player can score first)
- Configurable risk parameters (Kelly fraction, min edge)

## Usage Examples

### Command Line

```bash
# Collect specific seasons
python main.py collect --seasons 2022-23 2023-24

# Run predictability analysis
python main.py analyze

# Generate predictions for a game
python main.py predict --home LAL --away BOS \
    --home-starters "LeBron James" "Anthony Davis" "Austin Reaves" "D'Angelo Russell" "Rui Hachimura" \
    --away-starters "Jayson Tatum" "Jaylen Brown" "Derrick White" "Jrue Holiday" "Al Horford" \
    --home-center "Anthony Davis" \
    --away-center "Al Horford"
```

### Python API

```python
from src.models.first_scorer_model import FirstScorerPredictor
from src.models.betting_optimizer import MultiOutcomeBetOptimizer, BettingLine

# Load trained model
predictor = FirstScorerPredictor()
predictor.load('models/')

# Get predictions
probs = predictor.predict(
    home_team='LAL',
    away_team='BOS',
    home_starters=['LeBron James', 'Anthony Davis', ...],
    away_starters=['Jayson Tatum', 'Jaylen Brown', ...],
    home_center='Anthony Davis',
    away_center='Al Horford'
)

# Find value bets
lines = {
    'LeBron James': BettingLine('LeBron James', +400),
    'Jayson Tatum': BettingLine('Jayson Tatum', +350),
    # ...
}

optimizer = MultiOutcomeBetOptimizer(max_bet_fraction=0.25, min_edge=0.02)
recommendations = optimizer.generate_bet_recommendations(probs, lines, bankroll=1000)
print(recommendations)
```

## Limitations & Caveats

1. **Data latency**: Starting lineups announced ~30 min before tip-off
2. **Sample sizes**: Head-to-head matchups may have small samples
3. **Market efficiency**: Sportsbooks are sophisticated; edges are thin
4. **Model assumptions**: Independence assumptions may not hold perfectly
5. **Historical data**: Past performance doesn't guarantee future results

## Future Improvements

- [ ] Real-time lineup integration (FantasyLabs, RotoGrinders APIs)
- [ ] Player height/reach/vertical data for better jump ball predictions
- [ ] Team play-calling tendencies on first possession
- [ ] Live odds scraping for automated value detection
- [ ] Backtesting framework with realistic bet execution
- [ ] Player injury/rest adjustments

## Disclaimer

⚠️ **This project is for educational and research purposes only.**

- Gambling involves significant financial risk
- Past model performance does not guarantee future results
- Always gamble responsibly and within your means
- Check local laws regarding sports betting

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.
