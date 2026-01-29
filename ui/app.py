"""
Streamlit UI for First Scorer Prediction

Run with: streamlit run ui/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Conditional imports - will work once models are trained
try:
    from models.first_scorer_model import FirstScorerPredictor
    from models.betting_optimizer import (
        BettingLine, MultiOutcomeBetOptimizer, 
        calculate_total_vig, parse_odds_from_sportsbook
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# Team abbreviations and colors
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}


def create_demo_predictions(home_starters, away_starters):
    """Create demo predictions when model isn't loaded."""
    all_players = home_starters + away_starters
    
    # Simple heuristic: distribute probability somewhat evenly
    # with slight bias to certain positions
    probs = np.random.dirichlet(np.ones(10) * 2)
    
    predictions = dict(zip(all_players, probs * 0.95))
    predictions['Other'] = 0.05
    
    return predictions


def main():
    st.set_page_config(
        page_title="NBA First Scorer Predictor",
        page_icon="🏀",
        layout="wide"
    )
    
    st.title("🏀 NBA First Scorer Predictor")
    st.markdown("*Predict who will score first and find value bets*")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    bankroll = st.sidebar.number_input(
        "Bankroll ($)", 
        min_value=100, 
        max_value=100000, 
        value=1000,
        step=100
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more conservative betting"
    )
    
    min_edge = st.sidebar.slider(
        "Minimum Edge to Bet",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Only bet when our edge exceeds this"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox(
            "Select Home Team",
            options=list(NBA_TEAMS.keys()),
            format_func=lambda x: f"{x} - {NBA_TEAMS[x]}",
            key="home_team"
        )
        
        st.write("**Starting Lineup:**")
        home_starters = []
        for i in range(5):
            player = st.text_input(
                f"Starter {i+1}", 
                key=f"home_{i}",
                placeholder=f"Home Player {i+1}"
            )
            if player:
                home_starters.append(player)
        
        home_center = st.selectbox(
            "Who takes the tip?",
            options=home_starters if home_starters else [""],
            key="home_center"
        )
    
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox(
            "Select Away Team",
            options=list(NBA_TEAMS.keys()),
            format_func=lambda x: f"{x} - {NBA_TEAMS[x]}",
            key="away_team"
        )
        
        st.write("**Starting Lineup:**")
        away_starters = []
        for i in range(5):
            player = st.text_input(
                f"Starter {i+1}", 
                key=f"away_{i}",
                placeholder=f"Away Player {i+1}"
            )
            if player:
                away_starters.append(player)
        
        away_center = st.selectbox(
            "Who takes the tip?",
            options=away_starters if away_starters else [""],
            key="away_center"
        )
    
    st.divider()
    
    # Betting odds input
    st.subheader("💰 Enter Betting Odds")
    st.markdown("*Enter American odds (e.g., +450, -110) for each player*")
    
    all_players = home_starters + away_starters
    odds_input = {}
    
    if all_players:
        odds_cols = st.columns(2)
        
        with odds_cols[0]:
            st.write("**Home Team Odds:**")
            for player in home_starters:
                odds = st.text_input(
                    f"{player}",
                    value="+500",
                    key=f"odds_{player}"
                )
                odds_input[player] = odds
        
        with odds_cols[1]:
            st.write("**Away Team Odds:**")
            for player in away_starters:
                odds = st.text_input(
                    f"{player}",
                    value="+500",
                    key=f"odds_{player}"
                )
                odds_input[player] = odds
    
    st.divider()
    
    # Prediction button
    if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
        if len(home_starters) != 5 or len(away_starters) != 5:
            st.error("Please enter 5 starters for each team")
        else:
            with st.spinner("Calculating predictions..."):
                # Get predictions (demo for now)
                predictions = create_demo_predictions(home_starters, away_starters)
                
                # Parse betting lines
                betting_lines = {}
                for player, odds_str in odds_input.items():
                    try:
                        odds = parse_odds_from_sportsbook(odds_str)
                        betting_lines[player] = BettingLine(player, odds)
                    except:
                        st.warning(f"Could not parse odds for {player}: {odds_str}")
                
                # Display predictions
                st.subheader("📊 Predictions")
                
                pred_df = pd.DataFrame([
                    {
                        'Player': player,
                        'Team': home_team if player in home_starters else (away_team if player in away_starters else '-'),
                        'Model Probability': prob,
                        'Implied Prob': betting_lines[player].implied_prob if player in betting_lines else None,
                        'Edge': prob - betting_lines[player].implied_prob if player in betting_lines else None
                    }
                    for player, prob in predictions.items()
                    if player != 'Other'
                ])
                
                pred_df = pred_df.sort_values('Model Probability', ascending=False)
                
                # Format as percentages
                pred_df['Model Probability'] = pred_df['Model Probability'].apply(lambda x: f"{x:.1%}")
                pred_df['Implied Prob'] = pred_df['Implied Prob'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
                pred_df['Edge'] = pred_df['Edge'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
                
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Find value bets
                st.subheader("💎 Value Bets")
                
                optimizer = MultiOutcomeBetOptimizer(
                    max_bet_fraction=kelly_fraction,
                    min_edge=min_edge
                )
                
                # Need numeric probs for optimizer
                numeric_preds = {k: v for k, v in predictions.items() if k != 'Other'}
                
                recommendations = optimizer.generate_bet_recommendations(
                    numeric_preds,
                    betting_lines,
                    bankroll
                )
                
                if recommendations.empty:
                    st.info("No value bets found with current settings. Try lowering minimum edge.")
                else:
                    # Display recommendations
                    display_cols = ['player', 'model_prob', 'implied_prob', 'edge', 
                                   'american_odds', 'ev_per_dollar', 'recommended_bet', 'potential_profit']
                    
                    rec_display = recommendations[display_cols].copy()
                    rec_display.columns = ['Player', 'Model Prob', 'Implied Prob', 'Edge',
                                          'Odds', 'EV/$', 'Bet Amount', 'Potential Profit']
                    
                    # Format
                    rec_display['Model Prob'] = rec_display['Model Prob'].apply(lambda x: f"{x:.1%}")
                    rec_display['Implied Prob'] = rec_display['Implied Prob'].apply(lambda x: f"{x:.1%}")
                    rec_display['Edge'] = rec_display['Edge'].apply(lambda x: f"{x:+.1%}")
                    rec_display['Odds'] = rec_display['Odds'].apply(lambda x: f"{x:+d}")
                    rec_display['EV/$'] = rec_display['EV/$'].apply(lambda x: f"${x:.3f}")
                    rec_display['Bet Amount'] = rec_display['Bet Amount'].apply(lambda x: f"${x:.2f}")
                    rec_display['Potential Profit'] = rec_display['Potential Profit'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(rec_display, use_container_width=True, hide_index=True)
                    
                    # Summary metrics
                    total_bet = recommendations['recommended_bet'].sum()
                    total_ev = recommendations['expected_profit'].sum()
                    
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Total Bet Amount", f"${total_bet:.2f}")
                    metric_cols[1].metric("Expected Profit", f"${total_ev:.2f}")
                    metric_cols[2].metric("ROI", f"{(total_ev/total_bet*100):.1f}%" if total_bet > 0 else "N/A")
                
                # Vig analysis
                if betting_lines:
                    vig = calculate_total_vig(list(betting_lines.values()))
                    st.caption(f"📈 Market Vig: {vig:.1%}")
    
    # Footer with info
    st.divider()
    st.markdown("""
    ### How it works
    
    1. **Jump Ball Model**: Predicts who wins the opening tip based on historical matchup data
    2. **Team Model**: Predicts which team scores first given the tip result
    3. **Player Model**: Predicts which starter scores first given their team has possession
    
    The betting optimizer uses a modified Kelly Criterion for mutually exclusive outcomes 
    to recommend optimal bet sizes that maximize long-term growth while managing risk.
    
    ---
    
    ⚠️ **Disclaimer**: This tool is for educational purposes only. Gambling involves risk. 
    Past performance does not guarantee future results. Please gamble responsibly.
    """)


if __name__ == "__main__":
    main()
