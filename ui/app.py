"""
Streamlit UI for NBA First Basket Scorer Prediction

Run with: streamlit run ui/app.py
Or: python main.py ui
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Inject V6 model classes for pickle compatibility
from data.train_models_v6 import JumpBallModelV6, PlayerFirstScorerModelV6
import __main__
__main__.JumpBallModelV6 = JumpBallModelV6
__main__.PlayerFirstScorerModelV6 = PlayerFirstScorerModelV6

from inference.predict import FirstScorerPredictor
from betting.alternative_strategies import (
    AlternativeStrategies, BetRecommendation, _american_to_decimal
)

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

STRATEGY_LABELS = {
    'VALUE_BET': 'Value Bet',
    'EFFICIENCY_PLAY': 'Efficiency Play',
    'CONTRARIAN_LONGSHOT': 'Contrarian Longshot',
    'SECOND_CHOICE': 'Second Choice',
    'TIP_WINNER_TEAM': 'Tip Winner Team',
    'TEAM_FIRST_SCORER': 'Team First Scorer',
    'MULTI_PLAYER_HEDGE': 'Multi-Player Hedge',
    'SCORE_TYPE_FILTER': 'Score Type Filter',
    'CORRELATED_PARLAY': 'Correlated Parlay',
}

CONFIDENCE_COLORS = {
    'HIGH': '#22c55e',
    'MEDIUM': '#f59e0b',
    'LOW': '#ef4444',
}


@st.cache_resource
def load_predictor():
    """Load the V6 prediction model (cached across reruns)."""
    return FirstScorerPredictor(
        model_dir=str(PROJECT_ROOT / 'models'),
        data_dir=str(PROJECT_ROOT / 'data' / 'processed'),
    )


@st.cache_data(ttl=1800)
def fetch_todays_games(api_key, bookmaker='fanduel'):
    """Fetch today's NBA games from The Odds API (cached 30 min)."""
    from scrapers.odds_api import OddsAPIFetcher
    fetcher = OddsAPIFetcher(
        api_key=api_key,
        bookmaker=bookmaker,
        cache_dir=str(PROJECT_ROOT / 'data' / 'odds_cache'),
    )
    return fetcher.get_nba_games()


@st.cache_data(ttl=1800)
def fetch_game_odds(api_key, home_team, away_team, bookmaker='fanduel'):
    """Fetch first basket scorer odds for a specific game (cached 30 min)."""
    from scrapers.odds_api import OddsAPIFetcher
    fetcher = OddsAPIFetcher(
        api_key=api_key,
        bookmaker=bookmaker,
        cache_dir=str(PROJECT_ROOT / 'data' / 'odds_cache'),
    )
    return fetcher.get_odds_for_game(home_team, away_team)


@st.cache_data(ttl=900)
def fetch_team_starters(team_abbrev):
    """Fetch projected starters from ESPN depth charts (cached 15 min)."""
    from scrapers.espn_lineups import ESPNLineupFetcher
    fetcher = ESPNLineupFetcher()
    result = fetcher.get_starters(team_abbrev)
    if result:
        # Convert StarterInfo dataclasses to dicts for Streamlit cache serialization
        return {
            'players': [
                {
                    'name': p.name,
                    'position': p.position,
                    'injury_status': p.injury_status,
                    'injury_comment': p.injury_comment,
                }
                for p in result['players']
            ],
            'center': result['center'],
            'team': result['team'],
        }
    return None


def match_odds_names(odds_dict):
    """Match FanDuel player names to model canonical names."""
    from scrapers.player_name_matcher import PlayerNameMatcher
    matcher = PlayerNameMatcher(
        tokenizer_path=str(PROJECT_ROOT / 'data' / 'processed' / 'player_tokenizer.json')
    )
    matched, unmatched = matcher.match_odds_dict(odds_dict)
    return matched, unmatched


def main():
    st.set_page_config(
        page_title="NBA First Scorer Predictor",
        page_icon="🏀",
        layout="wide"
    )

    st.title("🏀 NBA First Basket Scorer Predictor")
    st.caption("Model predictions + betting strategy analysis")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "The Odds API Key",
            type="password",
            help="Get a free key at the-odds-api.com (500 credits/month)",
            value=st.session_state.get('api_key', ''),
        )
        if api_key:
            st.session_state['api_key'] = api_key

        bookmaker = st.selectbox(
            "Sportsbook",
            options=['fanduel', 'draftkings', 'betmgm', 'pointsbet'],
            format_func=lambda x: x.replace('_', ' ').title(),
        )

        budget = st.number_input(
            "Betting Budget ($)",
            min_value=5,
            max_value=500,
            value=30,
            step=5,
        )

        st.divider()
        st.caption("Model: XGBoost V6")
        st.caption("Top-1 accuracy: 14.2%")
        st.caption("Top-3 accuracy: 37.2%")

    # --- Game Selection ---
    st.subheader("Select Game")

    game_mode = st.radio(
        "How do you want to set up the game?",
        ["Today's Games (auto-fetch odds)", "Manual Entry"],
        horizontal=True,
    )

    home_team = None
    away_team = None
    live_odds = None

    if game_mode == "Today's Games (auto-fetch odds)":
        if not api_key:
            st.info("Enter your Odds API key in the sidebar to fetch today's games.")
        else:
            games = fetch_todays_games(api_key, bookmaker)
            if not games:
                st.warning("No upcoming NBA games found.")
            else:
                game_options = {
                    f"{g['away_team']} @ {g['home_team']}": g for g in games
                }
                selected_game = st.selectbox(
                    "Select a game",
                    options=list(game_options.keys()),
                )
                if selected_game:
                    game = game_options[selected_game]
                    # Parse team abbreviations from the API names
                    from scrapers.fanduel_odds import normalize_team
                    home_team = normalize_team(game['home_team'])
                    away_team = normalize_team(game['away_team'])

                    # Fetch odds
                    with st.spinner("Fetching odds..."):
                        raw_odds = fetch_game_odds(api_key, home_team, away_team, bookmaker)

                    if raw_odds:
                        live_odds = raw_odds
                        st.success(f"Loaded odds for {len(raw_odds)} players from {bookmaker.title()}")
                    else:
                        st.warning("First basket scorer odds not available yet for this game.")
    else:
        col_h, col_a = st.columns(2)
        with col_h:
            home_team = st.selectbox(
                "Home Team",
                options=list(NBA_TEAMS.keys()),
                format_func=lambda x: f"{x} - {NBA_TEAMS[x]}",
                key="manual_home",
            )
        with col_a:
            away_team = st.selectbox(
                "Away Team",
                options=list(NBA_TEAMS.keys()),
                format_func=lambda x: f"{x} - {NBA_TEAMS[x]}",
                key="manual_away",
                index=1,
            )

    if not home_team or not away_team:
        st.stop()

    st.divider()

    # --- Starters Input ---
    st.subheader(f"Starting Lineups: {away_team} @ {home_team}")

    # Fetch projected lineups from ESPN
    home_lineup = fetch_team_starters(home_team)
    away_lineup = fetch_team_starters(away_team)

    # Detect team changes and write new ESPN defaults into session state
    if st.session_state.get('_last_home_team') != home_team:
        for i in range(5):
            name = home_lineup['players'][i]['name'] if home_lineup and i < len(home_lineup['players']) else ""
            st.session_state[f'home_starter_{i}'] = name
        # Set tip-off to center
        center = home_lineup['center'] if home_lineup else ""
        st.session_state['home_tip'] = center
        st.session_state['_last_home_team'] = home_team

    if st.session_state.get('_last_away_team') != away_team:
        for i in range(5):
            name = away_lineup['players'][i]['name'] if away_lineup and i < len(away_lineup['players']) else ""
            st.session_state[f'away_starter_{i}'] = name
        # Set tip-off to center
        center = away_lineup['center'] if away_lineup else ""
        st.session_state['away_tip'] = center
        st.session_state['_last_away_team'] = away_team

    if home_lineup and away_lineup:
        st.success("Projected starters loaded from ESPN depth charts. Edit any name if needed.")
    elif home_lineup or away_lineup:
        st.warning("Could only load starters for one team. Enter the other manually.")
    else:
        st.info("Could not fetch projected starters. Enter lineups manually below.")

    col_home, col_away = st.columns(2)

    home_starters = []
    away_starters = []
    pos_labels = ['PG', 'SG', 'SF', 'PF', 'C']

    with col_home:
        st.markdown(f"**{home_team} - {NBA_TEAMS.get(home_team, 'Home')}**")
        for i in range(5):
            # Set initial value if not already in session state
            if f"home_starter_{i}" not in st.session_state:
                default_name = ""
                if home_lineup and i < len(home_lineup['players']):
                    default_name = home_lineup['players'][i]['name']
                st.session_state[f"home_starter_{i}"] = default_name

            injury_info = None
            if home_lineup and i < len(home_lineup['players']):
                p = home_lineup['players'][i]
                if p['injury_status'] and st.session_state[f"home_starter_{i}"] == p['name']:
                    injury_info = f"{p['injury_status']}: {p['injury_comment'] or 'Unknown'}"

            player = st.text_input(
                pos_labels[i],
                key=f"home_starter_{i}",
                placeholder=f"Home starter {i+1}",
            )
            if injury_info:
                st.caption(f"⚠️ {injury_info}")
            if player.strip():
                home_starters.append(player.strip())

        # Default tip-off to center
        default_tip_idx = min(4, len(home_starters) - 1) if home_starters else 0
        if home_lineup and home_lineup['center']:
            for idx, name in enumerate(home_starters):
                if name == home_lineup['center']:
                    default_tip_idx = idx
                    break

        home_center = st.selectbox(
            "Tip-off player",
            options=home_starters if home_starters else [""],
            key="home_tip",
            index=default_tip_idx,
        )

    with col_away:
        st.markdown(f"**{away_team} - {NBA_TEAMS.get(away_team, 'Away')}**")
        for i in range(5):
            # Set initial value if not already in session state
            if f"away_starter_{i}" not in st.session_state:
                default_name = ""
                if away_lineup and i < len(away_lineup['players']):
                    default_name = away_lineup['players'][i]['name']
                st.session_state[f"away_starter_{i}"] = default_name

            injury_info = None
            if away_lineup and i < len(away_lineup['players']):
                p = away_lineup['players'][i]
                if p['injury_status'] and st.session_state[f"away_starter_{i}"] == p['name']:
                    injury_info = f"{p['injury_status']}: {p['injury_comment'] or 'Unknown'}"

            player = st.text_input(
                pos_labels[i],
                key=f"away_starter_{i}",
                placeholder=f"Away starter {i+1}",
            )
            if injury_info:
                st.caption(f"⚠️ {injury_info}")
            if player.strip():
                away_starters.append(player.strip())

        # Default tip-off to center
        default_tip_idx = min(4, len(away_starters) - 1) if away_starters else 0
        if away_lineup and away_lineup['center']:
            for idx, name in enumerate(away_starters):
                if name == away_lineup['center']:
                    default_tip_idx = idx
                    break

        away_center = st.selectbox(
            "Tip-off player",
            options=away_starters if away_starters else [""],
            key="away_tip",
            index=default_tip_idx,
        )

    # --- Odds Display ---
    if live_odds:
        st.divider()
        st.subheader("First Basket Scorer Odds")
        odds_df = pd.DataFrame([
            {'Player': p, 'Odds': f"+{o}" if o > 0 else str(o)}
            for p, o in sorted(live_odds.items(), key=lambda x: x[1])
        ])
        st.dataframe(odds_df, use_container_width=True, hide_index=True)

    # --- Manual Odds Input (if no live odds) ---
    manual_odds = {}
    if not live_odds and game_mode == "Manual Entry":
        all_players = home_starters + away_starters
        if all_players:
            with st.expander("Enter Odds (optional)", expanded=False):
                odds_cols = st.columns(2)
                with odds_cols[0]:
                    for player in home_starters:
                        val = st.text_input(player, value="", key=f"odds_{player}", placeholder="+500")
                        if val.strip():
                            try:
                                manual_odds[player] = int(val.replace('+', ''))
                            except ValueError:
                                pass
                with odds_cols[1]:
                    for player in away_starters:
                        val = st.text_input(player, value="", key=f"odds_{player}", placeholder="+500")
                        if val.strip():
                            try:
                                manual_odds[player] = int(val.replace('+', ''))
                            except ValueError:
                                pass

    st.divider()

    # --- Analyze Button ---
    if st.button("🎯 Analyze Game", type="primary", use_container_width=True):
        if len(home_starters) != 5 or len(away_starters) != 5:
            st.error("Please enter exactly 5 starters for each team.")
            st.stop()

        if not home_center or not away_center:
            st.error("Please select a tip-off player for each team.")
            st.stop()

        with st.spinner("Running prediction model..."):
            predictor = load_predictor()

            prediction = predictor.predict(
                home_team=home_team,
                away_team=away_team,
                home_starters=home_starters,
                away_starters=away_starters,
                home_center=home_center,
                away_center=away_center,
            )

        # --- Jump Ball ---
        st.subheader("Jump Ball Prediction")
        jb_cols = st.columns(2)
        home_tip_pct = prediction.home_wins_tip_prob * 100
        away_tip_pct = (1 - prediction.home_wins_tip_prob) * 100

        with jb_cols[0]:
            st.metric(
                f"{home_team} wins tip",
                f"{home_tip_pct:.1f}%",
                delta=f"{home_tip_pct - 50:.1f}%" if home_tip_pct != 50 else None,
            )
        with jb_cols[1]:
            st.metric(
                f"{away_team} wins tip",
                f"{away_tip_pct:.1f}%",
                delta=f"{away_tip_pct - 50:.1f}%" if away_tip_pct != 50 else None,
            )

        tip_winner = home_team if prediction.home_wins_tip_prob > 0.5 else away_team
        st.caption(f"Predicted tip winner: **{tip_winner}** (tip winner's team scores first ~65% of the time)")

        # --- Player Probabilities ---
        st.subheader("First Scorer Probabilities")

        # Build model data
        model_probs = {}
        player_teams = {}
        player_ppg = {}
        player_fg_pct = {}
        player_fg3_rate = {}

        for p in prediction.player_probabilities:
            name = p['player_name']
            team_label = home_team if p['team'] == 'home' else away_team
            model_probs[name] = p['probability']
            player_teams[name] = team_label

            stats = predictor._get_player_api_stats(name)
            if stats:
                if stats.get('ppg', 0) > 0:
                    player_ppg[name] = stats['ppg']
                if stats.get('fg_pct', 0) > 0:
                    player_fg_pct[name] = stats['fg_pct'] / 100.0
                if stats.get('fg3_rate', 0) > 0:
                    player_fg3_rate[name] = stats['fg3_rate']

        # Determine active odds
        active_odds = {}
        if live_odds:
            matched, unmatched = match_odds_names(live_odds)
            active_odds = matched
            if unmatched:
                st.warning(f"Could not match: {', '.join(unmatched)}")
        elif manual_odds:
            active_odds = manual_odds

        # Build predictions table
        rows = []
        sorted_players = sorted(
            prediction.player_probabilities,
            key=lambda x: -x['probability']
        )

        for p in sorted_players:
            name = p['player_name']
            team_label = home_team if p['team'] == 'home' else away_team
            prob = p['probability']

            row = {
                'Player': name,
                'Team': team_label,
                'Model Prob': f"{prob:.1%}",
            }

            if name in active_odds:
                odds = active_odds[name]
                implied = 1 / _american_to_decimal(odds)
                edge = prob - implied
                row['Odds'] = f"{odds:+d}"
                row['Implied'] = f"{implied:.1%}"
                row['Edge'] = f"{edge:+.1%}"
            else:
                row['Odds'] = '-'
                row['Implied'] = '-'
                row['Edge'] = '-'

            rows.append(row)

        pred_df = pd.DataFrame(rows)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # --- Strategy Analysis ---
        if active_odds:
            st.subheader("Strategy Recommendations")

            strategies = AlternativeStrategies()
            recommendations = strategies.analyze_all_strategies(
                model_probs=model_probs,
                market_odds=active_odds,
                player_ppg=player_ppg,
                home_team=home_team,
                away_team=away_team,
                tip_winner_team=tip_winner,
                budget=budget,
                player_teams=player_teams,
                player_fg_pct=player_fg_pct if player_fg_pct else None,
                player_fg3_rate=player_fg3_rate if player_fg3_rate else None,
                home_tip_prob=prediction.home_wins_tip_prob,
            )

            positive_ev = [r for r in recommendations if r.ev > 0]

            if not positive_ev:
                st.info("No positive EV bets found for this game. Consider passing.")
            else:
                # Group by strategy
                strategy_groups = {}
                for rec in positive_ev:
                    label = STRATEGY_LABELS.get(rec.strategy, rec.strategy)
                    if label not in strategy_groups:
                        strategy_groups[label] = []
                    strategy_groups[label].append(rec)

                for strategy_name, recs in strategy_groups.items():
                    with st.expander(f"**{strategy_name}** ({len(recs)} bet{'s' if len(recs) > 1 else ''})", expanded=True):
                        for rec in recs:
                            target = rec.player if rec.player else f"{rec.team} (Team)"
                            conf_color = CONFIDENCE_COLORS.get(rec.confidence, '#888')

                            st.markdown(f"**{target}**")

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Odds", f"{rec.odds:+d}")
                            m2.metric("Model Prob", f"{rec.model_prob:.1%}")
                            m3.metric("Edge", f"{rec.edge:+.1%}")
                            m4.metric("EV / $1", f"${rec.ev:.3f}")

                            st.caption(
                                f"Bet type: {rec.bet_type} | "
                                f"Confidence: :{conf_color}[{rec.confidence}] | "
                                f"{rec.reasoning}"
                            )

                            if rec.sub_bets:
                                sub_df = pd.DataFrame([
                                    {
                                        'Player': sb['player'],
                                        'Odds': f"{sb['odds']:+d}",
                                        'Model Prob': f"{sb['model_prob']:.1%}",
                                    }
                                    for sb in rec.sub_bets
                                ])
                                st.dataframe(sub_df, use_container_width=True, hide_index=True)

                            st.divider()

                # --- Allocation ---
                st.subheader(f"Suggested Allocation (${budget:.0f} budget)")

                total_edge = sum(r.edge for r in positive_ev if r.edge > 0)
                allocations = []
                total_allocated = 0.0

                for rec in sorted(positive_ev, key=lambda x: -x.ev)[:4]:
                    if rec.edge > 0 and total_edge > 0:
                        alloc = (rec.edge / total_edge) * budget
                        alloc = min(alloc, budget * 0.5)
                        total_allocated += alloc
                        target = rec.player if rec.player else rec.team
                        allocations.append({
                            'Bet': target,
                            'Strategy': STRATEGY_LABELS.get(rec.strategy, rec.strategy),
                            'Odds': f"{rec.odds:+d}",
                            'Amount': f"${alloc:.2f}",
                        })

                reserve = budget - total_allocated
                if reserve > 0:
                    allocations.append({
                        'Bet': 'Reserve',
                        'Strategy': '-',
                        'Odds': '-',
                        'Amount': f"${reserve:.2f}",
                    })

                alloc_df = pd.DataFrame(allocations)
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)

                # Summary metrics
                sum_cols = st.columns(3)
                sum_cols[0].metric("Total Wagered", f"${total_allocated:.2f}")
                avg_ev = np.mean([r.ev for r in positive_ev[:4]]) if positive_ev else 0
                sum_cols[1].metric("Avg EV / $1", f"${avg_ev:.3f}")
                sum_cols[2].metric("Reserve", f"${reserve:.2f}")
        else:
            st.info("Enter odds or provide an API key to see strategy recommendations.")

    # --- Footer ---
    st.divider()
    st.markdown("""
    **How it works**: The V6 XGBoost model predicts jump ball outcomes and first scorer
    probabilities for all 10 starters. Nine betting strategies compare model probabilities
    against sportsbook odds to find edges.

    Strategies: Value Bet, Team First Scorer, Multi-Player Hedge, Score Type Filter,
    Correlated Parlay, Efficiency Play, Second Choice, Contrarian Longshot, Tip Winner Team.

    ---

    *This tool is for educational and entertainment purposes only. Gambling involves risk.
    Past performance does not guarantee future results. Please gamble responsibly.*
    """)


if __name__ == "__main__":
    main()
