"""
NBA Data Fetcher using nba_api

Primary data source for:
- Play-by-play data (contains jump ball and first scorer info)
- Game schedules
- Team rosters and lineups
- Player stats

nba_api is preferred over scraping because:
1. It's the official stats.nba.com API
2. More reliable and structured
3. Contains detailed play-by-play with event types
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# These imports will work once you pip install nba_api
try:
    from nba_api.stats.endpoints import (
        playbyplayv2,
        leaguegamefinder,
        boxscoretraditionalv2,
        commonteamroster,
        teamgamelog,
    )
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("nba_api not installed. Run: pip install nba_api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NBA Event Types in play-by-play data
EVENT_TYPES = {
    1: 'FIELD_GOAL_MADE',
    2: 'FIELD_GOAL_MISSED', 
    3: 'FREE_THROW',
    4: 'REBOUND',
    5: 'TURNOVER',
    6: 'FOUL',
    7: 'VIOLATION',
    8: 'SUBSTITUTION',
    9: 'TIMEOUT',
    10: 'JUMP_BALL',
    11: 'EJECTION',
    12: 'PERIOD_BEGIN',
    13: 'PERIOD_END',
    18: 'INSTANT_REPLAY',
    20: 'STOPPAGE'
}


class NBADataFetcher:
    """Fetches NBA data from stats.nba.com via nba_api"""
    
    def __init__(self, data_dir: str = "data/raw", request_delay: float = 0.6):
        """
        Args:
            data_dir: Directory to save raw data
            request_delay: Seconds between API requests (be nice to the API)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        
        if NBA_API_AVAILABLE:
            self.teams_df = pd.DataFrame(teams.get_teams())
            self.team_id_map = dict(zip(self.teams_df['abbreviation'], self.teams_df['id']))
        
    def _rate_limit(self):
        """Respect API rate limits"""
        time.sleep(self.request_delay)
    
    def get_season_games(self, season: str) -> pd.DataFrame:
        """
        Get all games for a season.
        
        Args:
            season: Season string like '2023-24'
            
        Returns:
            DataFrame with game_id, date, home_team, away_team
        """
        if not NBA_API_AVAILABLE:
            raise ImportError("nba_api required")
            
        logger.info(f"Fetching games for season {season}")
        
        # Get games for any team (we'll get all games)
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',  # NBA
            season_type_nullable='Regular Season'
        )
        self._rate_limit()
        
        games_df = gamefinder.get_data_frames()[0]
        
        # Each game appears twice (once per team), deduplicate
        games_df = games_df.drop_duplicates(subset=['GAME_ID'])
        
        # Parse home/away from MATCHUP column (e.g., "LAL vs. GSW" or "LAL @ GSW")
        def parse_matchup(row):
            matchup = row['MATCHUP']
            team = row['TEAM_ABBREVIATION']
            if ' vs. ' in matchup:
                # Home game
                opponent = matchup.split(' vs. ')[1]
                return pd.Series({'home_team': team, 'away_team': opponent})
            else:
                # Away game
                opponent = matchup.split(' @ ')[1]
                return pd.Series({'home_team': opponent, 'away_team': team})
        
        parsed = games_df.apply(parse_matchup, axis=1)
        games_df = pd.concat([games_df, parsed], axis=1)
        
        result = games_df[['GAME_ID', 'GAME_DATE', 'home_team', 'away_team']].copy()
        result.columns = ['game_id', 'date', 'home_team', 'away_team']
        result['date'] = pd.to_datetime(result['date'])
        
        logger.info(f"Found {len(result)} games for {season}")
        return result.sort_values('date').reset_index(drop=True)
    
    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        """
        Get play-by-play data for a single game.
        
        This contains:
        - EVENTMSGTYPE: Event type code (10 = jump ball, 1 = made FG, etc.)
        - PLAYER1_NAME: Primary player involved
        - PLAYER2_NAME: Secondary player (for jump balls, this is opponent)
        - PLAYER3_NAME: Tertiary player (for jump balls, this is who got possession)
        - HOMEDESCRIPTION/VISITORDESCRIPTION: Text description
        """
        if not NBA_API_AVAILABLE:
            raise ImportError("nba_api required")
            
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        self._rate_limit()
        
        return pbp.get_data_frames()[0]
    
    def extract_jump_ball_and_first_scorer(self, pbp_df: pd.DataFrame, game_id: str) -> Optional[Dict]:
        """
        Extract jump ball result and first scorer from play-by-play data.
        
        Returns dict with:
        - game_id
        - jump_ball_winner: Player who won the tip
        - jump_ball_loser: Player who lost the tip
        - possession_team: Team that got possession
        - first_scorer: Player who scored first
        - first_scorer_team: Team of first scorer
        - first_score_type: Type of first score (2PT, 3PT, FT)
        - possession_to_first_score: Did team that won tip score first?
        """
        if pbp_df.empty:
            return None
            
        # Filter to first period only for opening tip
        q1_pbp = pbp_df[pbp_df['PERIOD'] == 1].copy()
        
        if q1_pbp.empty:
            return None
        
        result = {'game_id': game_id}
        
        # Find opening jump ball (EVENTMSGTYPE = 10)
        jump_balls = q1_pbp[q1_pbp['EVENTMSGTYPE'] == 10]
        
        if not jump_balls.empty:
            first_jump = jump_balls.iloc[0]
            
            # Parse jump ball description to determine winner
            # Format varies but typically: "Jump Ball Player1 vs. Player2: Tip to Player3"
            home_desc = str(first_jump.get('HOMEDESCRIPTION', ''))
            away_desc = str(first_jump.get('VISITORDESCRIPTION', ''))
            neutral_desc = str(first_jump.get('NEUTRALDESCRIPTION', ''))
            
            desc = home_desc if home_desc and home_desc != 'nan' else (
                   away_desc if away_desc and away_desc != 'nan' else neutral_desc)
            
            result['jump_ball_player1'] = first_jump.get('PLAYER1_NAME')
            result['jump_ball_player2'] = first_jump.get('PLAYER2_NAME')
            result['jump_ball_possession_to'] = first_jump.get('PLAYER3_NAME')
            result['jump_ball_player1_team'] = first_jump.get('PLAYER1_TEAM_ABBREVIATION')
            result['jump_ball_player2_team'] = first_jump.get('PLAYER2_TEAM_ABBREVIATION')
            result['jump_ball_description'] = desc
            
            # Determine winner based on who got possession
            possession_player = first_jump.get('PLAYER3_NAME')
            player1_team = first_jump.get('PLAYER1_TEAM_ABBREVIATION')
            player3_team = first_jump.get('PLAYER3_TEAM_ABBREVIATION')
            
            if possession_player and player1_team and player3_team:
                if player1_team == player3_team:
                    result['jump_ball_winner'] = first_jump.get('PLAYER1_NAME')
                    result['jump_ball_loser'] = first_jump.get('PLAYER2_NAME')
                else:
                    result['jump_ball_winner'] = first_jump.get('PLAYER2_NAME')
                    result['jump_ball_loser'] = first_jump.get('PLAYER1_NAME')
                result['jump_ball_winning_team'] = player3_team
        
        # Find first made field goal or free throw (EVENTMSGTYPE 1 or 3 with made)
        # Note: For free throws, EVENTMSGACTIONTYPE matters for made/missed
        made_shots = q1_pbp[
            (q1_pbp['EVENTMSGTYPE'] == 1) |  # Made FG
            ((q1_pbp['EVENTMSGTYPE'] == 3) & (q1_pbp['EVENTMSGACTIONTYPE'].isin([10, 11, 12, 13, 14, 15])))  # Made FT
        ]
        
        if not made_shots.empty:
            first_score = made_shots.iloc[0]
            result['first_scorer'] = first_score.get('PLAYER1_NAME')
            result['first_scorer_team'] = first_score.get('PLAYER1_TEAM_ABBREVIATION')
            result['first_score_event_type'] = first_score.get('EVENTMSGTYPE')
            
            # Determine if 2PT or 3PT from description
            home_desc = str(first_score.get('HOMEDESCRIPTION', ''))
            away_desc = str(first_score.get('VISITORDESCRIPTION', ''))
            score_desc = home_desc if home_desc and home_desc != 'nan' else away_desc
            
            if '3PT' in score_desc:
                result['first_score_type'] = '3PT'
            elif first_score.get('EVENTMSGTYPE') == 3:
                result['first_score_type'] = 'FT'
            else:
                result['first_score_type'] = '2PT'
            
            result['first_score_description'] = score_desc
            
            # Did the team that won the tip score first?
            if 'jump_ball_winning_team' in result and 'first_scorer_team' in result:
                result['tip_winner_scored_first'] = (
                    result['jump_ball_winning_team'] == result['first_scorer_team']
                )
        
        return result
    
    def fetch_season_data(self, season: str, save: bool = True) -> pd.DataFrame:
        """
        Fetch all jump ball and first scorer data for a season.
        
        This is the main method to call for data collection.
        """
        games = self.get_season_games(season)
        
        results = []
        total_games = len(games)
        
        for idx, game in games.iterrows():
            game_id = game['game_id']
            logger.info(f"Processing game {idx+1}/{total_games}: {game_id}")
            
            try:
                pbp = self.get_play_by_play(game_id)
                game_data = self.extract_jump_ball_and_first_scorer(pbp, game_id)
                
                if game_data:
                    game_data['date'] = game['date']
                    game_data['home_team'] = game['home_team']
                    game_data['away_team'] = game['away_team']
                    results.append(game_data)
                    
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        if save and not df.empty:
            filepath = self.data_dir / f"jump_ball_first_scorer_{season.replace('-', '_')}.parquet"
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} games to {filepath}")
        
        return df
    
    def fetch_multiple_seasons(self, seasons: List[str]) -> pd.DataFrame:
        """Fetch data for multiple seasons and combine."""
        all_data = []
        
        for season in seasons:
            logger.info(f"\n{'='*50}\nFetching season {season}\n{'='*50}")
            season_df = self.fetch_season_data(season, save=True)
            all_data.append(season_df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(self.data_dir / "all_seasons_combined.parquet", index=False)
        
        return combined


class PlayerPhysicalData:
    """Fetch and manage player physical attributes (height, reach, etc.)"""
    
    def __init__(self):
        if NBA_API_AVAILABLE:
            self.all_players = pd.DataFrame(players.get_players())
    
    def get_player_info(self, player_name: str) -> Optional[Dict]:
        """Get player info by name. Note: Height is in player profiles."""
        # This is basic - for real project you'd want to scrape additional
        # physical data or use a supplementary source
        if not NBA_API_AVAILABLE:
            return None
            
        matches = self.all_players[
            self.all_players['full_name'].str.lower() == player_name.lower()
        ]
        
        if matches.empty:
            # Try partial match
            matches = self.all_players[
                self.all_players['full_name'].str.lower().str.contains(player_name.lower())
            ]
        
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None


def get_current_starters(team_abbrev: str) -> List[str]:
    """
    Get likely starters for a team.
    
    Note: True starting lineups are announced ~30 min before game.
    This function returns the most common recent starters.
    
    For production, you'd want to:
    1. Use a real-time lineup service (FantasyLabs, RotoGrinders, etc.)
    2. Or scrape team's recent games for most common starter combinations
    """
    if not NBA_API_AVAILABLE:
        return []
    
    # Get team ID
    teams_df = pd.DataFrame(teams.get_teams())
    team_row = teams_df[teams_df['abbreviation'] == team_abbrev]
    
    if team_row.empty:
        return []
    
    team_id = team_row.iloc[0]['id']
    
    # Get roster
    roster = commonteamroster.CommonTeamRoster(team_id=team_id)
    roster_df = roster.get_data_frames()[0]
    
    # Return all players - in production you'd analyze recent box scores
    # to determine actual starters
    return roster_df['PLAYER'].tolist()


if __name__ == "__main__":
    # Example usage
    fetcher = NBADataFetcher()
    
    # Fetch last 4 seasons
    seasons = ['2021-22', '2022-23', '2023-24', '2024-25']
    
    # Start with one season to test
    print("Testing with 2023-24 season...")
    df = fetcher.fetch_season_data('2023-24')
    print(f"\nCollected {len(df)} games")
    print(df.head())
