"""
ESPN Depth Chart Lineup Fetcher

Fetches projected starting lineups from ESPN's free depth chart API.
No API key required. Returns the first player at each position (PG, SG, SF, PF, C)
as the projected starter, along with injury information.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import requests

logger = logging.getLogger(__name__)

# Project team abbreviation -> ESPN numeric team ID
APP_ABBREV_TO_ESPN_ID = {
    'ATL': 1,   'BOS': 2,   'BKN': 17,  'CHA': 30,
    'CHI': 4,   'CLE': 5,   'DAL': 6,   'DEN': 7,
    'DET': 8,   'GSW': 9,   'HOU': 10,  'IND': 11,
    'LAC': 12,  'LAL': 13,  'MEM': 29,  'MIA': 14,
    'MIL': 15,  'MIN': 16,  'NOP': 3,   'NYK': 18,
    'OKC': 25,  'ORL': 19,  'PHI': 20,  'PHX': 21,
    'POR': 22,  'SAC': 23,  'SAS': 24,  'TOR': 28,
    'UTA': 26,  'WAS': 27,
}

POSITIONS = ['pg', 'sg', 'sf', 'pf', 'c']
POSITION_LABELS = {'pg': 'PG', 'sg': 'SG', 'sf': 'SF', 'pf': 'PF', 'c': 'C'}


@dataclass
class StarterInfo:
    name: str
    position: str
    injury_status: Optional[str] = None
    injury_comment: Optional[str] = None


class ESPNLineupFetcher:
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def get_starters(self, team_abbrev: str) -> Optional[Dict]:
        """
        Fetch projected starters for a team from ESPN depth charts.

        Returns dict with 'players' (List[StarterInfo]), 'center' (str), 'team' (str),
        or None on error.
        """
        espn_id = APP_ABBREV_TO_ESPN_ID.get(team_abbrev.upper())
        if espn_id is None:
            logger.error(f"Unknown team abbreviation: {team_abbrev}")
            return None

        url = f"{self.BASE_URL}/{espn_id}/depthcharts"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"ESPN API request failed for {team_abbrev}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON from ESPN for {team_abbrev}: {e}")
            return None

        return self._parse_depth_chart(data, team_abbrev)

    def _parse_depth_chart(self, data: dict, team_abbrev: str) -> Optional[Dict]:
        """Parse ESPN depth chart response into starters list."""
        try:
            depth_charts = data.get('depthchart', [])
            if not depth_charts:
                logger.warning(f"No depth chart data for {team_abbrev}")
                return None

            positions_data = depth_charts[0].get('positions', {})
            starters: List[StarterInfo] = []
            center_name = None

            for pos_key in POSITIONS:
                pos_data = positions_data.get(pos_key, {})
                athletes = pos_data.get('athletes', [])

                if not athletes:
                    logger.warning(f"No athletes at {pos_key} for {team_abbrev}")
                    starters.append(StarterInfo(name="", position=POSITION_LABELS[pos_key]))
                    continue

                athlete = athletes[0]
                name = athlete.get('displayName', '')

                injury_status = None
                injury_comment = None
                injuries = athlete.get('injuries', [])
                if injuries:
                    inj = injuries[0]
                    injury_status = inj.get('status')
                    injury_comment = inj.get('shortComment')

                starter = StarterInfo(
                    name=name,
                    position=POSITION_LABELS[pos_key],
                    injury_status=injury_status,
                    injury_comment=injury_comment,
                )
                starters.append(starter)

                if pos_key == 'c':
                    center_name = name

            return {
                'players': starters,
                'center': center_name or '',
                'team': team_abbrev,
            }

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing depth chart for {team_abbrev}: {e}")
            return None
