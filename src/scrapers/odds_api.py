"""
The Odds API - First Basket Scorer Odds Fetcher

Uses the-odds-api.com to fetch first basket scorer odds from FanDuel
and other sportsbooks. This is the primary/recommended approach since
it avoids bot detection issues with direct sportsbook scraping.

Free tier: 500 API credits/month.
Docs: https://the-odds-api.com/liveapi/guides/v4/

Usage:
    from scrapers.odds_api import OddsAPIFetcher

    fetcher = OddsAPIFetcher(api_key="YOUR_KEY")
    games = fetcher.get_nba_games()
    odds = fetcher.get_first_basket_odds(event_id, bookmaker="fanduel")
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bookmaker keys for The Odds API
BOOKMAKER_KEYS = {
    "fanduel": "fanduel",
    "draftkings": "draftkings",
    "betmgm": "betmgm",
    "caesars": "williamhill_us",
    "pointsbet": "pointsbetus",
    "bet365": "bet365",
}


class OddsAPIFetcher:
    """
    Fetches first basket scorer odds from The Odds API.

    This API aggregates odds from multiple sportsbooks including FanDuel.
    Free tier gives 500 credits/month, which is plenty for daily use.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"

    def __init__(
        self,
        api_key: str,
        bookmaker: str = "fanduel",
        cache_dir: str = "data/odds_cache",
        cache_ttl_minutes: int = 30,
    ):
        self.api_key = api_key
        self.bookmaker = BOOKMAKER_KEYS.get(bookmaker.lower(), bookmaker)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_minutes
        self.remaining_requests = None
        self.used_requests = None

    def _get(self, url: str, params: dict) -> Optional[dict]:
        """Make a GET request to The Odds API."""
        params["apiKey"] = self.api_key
        try:
            resp = requests.get(url, params=params, timeout=15)

            # Track API usage from headers
            self.remaining_requests = resp.headers.get("x-requests-remaining")
            self.used_requests = resp.headers.get("x-requests-used")

            if resp.status_code == 401:
                logger.error("Invalid API key. Get one at https://the-odds-api.com/")
                return None
            if resp.status_code == 422:
                logger.error(f"Invalid request parameters: {resp.text}")
                return None
            if resp.status_code == 429:
                logger.error("API rate limit exceeded. Free tier: 500 credits/month.")
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_nba_games(self) -> List[Dict]:
        """
        Get today's NBA games.

        Returns list of dicts with keys:
            - id: Event ID (needed for odds lookup)
            - home_team: Full team name
            - away_team: Full team name
            - commence_time: ISO datetime string
        """
        url = f"{self.BASE_URL}/sports/{self.SPORT}/events"
        data = self._get(url, {})

        if not data:
            return []

        games = []
        for event in data:
            games.append({
                "id": event["id"],
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event.get("commence_time", ""),
            })

        logger.info(f"Found {len(games)} upcoming NBA games")
        if self.remaining_requests is not None:
            logger.info(f"API credits remaining: {self.remaining_requests}")

        return games

    def get_first_basket_odds(
        self, event_id: str, bookmaker: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """
        Get first basket scorer odds for a specific game.

        Args:
            event_id: Event ID from get_nba_games()
            bookmaker: Override default bookmaker (e.g., "fanduel")

        Returns:
            Dict mapping player names to American odds, e.g.:
            {'Karl-Anthony Towns': 490, 'Jalen Brunson': 460, ...}
        """
        target_book = BOOKMAKER_KEYS.get(
            (bookmaker or "").lower(), bookmaker
        ) or self.bookmaker

        url = f"{self.BASE_URL}/sports/{self.SPORT}/events/{event_id}/odds"
        params = {
            "regions": "us",
            "markets": "player_first_basket",
            "oddsFormat": "american",
            "bookmakers": target_book,
        }

        data = self._get(url, params)
        if not data:
            return None

        # Parse the response to extract player -> odds
        odds = self._parse_odds_response(data, target_book)

        if odds:
            logger.info(f"Got first basket odds for {len(odds)} players from {target_book}")
        else:
            logger.warning(
                f"No first basket scorer odds found for event {event_id} "
                f"from {target_book}. Market may not be available yet."
            )

        if self.remaining_requests is not None:
            logger.info(f"API credits remaining: {self.remaining_requests}")

        return odds

    def _parse_odds_response(
        self, data: dict, target_book: str
    ) -> Optional[Dict[str, int]]:
        """Parse The Odds API response into player -> American odds dict."""
        odds = {}

        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return None

        # Find the target bookmaker
        book_data = None
        for bm in bookmakers:
            if bm.get("key") == target_book:
                book_data = bm
                break

        # Fall back to first available bookmaker
        if not book_data and bookmakers:
            book_data = bookmakers[0]
            logger.info(f"Using {book_data.get('key', 'unknown')} (requested {target_book} not available)")

        if not book_data:
            return None

        # Find the first basket market
        for market in book_data.get("markets", []):
            if market.get("key") == "player_first_basket":
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description") or outcome.get("name", "")
                    price = outcome.get("price")

                    # Only include "Yes" outcomes (player scores first basket)
                    if outcome.get("name", "").lower() == "no":
                        continue

                    if player_name and price is not None:
                        odds[player_name] = int(price)

        return odds if odds else None

    def get_odds_for_game(
        self, home_team: str, away_team: str
    ) -> Optional[Dict[str, int]]:
        """
        Find a specific game and return its first basket scorer odds.

        Args:
            home_team: Team abbreviation (e.g., "NYK") or full name
            away_team: Team abbreviation (e.g., "MIL") or full name

        Returns:
            Dict[str, int] mapping player names to American odds, or None
        """
        from scrapers.fanduel_odds import normalize_team, TEAM_ABBREV_TO_FULL

        home_abbrev = normalize_team(home_team)
        away_abbrev = normalize_team(away_team)

        # Check cache first
        cached = self._load_cached_odds(home_abbrev, away_abbrev)
        if cached is not None:
            logger.info(f"Using cached odds for {away_abbrev} @ {home_abbrev}")
            return cached

        # Get games
        games = self.get_nba_games()
        if not games:
            return None

        # Find matching game — The Odds API uses full team names
        home_full = TEAM_ABBREV_TO_FULL.get(home_abbrev, home_team)
        away_full = TEAM_ABBREV_TO_FULL.get(away_abbrev, away_team)

        target = None
        for game in games:
            game_home = game["home_team"]
            game_away = game["away_team"]

            # Match by full name or partial
            if (self._team_matches(game_home, home_abbrev, home_full) and
                self._team_matches(game_away, away_abbrev, away_full)):
                target = game
                break

        if not target:
            # Try reverse match
            for game in games:
                game_home = game["home_team"]
                game_away = game["away_team"]
                if (self._team_matches(game_home, away_abbrev, away_full) and
                    self._team_matches(game_away, home_abbrev, home_full)):
                    target = game
                    logger.info(f"Note: home/away might be swapped; using API's assignment")
                    break

        if not target:
            available = [f"{g['away_team']} @ {g['home_team']}" for g in games]
            logger.error(
                f"Game {away_abbrev} @ {home_abbrev} not found. "
                f"Available: {available}"
            )
            return None

        # Fetch odds
        odds = self.get_first_basket_odds(target["id"])

        if odds:
            self._cache_odds(home_abbrev, away_abbrev, odds)

        return odds

    def get_all_games_odds(self) -> Dict[str, Dict[str, int]]:
        """
        Get first basket scorer odds for all today's NBA games.

        Returns dict keyed by "AWAY @ HOME" -> odds dict.
        """
        from scrapers.fanduel_odds import normalize_team

        games = self.get_nba_games()
        all_odds = {}

        for game in games:
            home = game["home_team"]
            away = game["away_team"]
            home_abbrev = normalize_team(home)
            away_abbrev = normalize_team(away)
            key = f"{away_abbrev} @ {home_abbrev}"

            # Check cache
            cached = self._load_cached_odds(home_abbrev, away_abbrev)
            if cached:
                all_odds[key] = cached
                logger.info(f"{key}: loaded from cache ({len(cached)} players)")
                continue

            logger.info(f"Fetching odds for {key}...")
            odds = self.get_first_basket_odds(game["id"])

            if odds:
                all_odds[key] = odds
                self._cache_odds(home_abbrev, away_abbrev, odds)
                logger.info(f"{key}: {len(odds)} players")
            else:
                logger.warning(f"{key}: no first basket scorer odds available")

            time.sleep(0.5)  # Small delay between requests

        return all_odds

    @staticmethod
    def _team_matches(api_name: str, abbrev: str, full_name: str) -> bool:
        """Check if an API team name matches an abbreviation or full name."""
        api_lower = api_name.lower()
        full_lower = full_name.lower()
        # Exact match
        if full_lower == api_lower:
            return True
        # Substring match (e.g., "LA Clippers" in "Los Angeles Clippers")
        if full_lower in api_lower or api_lower in full_lower:
            return True
        # Last word of our full name in API name (e.g., "Bucks" in "Milwaukee Bucks")
        full_parts = full_name.split()
        if full_parts and full_parts[-1].lower() == api_name.split()[-1].lower():
            return True
        return False

    def _cache_odds(self, home_abbrev: str, away_abbrev: str, odds: Dict[str, int]):
        """Cache odds to JSON file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"{date_str}_{away_abbrev}_{home_abbrev}.json"

        data = {
            "fetched_at": datetime.now().isoformat(),
            "source": "the-odds-api",
            "bookmaker": self.bookmaker,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "odds": odds,
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cached_odds(
        self, home_abbrev: str, away_abbrev: str
    ) -> Optional[Dict[str, int]]:
        """Load cached odds if fresh enough."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"{date_str}_{away_abbrev}_{home_abbrev}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            fetched_at = datetime.fromisoformat(data["fetched_at"])
            age_minutes = (datetime.now() - fetched_at).total_seconds() / 60

            if age_minutes > self.cache_ttl:
                return None

            return data["odds"]
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
