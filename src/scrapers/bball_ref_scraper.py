"""
Basketball Reference Scraper with Parallel Processing

Scrapes play-by-play data from Basketball Reference to extract:
- Jump ball results (who won the opening tip)
- First scorer information

Uses concurrent processing with rate limiting to be respectful to the server.
Organized by date to enable parallel scraping without hitting the same pages.
"""

import time
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

try:
    import browser_cookie3
    BROWSER_COOKIES_AVAILABLE = True
except ImportError:
    BROWSER_COOKIES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for web requests."""

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed + random.uniform(0.1, 0.3)
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class BasketballReferenceScraper:
    """
    Scrapes play-by-play data from Basketball Reference.

    Basketball Reference URL patterns:
    - Schedule: https://www.basketball-reference.com/leagues/NBA_2024_games-october.html
    - Box score: https://www.basketball-reference.com/boxscores/202310240BOS.html
    - Play-by-play: https://www.basketball-reference.com/boxscores/pbp/202310240BOS.html

    Features:
    - Checkpointing: Saves progress every N games to allow resuming after failures
    - Resume capability: Automatically skips already-scraped games on restart
    """

    BASE_URL = "https://www.basketball-reference.com"

    # Team abbreviation mappings (some differ from standard)
    TEAM_ABBREV_MAP = {
        'BRK': 'BKN',  # Brooklyn Nets
        'CHO': 'CHA',  # Charlotte Hornets
        'PHO': 'PHX',  # Phoenix Suns
    }

    MONTHS = ['october', 'november', 'december', 'january', 'february',
              'march', 'april', 'may', 'june']

    def __init__(self, data_dir: str = "data/raw", max_workers: int = 5,
                 requests_per_second: float = 1.0, checkpoint_interval: int = 25,
                 use_browser_cookies: bool = False):
        """
        Args:
            data_dir: Directory to save raw data
            max_workers: Maximum parallel workers (keep <= 5 to be nice)
            requests_per_second: Rate limit for requests
            checkpoint_interval: Save progress every N games
            use_browser_cookies: Use cookies from Chrome/Firefox to bypass rate limits
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = min(max_workers, 5)  # Cap at 5 to be respectful
        self.rate_limiter = RateLimiter(requests_per_second)
        self.use_browser_cookies = use_browser_cookies
        self.session = self._create_session()
        self.request_semaphore = Semaphore(max_workers)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_lock = Lock()

    def _create_session(self):
        """Create a session with appropriate headers, using cloudscraper if available."""
        if CLOUDSCRAPER_AVAILABLE:
            session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'darwin',
                    'desktop': True
                }
            )
            logger.info("Using cloudscraper for bot protection bypass")
        else:
            session = requests.Session()
            logger.info("Using standard requests (cloudscraper not available)")

        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Load browser cookies if requested
        if self.use_browser_cookies:
            self._load_browser_cookies(session)

        return session

    def _load_browser_cookies(self, session):
        """Load cookies from Chrome or Firefox browser."""
        if not BROWSER_COOKIES_AVAILABLE:
            logger.warning("browser-cookie3 not installed. Run: pip install browser-cookie3")
            return

        cookies_loaded = False

        # Try Chrome first
        try:
            chrome_cookies = browser_cookie3.chrome(domain_name='.basketball-reference.com')
            session.cookies.update(chrome_cookies)
            cookie_count = len([c for c in chrome_cookies if 'basketball-reference' in c.domain])
            if cookie_count > 0:
                logger.info(f"Loaded {cookie_count} cookies from Chrome")
                cookies_loaded = True
        except Exception as e:
            logger.debug(f"Could not load Chrome cookies: {e}")

        # Try Firefox if Chrome didn't work
        if not cookies_loaded:
            try:
                firefox_cookies = browser_cookie3.firefox(domain_name='.basketball-reference.com')
                session.cookies.update(firefox_cookies)
                cookie_count = len([c for c in firefox_cookies if 'basketball-reference' in c.domain])
                if cookie_count > 0:
                    logger.info(f"Loaded {cookie_count} cookies from Firefox")
                    cookies_loaded = True
            except Exception as e:
                logger.debug(f"Could not load Firefox cookies: {e}")

        # Try Safari
        if not cookies_loaded:
            try:
                safari_cookies = browser_cookie3.safari(domain_name='.basketball-reference.com')
                session.cookies.update(safari_cookies)
                cookie_count = len([c for c in safari_cookies if 'basketball-reference' in c.domain])
                if cookie_count > 0:
                    logger.info(f"Loaded {cookie_count} cookies from Safari")
                    cookies_loaded = True
            except Exception as e:
                logger.debug(f"Could not load Safari cookies: {e}")

        if not cookies_loaded:
            logger.warning("Could not load cookies from any browser. Make sure you've visited basketball-reference.com recently.")

    def _get_checkpoint_path(self, season: str) -> Path:
        """Get path for checkpoint file for a season."""
        return self.data_dir / f"checkpoint_{season.replace('-', '_')}.parquet"

    def _get_completed_games_path(self, season: str) -> Path:
        """Get path for completed games list."""
        return self.data_dir / f"completed_games_{season.replace('-', '_')}.txt"

    def _load_completed_games(self, season: str) -> set:
        """Load set of already-scraped game IDs."""
        completed_path = self._get_completed_games_path(season)
        if completed_path.exists():
            with open(completed_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _save_completed_game(self, season: str, game_id: str):
        """Append a completed game ID to the tracking file."""
        with self.checkpoint_lock:
            completed_path = self._get_completed_games_path(season)
            with open(completed_path, 'a') as f:
                f.write(f"{game_id}\n")

    def _save_checkpoint(self, season: str, results: List[Dict]):
        """Save current results to checkpoint file."""
        with self.checkpoint_lock:
            if not results:
                return
            checkpoint_path = self._get_checkpoint_path(season)
            df = pd.DataFrame(results)
            df.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(results)} games to {checkpoint_path}")

    def _load_checkpoint(self, season: str) -> List[Dict]:
        """Load results from checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(season)
        if checkpoint_path.exists():
            df = pd.read_parquet(checkpoint_path)
            logger.info(f"Loaded checkpoint: {len(df)} games from {checkpoint_path}")
            return df.to_dict('records')
        return []

    def _finalize_season(self, season: str, results: List[Dict]) -> pd.DataFrame:
        """Finalize season data - save final file and clean up checkpoints."""
        df = pd.DataFrame(results)

        if not df.empty:
            # Save final output
            output_path = self.data_dir / f"bball_ref_{season.replace('-', '_')}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved final data: {len(df)} games to {output_path}")

            # Clean up checkpoint files
            checkpoint_path = self._get_checkpoint_path(season)
            completed_path = self._get_completed_games_path(season)

            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed checkpoint file: {checkpoint_path}")
            if completed_path.exists():
                completed_path.unlink()
                logger.info(f"Removed completed games file: {completed_path}")

        return df

    def _make_request(self, url: str, retries: int = 5) -> Optional[str]:
        """Make a rate-limited request with retries and exponential backoff."""
        for attempt in range(retries):
            try:
                with self.request_semaphore:
                    self.rate_limiter.wait()
                    # Add extra random delay to avoid detection
                    time.sleep(random.uniform(1.0, 2.0))
                    response = self.session.get(url, timeout=30)

                    if response.status_code == 200:
                        return response.text
                    elif response.status_code == 429:
                        # Rate limited - exponential backoff
                        wait_time = min(60 * (2 ** attempt), 300)  # Max 5 minutes
                        logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{retries})...")
                        time.sleep(wait_time)
                    elif response.status_code == 404:
                        logger.debug(f"Page not found: {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status_code} for {url}")
                        time.sleep(10 * (attempt + 1))

            except requests.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                time.sleep(10 * (attempt + 1))

        return None

    def get_season_schedule(self, season: str) -> pd.DataFrame:
        """
        Get all games for a season.

        Args:
            season: Season string like "2023-24" or "2024-25"

        Returns:
            DataFrame with columns: date, game_id, home_team, away_team, url
        """
        # Convert season format: "2023-24" -> 2024 (the ending year)
        start_year, end_year_short = season.split('-')
        end_year = int(start_year[:2] + end_year_short)

        all_games = []

        for month in self.MONTHS:
            url = f"{self.BASE_URL}/leagues/NBA_{end_year}_games-{month}.html"
            logger.info(f"Fetching schedule: {month} {end_year}")

            html = self._make_request(url)
            if not html:
                continue

            soup = BeautifulSoup(html, 'html.parser')
            schedule_table = soup.find('table', {'id': 'schedule'})

            if not schedule_table:
                continue

            tbody = schedule_table.find('tbody')
            if not tbody:
                continue

            for row in tbody.find_all('tr'):
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                cells = row.find_all(['td', 'th'])
                if len(cells) < 6:
                    continue

                # Extract game info
                date_cell = row.find('th', {'data-stat': 'date_game'})
                if not date_cell:
                    continue

                box_score_link = row.find('td', {'data-stat': 'box_score_text'})
                if not box_score_link or not box_score_link.find('a'):
                    continue  # Game hasn't been played yet

                game_url = box_score_link.find('a')['href']
                game_id = game_url.split('/')[-1].replace('.html', '')

                visitor_cell = row.find('td', {'data-stat': 'visitor_team_name'})
                home_cell = row.find('td', {'data-stat': 'home_team_name'})

                if not visitor_cell or not home_cell:
                    continue

                # Get team abbreviations from links
                visitor_link = visitor_cell.find('a')
                home_link = home_cell.find('a')

                if visitor_link and home_link:
                    visitor_abbrev = visitor_link['href'].split('/')[2].upper()
                    home_abbrev = home_link['href'].split('/')[2].upper()
                else:
                    continue

                # Normalize abbreviations
                visitor_abbrev = self.TEAM_ABBREV_MAP.get(visitor_abbrev, visitor_abbrev)
                home_abbrev = self.TEAM_ABBREV_MAP.get(home_abbrev, home_abbrev)

                # Parse date
                date_str = date_cell.get_text(strip=True)
                try:
                    game_date = datetime.strptime(date_str, '%a, %b %d, %Y')
                except ValueError:
                    continue

                all_games.append({
                    'date': game_date,
                    'game_id': game_id,
                    'home_team': home_abbrev,
                    'away_team': visitor_abbrev,
                    'pbp_url': f"{self.BASE_URL}/boxscores/pbp/{game_id}.html"
                })

        df = pd.DataFrame(all_games)
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Found {len(df)} games for {season}")
        return df

    def _extract_player_id(self, href: str) -> Optional[str]:
        """Extract player ID from Basketball Reference URL."""
        # URL format: /players/j/jamesle01.html -> jamesle01
        if href and '/players/' in href:
            match = re.search(r'/players/[a-z]/([a-z]+\d+)\.html', href)
            if match:
                return match.group(1)
        return None

    def _extract_players_from_cell(self, cell) -> List[Dict]:
        """Extract all players (with IDs) from a table cell."""
        players = []
        links = cell.find_all('a')
        for link in links:
            href = link.get('href', '')
            player_id = self._extract_player_id(href)
            if player_id:
                players.append({
                    'id': player_id,
                    'name': link.get_text(strip=True),
                    'href': href
                })
        return players

    def parse_play_by_play(self, html: str, game_id: str) -> Optional[Dict]:
        """
        Parse play-by-play HTML to extract jump ball and first scorer.

        Args:
            html: Raw HTML of play-by-play page
            game_id: Game identifier

        Returns:
            Dict with jump ball and first scorer info, or None if parsing fails
        """
        soup = BeautifulSoup(html, 'html.parser')

        result = {'game_id': game_id}
        players_found = {}  # Accumulate player ID -> name mapping

        # Find the play-by-play table for Q1
        # Basketball Reference uses id="pbp" or class="pbp"
        pbp_div = soup.find('div', {'id': 'div_pbp'})
        if not pbp_div:
            pbp_div = soup.find('table', {'id': 'pbp'})

        if not pbp_div:
            logger.debug(f"No PBP table found for {game_id}")
            return None

        # Get all rows
        rows = pbp_div.find_all('tr')

        found_q1 = False
        found_jump_ball = False
        found_first_score = False
        found_first_possession = False

        for row in rows:
            cells = row.find_all('td')

            # Check for quarter header
            row_text = row.get_text(strip=True).lower()
            if '1st q' in row_text or '1st quarter' in row_text:
                found_q1 = True
                continue
            elif ('2nd q' in row_text or '2nd quarter' in row_text or
                  '3rd q' in row_text or '3rd quarter' in row_text):
                # We've moved past Q1, stop looking
                break

            if not found_q1:
                continue

            # Jump ball rows have 2 cells, play rows have 6 cells
            if len(cells) == 2:
                # Jump ball row: [time, description]
                cell = cells[1]
                play_text = cell.get_text(strip=True)
                if not found_jump_ball and 'jump ball' in play_text.lower():
                    # Extract player IDs from links
                    players = self._extract_players_from_cell(cell)
                    for p in players:
                        players_found[p['id']] = p['name']

                    result.update(self._parse_jump_ball_with_ids(cell, players))
                    found_jump_ball = True

            elif len(cells) >= 6:
                # Play row: [time, away_play, away_score_change, score, home_score_change, home_play]
                away_cell = cells[1]
                home_cell = cells[5] if len(cells) > 5 else None

                away_play = away_cell.get_text(strip=True)
                home_play = home_cell.get_text(strip=True) if home_cell else ''
                score = cells[3].get_text(strip=True) if len(cells) > 3 else ''

                play_text = away_play if away_play else home_play
                play_cell = away_cell if away_play else home_cell
                is_home_play = bool(home_play and not away_play)

                # Extract players from this cell
                if play_cell:
                    for p in self._extract_players_from_cell(play_cell):
                        players_found[p['id']] = p['name']

                # Track first possession (who won tip = who has first offensive action)
                if not found_first_possession and play_text:
                    result['jump_ball_winning_team'] = 'HOME' if is_home_play else 'AWAY'
                    found_first_possession = True

                # Look for first score
                if not found_first_score and play_cell and self._is_made_shot(play_text):
                    result.update(self._parse_first_score_with_ids(play_cell, is_home_play, score))
                    found_first_score = True

            if found_jump_ball and found_first_score:
                break

        # Determine if tip winner scored first
        if 'jump_ball_winning_team' in result and 'first_scorer_team' in result:
            result['tip_winner_scored_first'] = (
                result['jump_ball_winning_team'] == result['first_scorer_team']
            )

        return result if (found_jump_ball or found_first_score) else None

    def _parse_jump_ball_with_ids(self, cell, players: List[Dict]) -> Dict:
        """Parse jump ball using player IDs from links."""
        result = {}
        play_text = cell.get_text(strip=True)

        # Jump ball typically has 3 players: player1 vs player2 (player3 gains possession)
        if len(players) >= 2:
            result['jump_ball_player1_id'] = players[0]['id']
            result['jump_ball_player1_name'] = players[0]['name']
            result['jump_ball_player2_id'] = players[1]['id']
            result['jump_ball_player2_name'] = players[1]['name']

            if len(players) >= 3:
                result['jump_ball_possession_to_id'] = players[2]['id']
                result['jump_ball_possession_to_name'] = players[2]['name']

        result['jump_ball_description'] = play_text
        return result

    def _parse_jump_ball(self, play_text: str, first_play_team) -> Dict:
        """Parse jump ball play text (fallback without IDs)."""
        result = {}

        # BBRef format: "Jump ball:A. Davisvs.N. JokiÄ(L. Jamesgains possession)"
        pattern = r"[Jj]ump\s*[Bb]all:?\s*([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\s*vs\.?\s*([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\s*\(([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\s*gains\s*possession\)"
        match = re.search(pattern, play_text)

        if match:
            result['jump_ball_player1_name'] = match.group(1).strip()
            result['jump_ball_player2_name'] = match.group(2).strip()
            result['jump_ball_possession_to_name'] = match.group(3).strip()
            result['jump_ball_description'] = play_text
        else:
            alt_pattern = r"[Jj]ump\s*[Bb]all:?\s*([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\s*vs\.?\s*([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\s*\([Tt]ip\s*to\s*([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+)\)"
            alt_match = re.search(alt_pattern, play_text)
            if alt_match:
                result['jump_ball_player1_name'] = alt_match.group(1).strip()
                result['jump_ball_player2_name'] = alt_match.group(2).strip()
                result['jump_ball_possession_to_name'] = alt_match.group(3).strip()
                result['jump_ball_description'] = play_text

        return result

    def _is_made_shot(self, play_text: str) -> bool:
        """Check if play text represents a made shot."""
        play_lower = play_text.lower()

        # Made field goals typically have patterns like:
        # "makes 2-pt shot" or "makes 3-pt shot" or just contain point values
        made_patterns = [
            r'makes?\s+\d-pt',
            r'makes?\s+.*shot',
            r'makes?\s+.*layup',
            r'makes?\s+.*dunk',
            r'makes?\s+free throw',
            r'\d+-\d+\s*\(',  # Score pattern like "2-0 ("
        ]

        for pattern in made_patterns:
            if re.search(pattern, play_lower):
                return True

        return False

    def _parse_first_score_with_ids(self, cell, is_home_play: bool, score: str) -> Dict:
        """Parse first score using player IDs from links."""
        result = {}
        play_text = cell.get_text(strip=True)

        # First player link is the scorer
        players = self._extract_players_from_cell(cell)
        if players:
            result['first_scorer_id'] = players[0]['id']
            result['first_scorer_name'] = players[0]['name']

        result['first_scorer_team'] = 'HOME' if is_home_play else 'AWAY'
        result['first_score_description'] = play_text

        # Determine score type
        if '3-pt' in play_text.lower() or '3pt' in play_text.lower():
            result['first_score_type'] = '3PT'
        elif 'free throw' in play_text.lower():
            result['first_score_type'] = 'FT'
        else:
            result['first_score_type'] = '2PT'

        return result

    def _parse_first_score(self, play_text: str, is_home_play: bool, score: str) -> Dict:
        """Parse first score play text (fallback without IDs)."""
        result = {}

        # Extract scorer name - BBRef format: "L. Jamesmakes 2-pt shot..."
        name_pattern = r'^([A-Z]\.\s*[A-Za-z\'\-]+(?:\s+[A-Za-z\'\-]+)?)\s*makes'
        match = re.match(name_pattern, play_text)

        if match:
            result['first_scorer_name'] = match.group(1).strip()
        else:
            fallback_pattern = r'^([A-Z]\.\s*[A-Za-z\'\-À-ÿ]+(?:\s+[A-Za-z\'\-]+)?)'
            fallback_match = re.match(fallback_pattern, play_text)
            if fallback_match:
                name = fallback_match.group(1).strip()
                name = re.sub(r'makes$', '', name).strip()
                result['first_scorer_name'] = name

        result['first_scorer_team'] = 'HOME' if is_home_play else 'AWAY'
        result['first_score_description'] = play_text

        if '3-pt' in play_text.lower() or '3pt' in play_text.lower():
            result['first_score_type'] = '3PT'
        elif 'free throw' in play_text.lower():
            result['first_score_type'] = 'FT'
        else:
            result['first_score_type'] = '2PT'

        return result

    def scrape_game(self, game_info: Dict) -> Optional[Dict]:
        """
        Scrape a single game's play-by-play data.

        Args:
            game_info: Dict with game_id, pbp_url, home_team, away_team, date

        Returns:
            Dict with game data or None if scraping fails
        """
        game_id = game_info['game_id']
        url = game_info['pbp_url']

        html = self._make_request(url)
        if not html:
            return None

        result = self.parse_play_by_play(html, game_id)

        if result:
            result['date'] = game_info['date']
            result['home_team'] = game_info['home_team']
            result['away_team'] = game_info['away_team']

            # Convert HOME/AWAY to actual team abbreviations
            if result.get('jump_ball_winning_team') == 'HOME':
                result['jump_ball_winning_team'] = game_info['home_team']
            elif result.get('jump_ball_winning_team') == 'AWAY':
                result['jump_ball_winning_team'] = game_info['away_team']

            if result.get('first_scorer_team') == 'HOME':
                result['first_scorer_team'] = game_info['home_team']
            elif result.get('first_scorer_team') == 'AWAY':
                result['first_scorer_team'] = game_info['away_team']

        return result

    def scrape_games_parallel(self, games: List[Dict], season: str,
                              progress_callback=None) -> List[Dict]:
        """
        Scrape multiple games in parallel with checkpointing.

        Args:
            games: List of game info dicts
            season: Season string for checkpoint naming
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of successfully scraped game results
        """
        # Load any existing checkpoint data
        results = self._load_checkpoint(season)
        completed_games = self._load_completed_games(season)

        # Filter out already-completed games
        remaining_games = [g for g in games if g['game_id'] not in completed_games]

        if len(remaining_games) < len(games):
            logger.info(f"Resuming: {len(games) - len(remaining_games)} games already completed, {len(remaining_games)} remaining")

        total = len(games)
        completed = len(games) - len(remaining_games)
        games_since_checkpoint = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_game = {
                executor.submit(self.scrape_game, game): game
                for game in remaining_games
            }

            for future in as_completed(future_to_game):
                game = future_to_game[future]
                completed += 1
                games_since_checkpoint += 1

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"[{completed}/{total}] Scraped {game['game_id']}")
                    else:
                        logger.debug(f"[{completed}/{total}] No data for {game['game_id']}")

                    # Mark game as completed (even if no data - don't retry)
                    self._save_completed_game(season, game['game_id'])

                    # Checkpoint every N games
                    if games_since_checkpoint >= self.checkpoint_interval:
                        self._save_checkpoint(season, results)
                        games_since_checkpoint = 0

                except Exception as e:
                    logger.error(f"[{completed}/{total}] Error scraping {game['game_id']}: {e}")
                    # Still mark as completed to avoid infinite retry loops
                    self._save_completed_game(season, game['game_id'])

                if progress_callback:
                    progress_callback(completed, total)

        # Final checkpoint
        if games_since_checkpoint > 0:
            self._save_checkpoint(season, results)

        return results

    def fetch_season_data(self, season: str, save: bool = True,
                          max_games: Optional[int] = None,
                          resume: bool = True) -> pd.DataFrame:
        """
        Fetch all jump ball and first scorer data for a season.

        Supports resuming from checkpoints if the scrape was interrupted.

        Args:
            season: Season string like "2023-24"
            save: Whether to save results to disk
            max_games: Optional limit on number of games (for testing)
            resume: Whether to resume from checkpoint (default True)

        Returns:
            DataFrame with game data
        """
        # Check for existing checkpoint
        checkpoint_path = self._get_checkpoint_path(season)
        if resume and checkpoint_path.exists():
            logger.info(f"Found checkpoint for {season}, will resume from where we left off")

        logger.info(f"Fetching schedule for {season}...")
        schedule = self.get_season_schedule(season)

        if schedule.empty:
            logger.warning(f"No games found for {season}")
            return pd.DataFrame()

        games = schedule.to_dict('records')
        total_games = len(games)

        if max_games:
            games = games[:max_games]
            logger.info(f"Limited to {max_games} games for testing")

        logger.info(f"Scraping {len(games)} games with {self.max_workers} workers (checkpointing every {self.checkpoint_interval} games)...")
        results = self.scrape_games_parallel(games, season)

        # Finalize - save final output and clean up checkpoints
        if save and results:
            df = self._finalize_season(season, results)
        else:
            df = pd.DataFrame(results)

        logger.info(f"Season {season} complete: {len(df)} games collected out of {total_games} total")
        return df

    def fetch_multiple_seasons(self, seasons: List[str],
                               save: bool = True) -> pd.DataFrame:
        """
        Fetch data for multiple seasons.

        Args:
            seasons: List of season strings like ["2022-23", "2023-24"]
            save: Whether to save results

        Returns:
            Combined DataFrame with all seasons
        """
        all_data = []

        for season in seasons:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing season: {season}")
            logger.info(f"{'='*50}")

            season_df = self.fetch_season_data(season, save=True)
            if not season_df.empty:
                season_df['season'] = season
                all_data.append(season_df)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        if save:
            output_path = self.data_dir / "all_seasons_combined.parquet"
            combined.to_parquet(output_path, index=False)
            logger.info(f"Saved combined data ({len(combined)} games) to {output_path}")

            # Build and save player lookup table
            player_lookup = self.build_player_lookup(combined)
            if not player_lookup.empty:
                lookup_path = self.data_dir / "player_lookup.parquet"
                player_lookup.to_parquet(lookup_path, index=False)
                logger.info(f"Saved player lookup ({len(player_lookup)} players) to {lookup_path}")

        return combined

    def build_player_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a player lookup table from scraped game data.

        Extracts unique player_id -> name mappings from all player columns.

        Args:
            df: DataFrame with game data containing player ID and name columns

        Returns:
            DataFrame with columns: player_id, player_name, bball_ref_url
        """
        players = {}

        # Extract from jump ball columns
        id_name_pairs = [
            ('jump_ball_player1_id', 'jump_ball_player1_name'),
            ('jump_ball_player2_id', 'jump_ball_player2_name'),
            ('jump_ball_possession_to_id', 'jump_ball_possession_to_name'),
            ('first_scorer_id', 'first_scorer_name'),
        ]

        for id_col, name_col in id_name_pairs:
            if id_col in df.columns and name_col in df.columns:
                for _, row in df[[id_col, name_col]].dropna().iterrows():
                    player_id = row[id_col]
                    player_name = row[name_col]
                    if player_id and player_id not in players:
                        players[player_id] = player_name

        if not players:
            return pd.DataFrame()

        # Build lookup table
        records = []
        for player_id, player_name in players.items():
            # Construct BBRef URL
            first_letter = player_id[0] if player_id else ''
            url = f"https://www.basketball-reference.com/players/{first_letter}/{player_id}.html"
            records.append({
                'player_id': player_id,
                'player_name': player_name,
                'bball_ref_url': url
            })

        lookup_df = pd.DataFrame(records)
        lookup_df = lookup_df.sort_values('player_name').reset_index(drop=True)

        return lookup_df


def main():
    """Test the scraper with a small sample."""
    scraper = BasketballReferenceScraper(
        data_dir="data/raw",
        max_workers=3,
        requests_per_second=0.5
    )

    # Test with just a few games
    print("Testing Basketball Reference scraper...")
    df = scraper.fetch_season_data('2023-24', max_games=10)

    print(f"\nCollected {len(df)} games")
    if not df.empty:
        print("\nSample data:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())


if __name__ == "__main__":
    main()
