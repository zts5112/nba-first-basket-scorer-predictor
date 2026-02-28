"""
Basketball Reference Selenium Scraper

Uses Selenium with a real Chrome browser to scrape Basketball Reference,
bypassing rate limits and bot detection.

Features:
- Uses real Chrome browser (headless or visible)
- Automatically manages ChromeDriver
- Checkpointing and resume capability
- Respects rate limits with configurable delays
"""

import time
import logging
import re
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
from threading import Lock
import random

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeleniumBasketballReferenceScraper:
    """
    Scrapes play-by-play data from Basketball Reference using Selenium.

    Uses a real Chrome browser to bypass bot detection and rate limits.
    """

    BASE_URL = "https://www.basketball-reference.com"

    TEAM_ABBREV_MAP = {
        'BRK': 'BKN',
        'CHO': 'CHA',
        'PHO': 'PHX',
    }

    MONTHS = ['october', 'november', 'december', 'january', 'february',
              'march', 'april', 'may', 'june']

    def __init__(self, data_dir: str = "data/raw", headless: bool = True,
                 delay_between_requests: float = 3.0, checkpoint_interval: int = 25):
        """
        Args:
            data_dir: Directory to save raw data
            headless: Run Chrome in headless mode (no visible window)
            delay_between_requests: Seconds to wait between requests
            checkpoint_interval: Save progress every N games
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.delay = delay_between_requests
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_lock = Lock()
        self.driver = None

    def _create_driver(self):
        """Create and configure Chrome WebDriver."""
        options = Options()

        if self.headless:
            options.add_argument('--headless=new')

        # Make Chrome look more like a real browser
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Disable automation flags
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)

        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        # Remove webdriver property
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })

        logger.info(f"Chrome WebDriver initialized (headless={self.headless})")
        return driver

    def _ensure_driver(self):
        """Ensure driver is initialized."""
        if self.driver is None:
            self.driver = self._create_driver()

    def _get_page(self, url: str, retries: int = 3) -> Optional[str]:
        """Get page content with retries and Cloudflare handling."""
        self._ensure_driver()

        # Set page load timeout to avoid indefinite hangs
        self.driver.set_page_load_timeout(30)

        for attempt in range(retries):
            try:
                # Random delay to appear more human
                time.sleep(self.delay + random.uniform(0.5, 1.5))

                try:
                    self.driver.get(url)
                except TimeoutException:
                    logger.warning(f"Page load timeout for {url}, checking if Cloudflare challenge...")

                # Wait for body
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except TimeoutException:
                    pass

                page_source = self.driver.page_source

                # Handle Cloudflare challenge
                if "just a moment" in page_source.lower() or "cf-challenge" in page_source.lower():
                    logger.warning(f"Cloudflare challenge detected, waiting for it to resolve (attempt {attempt + 1}/{retries})...")
                    # Wait up to 15 seconds for Cloudflare to auto-resolve
                    for _ in range(15):
                        time.sleep(1)
                        page_source = self.driver.page_source
                        if "just a moment" not in page_source.lower():
                            break
                    else:
                        logger.warning(f"Cloudflare challenge not resolved after 15s, retrying...")
                        time.sleep(5)
                        continue

                # Check for rate limiting page
                if "rate limit" in page_source.lower() or "too many requests" in page_source.lower():
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    continue

                return page_source

            except TimeoutException:
                logger.warning(f"Timeout loading {url} (attempt {attempt + 1}/{retries})")
                time.sleep(5)
            except WebDriverException as e:
                logger.warning(f"WebDriver error: {e} (attempt {attempt + 1}/{retries})")
                # Recreate driver on error
                self.close()
                self._ensure_driver()
                self.driver.set_page_load_timeout(30)
                time.sleep(5)

        return None

    def close(self):
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Checkpoint methods
    def _get_checkpoint_path(self, season: str) -> Path:
        return self.data_dir / f"checkpoint_{season.replace('-', '_')}.parquet"

    def _get_completed_games_path(self, season: str) -> Path:
        return self.data_dir / f"completed_games_{season.replace('-', '_')}.txt"

    def _load_completed_games(self, season: str) -> set:
        completed_path = self._get_completed_games_path(season)
        if completed_path.exists():
            with open(completed_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _save_completed_game(self, season: str, game_id: str):
        with self.checkpoint_lock:
            completed_path = self._get_completed_games_path(season)
            with open(completed_path, 'a') as f:
                f.write(f"{game_id}\n")

    def _save_checkpoint(self, season: str, results: List[Dict]):
        with self.checkpoint_lock:
            if not results:
                return
            checkpoint_path = self._get_checkpoint_path(season)
            df = pd.DataFrame(results)
            df.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(results)} games to {checkpoint_path}")

    def _load_checkpoint(self, season: str) -> List[Dict]:
        checkpoint_path = self._get_checkpoint_path(season)
        if checkpoint_path.exists():
            df = pd.read_parquet(checkpoint_path)
            logger.info(f"Loaded checkpoint: {len(df)} games from {checkpoint_path}")
            return df.to_dict('records')
        return []

    def _finalize_season(self, season: str, results: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)

        if not df.empty:
            output_path = self.data_dir / f"bball_ref_{season.replace('-', '_')}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved final data: {len(df)} games to {output_path}")

            # Clean up checkpoint files
            checkpoint_path = self._get_checkpoint_path(season)
            completed_path = self._get_completed_games_path(season)

            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if completed_path.exists():
                completed_path.unlink()

        return df

    def get_season_schedule(self, season: str) -> pd.DataFrame:
        """Get all games for a season."""
        start_year, end_year_short = season.split('-')
        end_year = int(start_year[:2] + end_year_short)

        all_games = []

        for month in self.MONTHS:
            url = f"{self.BASE_URL}/leagues/NBA_{end_year}_games-{month}.html"
            logger.info(f"Fetching schedule: {month} {end_year}")

            html = self._get_page(url)
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
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                cells = row.find_all(['td', 'th'])
                if len(cells) < 6:
                    continue

                date_cell = row.find('th', {'data-stat': 'date_game'})
                if not date_cell:
                    continue

                box_score_link = row.find('td', {'data-stat': 'box_score_text'})
                if not box_score_link or not box_score_link.find('a'):
                    continue

                game_url = box_score_link.find('a')['href']
                game_id = game_url.split('/')[-1].replace('.html', '')

                visitor_cell = row.find('td', {'data-stat': 'visitor_team_name'})
                home_cell = row.find('td', {'data-stat': 'home_team_name'})

                if not visitor_cell or not home_cell:
                    continue

                visitor_link = visitor_cell.find('a')
                home_link = home_cell.find('a')

                if visitor_link and home_link:
                    visitor_abbrev = visitor_link['href'].split('/')[2].upper()
                    home_abbrev = home_link['href'].split('/')[2].upper()
                else:
                    continue

                visitor_abbrev = self.TEAM_ABBREV_MAP.get(visitor_abbrev, visitor_abbrev)
                home_abbrev = self.TEAM_ABBREV_MAP.get(home_abbrev, home_abbrev)

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
        if href and '/players/' in href:
            match = re.search(r'/players/[a-z]/([a-z]+\d+)\.html', href)
            if match:
                return match.group(1)
        return None

    def _extract_players_from_cell(self, cell) -> List[Dict]:
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

    def _is_made_shot(self, play_text: str) -> bool:
        play_lower = play_text.lower()
        made_patterns = [
            r'makes?\s+\d-pt',
            r'makes?\s+.*shot',
            r'makes?\s+.*layup',
            r'makes?\s+.*dunk',
            r'makes?\s+free throw',
        ]
        for pattern in made_patterns:
            if re.search(pattern, play_lower):
                return True
        return False

    def parse_box_score_starters(self, html: str) -> Dict[str, List[Dict]]:
        """
        Extract starting lineups from box score page.

        Returns dict with 'home' and 'away' lists, each containing 5 starter dicts.
        BBRef puts basic box score tables with IDs like "box-LAL-game-basic" (away)
        and "box-DEN-game-basic" (home), with away team first in the HTML.
        """
        soup = BeautifulSoup(html, 'html.parser')

        starters = {'home': [], 'away': []}

        # Find basic box score tables by their ID pattern: box-XXX-game-basic
        # The order in HTML is: away team first, then home team
        basic_tables = []

        for table in soup.find_all('table'):
            table_id = table.get('id', '')
            if 'box-' in table_id and '-game-basic' in table_id:
                basic_tables.append(table)

        # Should have exactly 2 basic tables (away, then home)
        for i, table in enumerate(basic_tables[:2]):
            team_key = 'away' if i == 0 else 'home'
            team_starters = []

            tbody = table.find('tbody')
            if not tbody:
                continue

            for row in tbody.find_all('tr'):
                # Stop at reserves header row
                row_class = row.get('class', [])
                if 'thead' in row_class:
                    break

                # Get player info
                player_cell = row.find('th', {'data-stat': 'player'})
                if not player_cell:
                    continue

                player_link = player_cell.find('a')
                if not player_link:
                    continue

                player_id = player_cell.get('data-append-csv')
                player_name = player_link.get_text(strip=True)

                if player_id and player_name:
                    team_starters.append({
                        'id': player_id,
                        'name': player_name
                    })

                # Stop after 5 starters
                if len(team_starters) >= 5:
                    break

            starters[team_key] = team_starters

        return starters

    def parse_play_by_play(self, html: str, game_id: str) -> Optional[Dict]:
        """Parse play-by-play HTML to extract jump ball and first scorer."""
        soup = BeautifulSoup(html, 'html.parser')

        result = {'game_id': game_id}

        pbp_div = soup.find('div', {'id': 'div_pbp'})
        if not pbp_div:
            pbp_div = soup.find('table', {'id': 'pbp'})

        if not pbp_div:
            return None

        rows = pbp_div.find_all('tr')

        found_q1 = False
        found_jump_ball = False
        found_first_score = False
        found_first_possession = False

        for row in rows:
            cells = row.find_all('td')

            row_text = row.get_text(strip=True).lower()
            if '1st q' in row_text or '1st quarter' in row_text:
                found_q1 = True
                continue
            elif ('2nd q' in row_text or '2nd quarter' in row_text or
                  '3rd q' in row_text or '3rd quarter' in row_text):
                break

            if not found_q1:
                continue

            if len(cells) == 2:
                cell = cells[1]
                play_text = cell.get_text(strip=True)
                if not found_jump_ball and 'jump ball' in play_text.lower():
                    players = self._extract_players_from_cell(cell)

                    if len(players) >= 2:
                        result['jump_ball_player1_id'] = players[0]['id']
                        result['jump_ball_player1_name'] = players[0]['name']
                        result['jump_ball_player2_id'] = players[1]['id']
                        result['jump_ball_player2_name'] = players[1]['name']

                        if len(players) >= 3:
                            result['jump_ball_possession_to_id'] = players[2]['id']
                            result['jump_ball_possession_to_name'] = players[2]['name']

                    result['jump_ball_description'] = play_text
                    found_jump_ball = True

            elif len(cells) >= 6:
                away_cell = cells[1]
                home_cell = cells[5] if len(cells) > 5 else None

                away_play = away_cell.get_text(strip=True)
                home_play = home_cell.get_text(strip=True) if home_cell else ''

                play_text = away_play if away_play else home_play
                play_cell = away_cell if away_play else home_cell
                is_home_play = bool(home_play and not away_play)

                if not found_first_possession and play_text:
                    result['jump_ball_winning_team'] = 'HOME' if is_home_play else 'AWAY'
                    found_first_possession = True

                if not found_first_score and play_cell and self._is_made_shot(play_text):
                    players = self._extract_players_from_cell(play_cell)
                    if players:
                        result['first_scorer_id'] = players[0]['id']
                        result['first_scorer_name'] = players[0]['name']

                    result['first_scorer_team'] = 'HOME' if is_home_play else 'AWAY'
                    result['first_score_description'] = play_text

                    if '3-pt' in play_text.lower():
                        result['first_score_type'] = '3PT'
                    elif 'free throw' in play_text.lower():
                        result['first_score_type'] = 'FT'
                    else:
                        result['first_score_type'] = '2PT'

                    found_first_score = True

            if found_jump_ball and found_first_score:
                break

        if 'jump_ball_winning_team' in result and 'first_scorer_team' in result:
            result['tip_winner_scored_first'] = (
                result['jump_ball_winning_team'] == result['first_scorer_team']
            )

        return result if (found_jump_ball or found_first_score) else None

    def scrape_game(self, game_info: Dict) -> Optional[Dict]:
        """Scrape a single game's play-by-play data and starting lineups."""
        game_id = game_info['game_id']
        pbp_url = game_info['pbp_url']
        box_score_url = f"{self.BASE_URL}/boxscores/{game_id}.html"

        # Fetch play-by-play page
        pbp_html = self._get_page(pbp_url)
        if not pbp_html:
            return None

        result = self.parse_play_by_play(pbp_html, game_id)

        if result:
            result['date'] = game_info['date']
            result['home_team'] = game_info['home_team']
            result['away_team'] = game_info['away_team']

            if result.get('jump_ball_winning_team') == 'HOME':
                result['jump_ball_winning_team'] = game_info['home_team']
            elif result.get('jump_ball_winning_team') == 'AWAY':
                result['jump_ball_winning_team'] = game_info['away_team']

            if result.get('first_scorer_team') == 'HOME':
                result['first_scorer_team'] = game_info['home_team']
            elif result.get('first_scorer_team') == 'AWAY':
                result['first_scorer_team'] = game_info['away_team']

            # Fetch box score page for starting lineups
            box_html = self._get_page(box_score_url)
            if box_html:
                starters = self.parse_box_score_starters(box_html)
                result['home_starters'] = starters.get('home', [])
                result['away_starters'] = starters.get('away', [])

                # Extract just the IDs for easier querying
                result['home_starter_ids'] = [p['id'] for p in result['home_starters']]
                result['away_starter_ids'] = [p['id'] for p in result['away_starters']]
            else:
                result['home_starters'] = []
                result['away_starters'] = []
                result['home_starter_ids'] = []
                result['away_starter_ids'] = []

        return result

    def fetch_season_data(self, season: str, save: bool = True,
                          max_games: Optional[int] = None,
                          resume: bool = True) -> pd.DataFrame:
        """Fetch all jump ball and first scorer data for a season."""
        # Load checkpoint data
        results = self._load_checkpoint(season) if resume else []
        completed_games = self._load_completed_games(season) if resume else set()

        if completed_games:
            logger.info(f"Resuming from checkpoint: {len(completed_games)} games already completed")

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

        # Filter out already completed games
        remaining_games = [g for g in games if g['game_id'] not in completed_games]

        if len(remaining_games) < len(games):
            logger.info(f"Skipping {len(games) - len(remaining_games)} already-scraped games")

        logger.info(f"Scraping {len(remaining_games)} games (checkpointing every {self.checkpoint_interval})...")

        games_since_checkpoint = 0

        for i, game in enumerate(remaining_games):
            completed = len(completed_games) + i + 1

            try:
                result = self.scrape_game(game)
                if result:
                    results.append(result)
                    logger.info(f"[{completed}/{total_games}] Scraped {game['game_id']}")
                else:
                    logger.debug(f"[{completed}/{total_games}] No data for {game['game_id']}")

                self._save_completed_game(season, game['game_id'])
                games_since_checkpoint += 1

                if games_since_checkpoint >= self.checkpoint_interval:
                    self._save_checkpoint(season, results)
                    games_since_checkpoint = 0

            except Exception as e:
                logger.error(f"[{completed}/{total_games}] Error scraping {game['game_id']}: {e}")
                self._save_completed_game(season, game['game_id'])

        # Final checkpoint
        if games_since_checkpoint > 0:
            self._save_checkpoint(season, results)

        # Finalize
        if save and results:
            df = self._finalize_season(season, results)
        else:
            df = pd.DataFrame(results)

        logger.info(f"Season {season} complete: {len(df)} games collected")
        return df

    def fetch_multiple_seasons(self, seasons: List[str], save: bool = True) -> pd.DataFrame:
        """Fetch data for multiple seasons."""
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

        return combined


def main():
    """Test the Selenium scraper."""
    print("Testing Selenium Basketball Reference scraper...")

    with SeleniumBasketballReferenceScraper(
        data_dir="data/raw",
        headless=True,
        delay_between_requests=3.0
    ) as scraper:
        df = scraper.fetch_season_data('2023-24', max_games=5)

        print(f"\nCollected {len(df)} games")
        if not df.empty:
            print("\nSample data:")
            print(df.head())


if __name__ == "__main__":
    main()
