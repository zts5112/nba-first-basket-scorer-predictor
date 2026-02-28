"""
FanDuel First Basket Scorer Odds Scraper

Scrapes first basket scorer odds from FanDuel Sportsbook using Selenium.
FanDuel has no public API, so we use a real Chrome browser to navigate
the site and extract odds from the rendered DOM.

Usage:
    from scrapers.fanduel_odds import FanDuelOddsScraper

    with FanDuelOddsScraper(headless=False) as scraper:
        games = scraper.get_todays_games()
        odds = scraper.get_first_basket_odds(games[0]['game_url'])
"""

import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
    ElementClickInterceptedException,
)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FanDuel full team name -> standard NBA abbreviation
TEAM_FULL_TO_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

TEAM_ABBREV_TO_FULL = {}
for full, abbrev in TEAM_FULL_TO_ABBREV.items():
    if abbrev not in TEAM_ABBREV_TO_FULL:
        TEAM_ABBREV_TO_FULL[abbrev] = full


def normalize_team(team_str: str) -> str:
    """Accept either full name or abbreviation, return abbreviation."""
    team_str = team_str.strip()
    if team_str.upper() in TEAM_ABBREV_TO_FULL:
        return team_str.upper()
    for full, abbrev in TEAM_FULL_TO_ABBREV.items():
        if full.lower() == team_str.lower():
            return abbrev
        # Partial match: "Knicks" matches "New York Knicks"
        if team_str.lower() in full.lower():
            return abbrev
    return team_str.upper()


class FanDuelOddsScraper:
    """
    Scrapes first basket scorer odds from FanDuel Sportsbook.

    FanDuel URL patterns:
      - NBA landing: https://sportsbook.fanduel.com/navigation/nba
      - Game page:   https://sportsbook.fanduel.com/basketball/nba/{game-slug}
    """

    BASE_URL = "https://sportsbook.fanduel.com"
    NBA_URL = f"{BASE_URL}/navigation/nba"

    def __init__(
        self,
        headless: bool = True,
        cache_dir: str = "data/odds_cache",
        cache_ttl_minutes: int = 30,
    ):
        self.headless = headless
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_minutes
        self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def _create_driver(self) -> webdriver.Chrome:
        """Create Chrome WebDriver with anti-detection settings."""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        # Remove webdriver property
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
            },
        )

        driver.set_page_load_timeout(30)
        logger.info(f"Chrome WebDriver initialized (headless={self.headless})")
        return driver

    def _ensure_driver(self):
        """Ensure driver is initialized."""
        if self.driver is None:
            self.driver = self._create_driver()

    def _navigate(self, url: str, wait_seconds: float = 3.0) -> bool:
        """Navigate to URL and wait for page load."""
        self._ensure_driver()
        try:
            self.driver.get(url)
            time.sleep(wait_seconds)
            return True
        except TimeoutException:
            logger.warning(f"Page load timeout for {url}")
            time.sleep(2)
            return True  # Page may still be partially loaded
        except WebDriverException as e:
            logger.error(f"WebDriver error navigating to {url}: {e}")
            return False

    def _dismiss_modals(self):
        """Dismiss any age verification or cookie modals."""
        try:
            # Look for common modal dismiss buttons
            dismiss_selectors = [
                "//button[contains(text(), 'I am')]",
                "//button[contains(text(), 'Accept')]",
                "//button[contains(text(), 'OK')]",
                "//button[contains(text(), 'Continue')]",
                "//button[@aria-label='Close']",
                "//button[contains(@class, 'close')]",
            ]
            for selector in dismiss_selectors:
                try:
                    btn = self.driver.find_element(By.XPATH, selector)
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
                        logger.info(f"Dismissed modal with selector: {selector}")
                        break
                except (NoSuchElementException, ElementClickInterceptedException):
                    continue
        except Exception:
            pass

    def _check_geo_restriction(self) -> bool:
        """Check if we're geo-restricted from FanDuel."""
        page_source = self.driver.page_source.lower()
        geo_indicators = [
            "not available in your state",
            "not available in your location",
            "geo-restricted",
            "not licensed",
            "location restricted",
        ]
        for indicator in geo_indicators:
            if indicator in page_source:
                logger.error(
                    "FanDuel is geo-restricted in your location. "
                    "FanDuel Sportsbook is only available in licensed US states."
                )
                return True
        return False

    def get_todays_games(self) -> List[Dict]:
        """
        Navigate to FanDuel NBA page and extract today's games.

        Returns list of dicts with keys:
            - home_team: Full team name
            - away_team: Full team name
            - home_abbrev: Team abbreviation
            - away_abbrev: Team abbreviation
            - game_url: URL path to game page
            - game_time: Displayed game time string
        """
        logger.info("Fetching today's NBA games from FanDuel...")

        if not self._navigate(self.NBA_URL, wait_seconds=5):
            return []

        self._dismiss_modals()

        if self._check_geo_restriction():
            return []

        # Wait for game links to appear
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/basketball/nba/']"))
            )
        except TimeoutException:
            logger.warning("No NBA game links found on page. Trying alternative selectors...")

        games = []

        # Strategy 1: Extract from links containing /basketball/nba/
        try:
            game_links = self.driver.find_elements(
                By.CSS_SELECTOR, "a[href*='/basketball/nba/']"
            )

            seen_urls = set()
            for link in game_links:
                href = link.get_attribute("href") or ""
                if "/basketball/nba/" not in href or href in seen_urls:
                    continue
                # Skip navigation/category links
                if href.endswith("/nba") or href.endswith("/nba/"):
                    continue

                seen_urls.add(href)
                text = link.text.strip()

                # Try to parse team names from link text
                game_info = self._parse_game_from_element(link, href)
                if game_info:
                    games.append(game_info)

        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")

        # Strategy 2: Extract from page via JavaScript
        if not games:
            try:
                games = self._extract_games_via_js()
            except Exception as e:
                logger.warning(f"Strategy 2 (JS extraction) failed: {e}")

        # Strategy 3: Parse page source with regex
        if not games:
            try:
                games = self._extract_games_from_source()
            except Exception as e:
                logger.warning(f"Strategy 3 (source parsing) failed: {e}")

        logger.info(f"Found {len(games)} NBA games on FanDuel")
        for g in games:
            logger.info(f"  {g['away_abbrev']} @ {g['home_abbrev']} - {g.get('game_time', 'TBD')}")

        return games

    def _parse_game_from_element(self, link_element, href: str) -> Optional[Dict]:
        """Parse game info from a link element."""
        text = link_element.text.strip()
        if not text:
            return None

        # FanDuel game links typically contain team names
        # Try to find two team names in the text
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        away_team = None
        home_team = None
        game_time = ""

        # Look for team names in lines
        team_lines = []
        for line in lines:
            matched = self._match_team_name(line)
            if matched:
                team_lines.append(matched)
            elif re.search(r'\d{1,2}:\d{2}', line):
                game_time = line

        if len(team_lines) >= 2:
            # FanDuel typically lists away team first, home team second
            away_team = team_lines[0]
            home_team = team_lines[1]
        elif len(team_lines) == 1:
            # Try parsing from URL slug
            slug_teams = self._parse_teams_from_url(href)
            if slug_teams:
                away_team, home_team = slug_teams
            else:
                return None
        else:
            # Try parsing from URL slug
            slug_teams = self._parse_teams_from_url(href)
            if slug_teams:
                away_team, home_team = slug_teams
            else:
                return None

        if away_team and home_team:
            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_abbrev": normalize_team(home_team),
                "away_abbrev": normalize_team(away_team),
                "game_url": href,
                "game_time": game_time,
            }
        return None

    def _match_team_name(self, text: str) -> Optional[str]:
        """Check if text contains an NBA team name."""
        text_lower = text.lower().strip()
        for full_name in TEAM_FULL_TO_ABBREV:
            if full_name.lower() in text_lower:
                return full_name
            # Check last word (e.g., "Knicks", "Celtics")
            team_parts = full_name.lower().split()
            if team_parts[-1] in text_lower:
                return full_name
        return None

    def _parse_teams_from_url(self, url: str) -> Optional[tuple]:
        """Try to parse team names from FanDuel URL slug."""
        # URL pattern: .../basketball/nba/celtics-@-knicks-123456
        match = re.search(r"/basketball/nba/([^?]+)", url)
        if not match:
            return None

        slug = match.group(1).lower()

        # Try to find team names in slug
        found_teams = []
        for full_name, abbrev in TEAM_FULL_TO_ABBREV.items():
            # Check for last word of team name in slug
            team_word = full_name.split()[-1].lower()
            if team_word in slug:
                found_teams.append(full_name)

        if len(found_teams) >= 2:
            # In FanDuel URLs, away team is listed first
            return found_teams[0], found_teams[1]

        return None

    def _extract_games_via_js(self) -> List[Dict]:
        """Extract games using JavaScript to query the DOM."""
        script = """
        const games = [];
        // Look for event cards/containers with NBA game links
        const links = document.querySelectorAll('a[href*="/basketball/nba/"]');
        links.forEach(link => {
            const href = link.getAttribute('href');
            if (!href || href.endsWith('/nba') || href.endsWith('/nba/')) return;
            const text = link.innerText.trim();
            if (text) {
                games.push({url: href, text: text});
            }
        });
        return games;
        """
        results = self.driver.execute_script(script)
        games = []
        seen = set()
        for r in results or []:
            url = r.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            text = r.get("text", "")
            # Parse teams from text
            info = self._parse_game_text(text, url)
            if info:
                games.append(info)
        return games

    def _parse_game_text(self, text: str, url: str) -> Optional[Dict]:
        """Parse game info from extracted text content."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        team_names = []
        game_time = ""

        for line in lines:
            matched = self._match_team_name(line)
            if matched and matched not in team_names:
                team_names.append(matched)
            elif re.search(r'\d{1,2}:\d{2}', line):
                game_time = line

        if len(team_names) >= 2:
            return {
                "home_team": team_names[1],
                "away_team": team_names[0],
                "home_abbrev": normalize_team(team_names[1]),
                "away_abbrev": normalize_team(team_names[0]),
                "game_url": url if url.startswith("http") else self.BASE_URL + url,
                "game_time": game_time,
            }
        return None

    def _extract_games_from_source(self) -> List[Dict]:
        """Extract games by parsing page source for game URLs."""
        source = self.driver.page_source
        # Find all NBA game URLs
        urls = re.findall(r'href="(/basketball/nba/[^"]+)"', source)
        games = []
        seen = set()

        for url in urls:
            if url in seen or url.endswith("/nba") or url.endswith("/nba/"):
                continue
            seen.add(url)
            teams = self._parse_teams_from_url(url)
            if teams:
                away, home = teams
                games.append({
                    "home_team": home,
                    "away_team": away,
                    "home_abbrev": normalize_team(home),
                    "away_abbrev": normalize_team(away),
                    "game_url": self.BASE_URL + url,
                    "game_time": "",
                })
        return games

    def get_first_basket_odds(self, game_url: str) -> Optional[Dict[str, int]]:
        """
        Navigate to a game page and extract first basket scorer odds.

        Args:
            game_url: Full URL or path to the game page

        Returns:
            Dict mapping player names to American odds, e.g.:
            {'Karl-Anthony Towns': 490, 'Jalen Brunson': 460, ...}
            Returns None if market not found.
        """
        if not game_url.startswith("http"):
            game_url = self.BASE_URL + game_url

        logger.info(f"Fetching first basket scorer odds from {game_url}")

        if not self._navigate(game_url, wait_seconds=4):
            return None

        self._dismiss_modals()

        # Step 1: Navigate to Player Props tab
        if not self._click_player_props_tab():
            logger.warning("Could not find Player Props tab")
            # Still try - might be on the right page already

        time.sleep(2)

        # Step 2: Find and expand "First Basket Scorer" section
        if not self._find_first_basket_section():
            logger.warning("Could not find First Basket Scorer section")
            return None

        time.sleep(1)

        # Step 3: Extract odds
        odds = self._extract_odds()

        if odds:
            logger.info(f"Extracted odds for {len(odds)} players")
        else:
            logger.warning("No odds extracted")

        return odds

    def _click_player_props_tab(self) -> bool:
        """Click the Player Props tab on a game page."""
        tab_selectors = [
            "//a[contains(text(), 'Player Props')]",
            "//button[contains(text(), 'Player Props')]",
            "//span[contains(text(), 'Player Props')]/..",
            "//div[contains(text(), 'Player Props')]",
            "//a[contains(@href, 'player-props')]",
            "//nav//a[contains(text(), 'Player')]",
        ]

        for selector in tab_selectors:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                element.click()
                logger.info("Clicked Player Props tab")
                time.sleep(2)
                return True
            except (TimeoutException, ElementClickInterceptedException):
                continue

        # Try via JavaScript click as fallback
        try:
            clicked = self.driver.execute_script("""
                const links = document.querySelectorAll('a, button, [role="tab"]');
                for (const el of links) {
                    if (el.textContent.includes('Player Props') || el.textContent.includes('Player props')) {
                        el.click();
                        return true;
                    }
                }
                return false;
            """)
            if clicked:
                logger.info("Clicked Player Props tab via JS")
                time.sleep(2)
                return True
        except Exception:
            pass

        return False

    def _find_first_basket_section(self) -> bool:
        """Find and scroll to the First Basket Scorer section."""
        section_selectors = [
            "//span[contains(text(), 'First Basket Scorer')]",
            "//h3[contains(text(), 'First Basket')]",
            "//div[contains(text(), 'First Basket Scorer')]",
            "//*[contains(text(), '1st Basket Scorer')]",
            "//*[contains(text(), 'First Basket')]",
        ]

        for selector in section_selectors:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                # Scroll into view
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'})",
                    element,
                )
                time.sleep(1)

                # Try to expand if it's collapsed
                try:
                    element.click()
                    time.sleep(1)
                except (ElementClickInterceptedException, WebDriverException):
                    pass

                logger.info("Found First Basket Scorer section")
                return True
            except TimeoutException:
                continue

        # Try scrolling down to find it
        logger.info("Scrolling to find First Basket Scorer section...")
        for _ in range(10):
            self.driver.execute_script("window.scrollBy(0, 500)")
            time.sleep(0.5)

            # Check if section appeared
            try:
                for selector in section_selectors:
                    el = self.driver.find_element(By.XPATH, selector)
                    if el.is_displayed():
                        self.driver.execute_script(
                            "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'})",
                            el,
                        )
                        time.sleep(1)
                        logger.info("Found First Basket Scorer section after scrolling")
                        return True
            except NoSuchElementException:
                continue

        return False

    def _extract_odds(self) -> Optional[Dict[str, int]]:
        """Extract player names and odds from the page."""
        odds = {}

        # Strategy 1: JavaScript DOM extraction
        try:
            js_odds = self._extract_odds_via_js()
            if js_odds:
                return js_odds
        except Exception as e:
            logger.debug(f"JS odds extraction failed: {e}")

        # Strategy 2: Find elements with odds patterns
        try:
            pattern_odds = self._extract_odds_via_elements()
            if pattern_odds:
                return pattern_odds
        except Exception as e:
            logger.debug(f"Element odds extraction failed: {e}")

        # Strategy 3: Parse page source
        try:
            source_odds = self._extract_odds_from_source()
            if source_odds:
                return source_odds
        except Exception as e:
            logger.debug(f"Source odds extraction failed: {e}")

        return None

    def _extract_odds_via_js(self) -> Optional[Dict[str, int]]:
        """Extract odds using JavaScript to find player-odds pairs."""
        # Try __NEXT_DATA__ first (common in Next.js apps like FanDuel)
        try:
            next_data = self.driver.execute_script("return window.__NEXT_DATA__")
            if next_data:
                odds = self._parse_next_data(next_data)
                if odds:
                    return odds
        except Exception:
            pass

        # Try querying the DOM for odds containers
        script = """
        const results = [];

        // Strategy A: Look for selection containers with name + odds pattern
        // FanDuel typically renders each selection as a button or div
        const allButtons = document.querySelectorAll(
            'button, [role="button"], [class*="selection"], [class*="outcome"]'
        );

        for (const btn of allButtons) {
            const text = btn.innerText.trim();
            // Look for pattern: "Player Name\\n+450" or "Player Name +450"
            const match = text.match(/^(.+?)\\s*\\n?\\s*([+-]\\d{3,4})$/);
            if (match) {
                results.push({name: match[1].trim(), odds: match[2]});
            }
        }

        // Strategy B: Look for paired name/odds elements
        if (results.length === 0) {
            const containers = document.querySelectorAll(
                '[class*="market-row"], [class*="runner"], [class*="participant"]'
            );
            for (const container of containers) {
                const nameEl = container.querySelector(
                    '[class*="name"], [class*="label"], [class*="participant"], span'
                );
                const oddsEl = container.querySelector(
                    '[class*="odds"], [class*="price"], [class*="american"]'
                );
                if (nameEl && oddsEl) {
                    const name = nameEl.innerText.trim();
                    const odds = oddsEl.innerText.trim();
                    if (name && /^[+-]?\\d{3,4}$/.test(odds.replace('+', '').replace('-', ''))) {
                        results.push({name: name, odds: odds.startsWith('-') || odds.startsWith('+') ? odds : '+' + odds});
                    }
                }
            }
        }

        return results;
        """

        results = self.driver.execute_script(script)
        if not results:
            return None

        odds = {}
        for r in results:
            name = r.get("name", "").strip()
            odds_str = r.get("odds", "")
            if name and odds_str:
                parsed = self._parse_american_odds(odds_str)
                if parsed is not None:
                    odds[name] = parsed

        return odds if odds else None

    def _parse_next_data(self, data: dict) -> Optional[Dict[str, int]]:
        """Try to extract first basket scorer odds from __NEXT_DATA__."""
        try:
            # Walk the data structure looking for first basket scorer market
            # This is heuristic and may need adjustment
            def search(obj, depth=0):
                if depth > 10:
                    return None
                if isinstance(obj, dict):
                    # Look for market with "first basket" in name
                    name = obj.get("marketName", obj.get("name", "")).lower()
                    if "first basket" in name:
                        runners = obj.get("runners", obj.get("selections", []))
                        odds = {}
                        for runner in runners:
                            player_name = runner.get("runnerName", runner.get("name", ""))
                            handicap = runner.get("winRunnerOdds", {})
                            american = handicap.get("americanOdds", handicap.get("american"))
                            if player_name and american:
                                odds[player_name] = int(american)
                        if odds:
                            return odds

                    for v in obj.values():
                        result = search(v, depth + 1)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = search(item, depth + 1)
                        if result:
                            return result
                return None

            return search(data)
        except Exception:
            return None

    def _extract_odds_via_elements(self) -> Optional[Dict[str, int]]:
        """Extract odds by finding DOM elements near the First Basket section."""
        odds = {}

        # Find the section header
        section = None
        for text in ["First Basket Scorer", "1st Basket Scorer", "First Basket"]:
            try:
                elements = self.driver.find_elements(
                    By.XPATH, f"//*[contains(text(), '{text}')]"
                )
                if elements:
                    section = elements[0]
                    break
            except Exception:
                continue

        if not section:
            return None

        # Get the parent container
        try:
            # Walk up to find the market container
            container = self.driver.execute_script("""
                let el = arguments[0];
                for (let i = 0; i < 5; i++) {
                    el = el.parentElement;
                    if (!el) break;
                    // Look for a container that has multiple buttons/selections
                    const buttons = el.querySelectorAll('button, [role="button"]');
                    if (buttons.length >= 3) return el;
                }
                return null;
            """, section)

            if container:
                # Extract player-odds pairs from this container
                pairs = self.driver.execute_script("""
                    const container = arguments[0];
                    const results = [];
                    const buttons = container.querySelectorAll('button, [role="button"], [class*="selection"]');
                    for (const btn of buttons) {
                        const text = btn.innerText.trim();
                        const lines = text.split('\\n').map(l => l.trim()).filter(l => l);
                        if (lines.length >= 2) {
                            const name = lines[0];
                            const oddsText = lines[lines.length - 1];
                            if (/^[+-]?\\d{3,4}$/.test(oddsText.replace('+','').replace('-',''))) {
                                results.push({name: name, odds: oddsText});
                            }
                        }
                    }
                    return results;
                """, container)

                for pair in pairs or []:
                    name = pair.get("name", "").strip()
                    odds_str = pair.get("odds", "")
                    parsed = self._parse_american_odds(odds_str)
                    if name and parsed is not None:
                        odds[name] = parsed

        except Exception as e:
            logger.debug(f"Container extraction failed: {e}")

        return odds if odds else None

    def _extract_odds_from_source(self) -> Optional[Dict[str, int]]:
        """Last resort: parse page source for odds patterns."""
        source = self.driver.page_source
        odds = {}

        # Look for JSON-like odds data in the page
        # FanDuel often embeds market data in script tags
        pattern = r'"runnerName"\s*:\s*"([^"]+)".*?"americanOdds"\s*:\s*"?([+-]?\d+)"?'
        matches = re.findall(pattern, source, re.DOTALL)

        for name, odds_str in matches:
            parsed = self._parse_american_odds(odds_str)
            if parsed is not None:
                odds[name] = parsed

        return odds if odds else None

    @staticmethod
    def _parse_american_odds(odds_str: str) -> Optional[int]:
        """Parse American odds string to integer."""
        odds_str = odds_str.strip()
        if odds_str.upper() == "EVEN":
            return 100

        # Remove '+' prefix for positive odds
        try:
            val = int(odds_str.replace("+", ""))
            # Sanity check: American odds are typically 100-50000 or -50000 to -100
            if abs(val) >= 100:
                return val
        except ValueError:
            pass
        return None

    def get_odds_for_game(
        self, home_team: str, away_team: str
    ) -> Optional[Dict[str, int]]:
        """
        High-level method: find a specific game and return its first basket odds.

        Args:
            home_team: Team abbreviation (e.g., "NYK") or full name
            away_team: Team abbreviation (e.g., "BOS") or full name

        Returns:
            Dict[str, int] mapping player names to American odds, or None
        """
        home_abbrev = normalize_team(home_team)
        away_abbrev = normalize_team(away_team)

        # Check cache first
        cached = self._load_cached_odds(home_abbrev, away_abbrev)
        if cached is not None:
            logger.info(f"Using cached odds for {away_abbrev} @ {home_abbrev}")
            return cached

        # Fetch games list
        games = self.get_todays_games()
        if not games:
            logger.error("No NBA games found on FanDuel")
            return None

        # Find the matching game
        target_game = None
        for game in games:
            if game["home_abbrev"] == home_abbrev and game["away_abbrev"] == away_abbrev:
                target_game = game
                break

        if not target_game:
            # Try reverse (in case home/away mapping is off)
            for game in games:
                if (home_abbrev in (game["home_abbrev"], game["away_abbrev"]) and
                    away_abbrev in (game["home_abbrev"], game["away_abbrev"])):
                    target_game = game
                    break

        if not target_game:
            available = [g["away_abbrev"] + " @ " + g["home_abbrev"] for g in games]
            logger.error(
                f"Game {away_abbrev} @ {home_abbrev} not found. "
                f"Available: {available}"
            )
            return None

        # Fetch odds
        odds = self.get_first_basket_odds(target_game["game_url"])

        if odds:
            self._cache_odds(home_abbrev, away_abbrev, odds)

        return odds

    def get_all_games_odds(self) -> Dict[str, Dict[str, int]]:
        """
        Get first basket scorer odds for all of today's NBA games.

        Returns dict keyed by "AWAY @ HOME" -> odds dict.
        """
        games = self.get_todays_games()
        all_odds = {}

        for game in games:
            key = f"{game['away_abbrev']} @ {game['home_abbrev']}"
            logger.info(f"\nFetching odds for {key}...")

            # Check cache
            cached = self._load_cached_odds(game["home_abbrev"], game["away_abbrev"])
            if cached:
                all_odds[key] = cached
                logger.info(f"  Loaded from cache ({len(cached)} players)")
                continue

            odds = self.get_first_basket_odds(game["game_url"])
            if odds:
                all_odds[key] = odds
                self._cache_odds(game["home_abbrev"], game["away_abbrev"], odds)
                logger.info(f"  Found odds for {len(odds)} players")
            else:
                logger.warning(f"  No first basket scorer odds available for {key}")

            time.sleep(2)  # Be polite between games

        return all_odds

    def _cache_odds(self, home_abbrev: str, away_abbrev: str, odds: Dict[str, int]):
        """Cache odds to JSON file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"{date_str}_{away_abbrev}_{home_abbrev}.json"

        data = {
            "fetched_at": datetime.now().isoformat(),
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "odds": odds,
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Cached odds to {cache_file}")

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
                logger.info(f"Cache expired ({age_minutes:.0f}m old, TTL={self.cache_ttl}m)")
                return None

            return data["odds"]

        except (json.JSONDecodeError, KeyError, ValueError):
            return None
