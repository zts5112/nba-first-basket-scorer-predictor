"""
Player Name Matcher

Matches FanDuel player names to Basketball Reference player names/IDs
stored in the player tokenizer.
"""

import json
import difflib
import unicodedata
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)


# Known aliases: FanDuel name -> BBRef name
KNOWN_ALIASES = {
    "Nic Claxton": "Nicolas Claxton",
    "Herb Jones": "Herbert Jones",
    "Cam Thomas": "Cameron Thomas",
    "Cam Johnson": "Cameron Johnson",
    "PJ Washington": "P.J. Washington",
    "CJ McCollum": "C.J. McCollum",
    "RJ Barrett": "R.J. Barrett",
    "OG Anunoby": "O.G. Anunoby",
    "AJ Green": "A.J. Green",
    "TJ McConnell": "T.J. McConnell",
    "TJ Warren": "T.J. Warren",
    "KJ Martin": "K.J. Martin",
    "JT Thor": "J.T. Thor",
    "GG Jackson": "G.G. Jackson",
    "Kenyon Martin": "Kenyon Martin Jr.",
    "Gary Trent": "Gary Trent Jr.",
    "Tim Hardaway": "Tim Hardaway Jr.",
    "Larry Nance": "Larry Nance Jr.",
    "Wendell Carter": "Wendell Carter Jr.",
    "Kevin Porter": "Kevin Porter Jr.",
    "Kelly Oubre": "Kelly Oubre Jr.",
    "Marcus Morris": "Marcus Morris Sr.",
    "Derrick Jones": "Derrick Jones Jr.",
    "Dennis Smith": "Dennis Smith Jr.",
    "Jabari Smith": "Jabari Smith Jr.",
    "Troy Brown": "Troy Brown Jr.",
    "Jaren Jackson": "Jaren Jackson Jr.",
    "Robert Williams": "Robert Williams III",
    "Lonnie Walker": "Lonnie Walker IV",
    "Trey Murphy": "Trey Murphy III",
    "Santi Aldama": "Santi Aldama",
    "Alexandre Sarr": "Alex Sarr",
    "Naz Reid": "Naz Reid",
}


class PlayerNameMatcher:
    """Matches FanDuel player names to the model's canonical BBRef names."""

    def __init__(self, tokenizer_path: str = "data/processed/player_tokenizer.json"):
        self.tokenizer_path = Path(tokenizer_path)
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load player name mappings from tokenizer."""
        with open(self.tokenizer_path) as f:
            data = json.load(f)

        self.name_to_id: Dict[str, str] = data.get("player_name_to_id", {})
        self.id_to_name: Dict[str, str] = data.get("player_id_to_name", {})

        # Build normalized lookup
        self._normalized_lookup: Dict[str, str] = {}
        for name in self.name_to_id:
            norm = self._normalize(name)
            self._normalized_lookup[norm] = name

        logger.info(f"Loaded {len(self.name_to_id)} players from tokenizer")

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a name for matching: lowercase, strip accents, remove periods."""
        # Remove accents/diacritics
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        # Lowercase, strip, remove periods
        name = name.lower().strip().replace(".", "")
        return name

    def match_name(self, fanduel_name: str) -> Optional[Tuple[str, str]]:
        """
        Match a FanDuel name to (player_id, canonical_name).

        Returns None if no match found.
        """
        # 0. Check known aliases first
        if fanduel_name in KNOWN_ALIASES:
            alias = KNOWN_ALIASES[fanduel_name]
            if alias in self.name_to_id:
                return self.name_to_id[alias], alias

        # 1. Exact match
        if fanduel_name in self.name_to_id:
            return self.name_to_id[fanduel_name], fanduel_name

        # 2. Normalized exact match
        norm_fd = self._normalize(fanduel_name)
        if norm_fd in self._normalized_lookup:
            canonical = self._normalized_lookup[norm_fd]
            return self.name_to_id[canonical], canonical

        # 3. Fuzzy match
        all_normalized = list(self._normalized_lookup.keys())
        matches = difflib.get_close_matches(norm_fd, all_normalized, n=1, cutoff=0.80)
        if matches:
            canonical = self._normalized_lookup[matches[0]]
            logger.info(f"Fuzzy matched '{fanduel_name}' -> '{canonical}'")
            return self.name_to_id[canonical], canonical

        # 4. Last name + first initial match
        parts = fanduel_name.strip().split()
        if len(parts) >= 2:
            last = parts[-1].lower()
            first_init = parts[0][0].lower()
            for name in self.name_to_id:
                name_parts = name.strip().split()
                if len(name_parts) >= 2:
                    if name_parts[-1].lower() == last and name_parts[0][0].lower() == first_init:
                        logger.info(f"Last+initial matched '{fanduel_name}' -> '{name}'")
                        return self.name_to_id[name], name

        logger.warning(f"Could not match FanDuel name: '{fanduel_name}'")
        return None

    def match_odds_dict(
        self, fanduel_odds: Dict[str, int]
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Match a full FanDuel odds dict to canonical model player names.

        Returns:
            - matched_odds: Dict[str, int] using canonical BBRef names
            - unmatched: List of FanDuel names that couldn't be matched
        """
        matched = {}
        unmatched = []

        for fd_name, odds in fanduel_odds.items():
            result = self.match_name(fd_name)
            if result:
                _, canonical_name = result
                matched[canonical_name] = odds
            else:
                # Use the FanDuel name as-is (may still work for display)
                matched[fd_name] = odds
                unmatched.append(fd_name)

        if unmatched:
            logger.warning(f"Could not match {len(unmatched)} players: {unmatched}")

        return matched, unmatched
