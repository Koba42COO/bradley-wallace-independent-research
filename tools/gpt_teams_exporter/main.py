#!/usr/bin/env python3
"""
GPT Teams Archive Exporter
Exports conversations from ChatGPT Teams and integrates them into AIVA memory.

Usage:
    python tools/gpt_teams_exporter/main.py --since 2024-01-01 --limit 500 --headful

Options:
    --dry-run                 Show what would be exported without writing files
    --since DATE              Only export conversations after this date (YYYY-MM-DD)
    --limit N                 Maximum conversations to export
    --headful                 Run browser in headed mode (visible)
    --only-team NAME          Only export from specific team workspace
    --include-personal        Include personal conversations (sanitized)
    --max-concurrency N       Maximum concurrent requests (default: 3)
    --retry N                 Number of retries on failure (default: 3)
    --seed-folder PATH        Import from local folder instead of scraping
    --web-ui                  Start web UI for control (default port 8765)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
import time as time_module
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse

import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify
from markdown import markdown
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

# Add parent directory to path for aiva-core imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Lazy import AIVA modules to make them optional
try:
    from aiva_core.core.memory_bank import AIVAMemoryBank
    from aiva_core.core.delta_memory import DeltaMemory
    AIVA_AVAILABLE = True
except ImportError:
    AIVA_AVAILABLE = False
    AIVAMemoryBank = None
    DeltaMemory = None


# Load environment variables
load_dotenv()

# Constants
DEFAULT_CONFIG = {
    'CHATGPT_EMAIL': os.getenv('CHATGPT_EMAIL'),
    'CHATGPT_PASSWORD': os.getenv('CHATGPT_PASSWORD'),
    'CHATGPT_TEAM_NAME': os.getenv('CHATGPT_TEAM_NAME'),
    'SCRAPE_HEADFUL': os.getenv('SCRAPE_HEADFUL', 'true').lower() == 'true',
    'SEED_FOLDER': os.getenv('SEED_FOLDER'),
    'EXPORT_SINCE': os.getenv('EXPORT_SINCE', '2024-01-01'),
    'EXPORT_LIMIT': int(os.getenv('EXPORT_LIMIT', '500')),
    'INCLUDE_PERSONAL_SANITIZED': os.getenv('INCLUDE_PERSONAL_SANITIZED', 'false').lower() == 'true',
    'CLASSIFICATION_CONFIG': os.getenv('CLASSIFICATION_CONFIG', 'configs/gpt_scraper_classification.yaml'),
    'ARTIFACTS_DIR': os.getenv('ARTIFACTS_DIR', 'artifacts/gpt_convos'),
    'AIVA_MEMORY_DIR': os.getenv('AIVA_MEMORY_DIR', 'aiva-core/data/memories'),
    'RUN_DATA_DIR': os.getenv('RUN_DATA_DIR', 'artifacts/run-data'),
}

# Global state for web UI
global_state = {
    'running': False,
    'progress': {'current': 0, 'total': 0, 'message': ''},
    'last_run': None,
    'errors': []
}


class GPTTeamsExporter:
    """Main exporter class for GPT Teams conversations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Create output directories first (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        self.artifacts_dir = project_root / config['ARTIFACTS_DIR']
        self.raw_dir = self.artifacts_dir / 'raw'
        self.raw_md_dir = self.artifacts_dir / 'raw_md'
        self.science_dir = self.artifacts_dir / 'science'
        self.personal_dir = self.artifacts_dir / 'personal_sanitized'
        self.run_data_dir = project_root / config['RUN_DATA_DIR']

        # Setup logging after directories are created
        self.logger = self._setup_logging()
        self.classification_config = self._load_classification_config()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        for dir_path in [self.artifacts_dir, self.raw_dir, self.raw_md_dir,
                        self.science_dir, self.personal_dir, self.run_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Index for deduplication
        self.index_file = self.artifacts_dir / 'index.jsonl'
        self.processed_ids: Set[str] = self._load_processed_ids()

        # AIVA memory integration (optional)
        # Path should be relative to project root, not script location
        project_root = Path(__file__).parent.parent.parent
        self.aiva_memory_dir = project_root / config['AIVA_MEMORY_DIR']
        self.memory_bank = None
        if AIVA_AVAILABLE:
            try:
                self.memory_bank = AIVAMemoryBank(str(self.aiva_memory_dir))
                self.logger.info("AIVA memory integration enabled")
            except Exception as e:
                self.logger.warning(f"AIVA memory integration failed: {e}")
        else:
            self.logger.info("AIVA memory integration not available")

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger('gpt_teams_exporter')
        logger.setLevel(logging.INFO)

        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        log_file = self.run_data_dir / 'gpt_teams_export.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _load_classification_config(self) -> Dict[str, Any]:
        """Load classification configuration from YAML."""
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / self.config['CLASSIFICATION_CONFIG']
        if not config_path.exists():
            self.logger.warning(f"Classification config not found: {config_path}")
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_processed_ids(self) -> Set[str]:
        """Load set of already processed conversation IDs."""
        processed_ids = set()
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        processed_ids.add(entry['id'])
                    except json.JSONDecodeError:
                        continue
        return processed_ids

    def _save_to_index(self, conversation: Dict[str, Any]):
        """Save conversation metadata to index for deduplication."""
        entry = {
            'id': conversation['id'],
            'title': conversation.get('title', 'Untitled'),
            'created_at': conversation.get('create_time'),
            'updated_at': conversation.get('update_time'),
            'message_count': len(conversation.get('mapping', {})),
            'model': conversation.get('current_node', {}).get('message', {}).get('metadata', {}).get('model_slug'),
            'exported_at': datetime.now().isoformat(),
            'classification': getattr(conversation, '_classification', 'unknown')
        }

        with open(self.index_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')

        self.processed_ids.add(conversation['id'])

    async def run(self, dry_run: bool = False, limit: Optional[int] = None,
                  since: Optional[str] = None) -> Dict[str, Any]:
        """Main export execution."""
        self.logger.info("Starting GPT Teams export")

        try:
            # Initialize browser if not using seed folder
            if not self.config.get('SEED_FOLDER'):
                await self._init_browser()

            # Get conversations
            conversations = await self._get_conversations(limit=limit, since=since)

            # Process conversations
            results = await self._process_conversations(conversations, dry_run=dry_run)

            # Update AIVA memory
            if not dry_run:
                self._update_aiva_memory(results)

            self.logger.info(f"Export completed. Processed {len(results)} conversations")
            return {'status': 'success', 'processed': len(results), 'results': results}

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
        finally:
            await self._cleanup()

    async def _init_browser(self):
        """Initialize Playwright browser with persistent context."""
        playwright = await async_playwright().start()

        # Use persistent context to maintain login state
        user_data_dir = self.run_data_dir / 'browser_data'
        user_data_dir.mkdir(exist_ok=True)

        self.browser = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=not self.config['SCRAPE_HEADFUL'],
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )

        self.page = await self.browser.new_page()

    async def _get_conversations(self, limit: Optional[int] = None,
                               since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversations from GPT Teams or seed folder."""
        if self.config.get('SEED_FOLDER'):
            return self._get_conversations_from_seed()

        # Browser-based scraping
        await self.page.goto('https://chatgpt.com')
        await self._ensure_logged_in()

        # Switch to team workspace if specified
        if self.config.get('CHATGPT_TEAM_NAME'):
            await self._switch_to_team(self.config['CHATGPT_TEAM_NAME'])

        # Enumerate conversations
        conversations = await self._enumerate_conversations(limit=limit, since=since)

        self.logger.info(f"Found {len(conversations)} conversations to process")

        return conversations

    async def _switch_to_team(self, team_name: str):
        """Switch to specified team workspace."""
        try:
            # Look for team selector
            team_selector = await self.page.query_selector('[data-testid="workspace-selector"]')
            if team_selector:
                await team_selector.click()

                # Wait for team options and click the specified team
                team_option = await self.page.wait_for_selector(f'text="{team_name}"', timeout=5000)
                await team_option.click()

                await self.page.wait_for_load_state('networkidle')
                self.logger.info(f"Switched to team: {team_name}")
            else:
                self.logger.warning("Team selector not found, staying on default workspace")

        except Exception as e:
            self.logger.warning(f"Could not switch to team {team_name}: {e}")

    async def _enumerate_conversations(self, limit: Optional[int] = None,
                                     since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enumerate conversations from the sidebar."""
        conversations = []
        since_date = datetime.fromisoformat(since) if since else None

        try:
            # Wait for sidebar to load
            await self.page.wait_for_selector('[data-testid="conversation-list"]', timeout=10000)

            # Scroll and collect conversation links
            last_height = 0
            scroll_attempts = 0
            max_scrolls = 50  # Prevent infinite scrolling

            while scroll_attempts < max_scrolls:
                # Get current conversation items
                conv_items = await self.page.query_selector_all('[data-testid="conversation-item"]')

                for item in conv_items[len(conversations):]:  # Only process new items
                    try:
                        # Extract conversation metadata from DOM
                        conv_data = await self._extract_conversation_metadata(item)
                        if conv_data:
                            # Check date filter
                            if since_date and conv_data.get('create_time'):
                                conv_date = datetime.fromtimestamp(conv_data['create_time'])
                                if conv_date < since_date:
                                    continue

                            conversations.append(conv_data)

                            # Check limit
                            if limit and len(conversations) >= limit:
                                return conversations

                    except Exception as e:
                        self.logger.warning(f"Error extracting conversation metadata: {e}")
                        continue

                # Scroll down to load more
                current_height = await self.page.evaluate("""
                    () => {
                        const sidebar = document.querySelector('[data-testid="conversation-list"]');
                        return sidebar ? sidebar.scrollHeight : 0;
                    }
                """)

                if current_height == last_height:
                    # No more content to load
                    break

                # Scroll down
                await self.page.evaluate("""
                    () => {
                        const sidebar = document.querySelector('[data-testid="conversation-list"]');
                        if (sidebar) {
                            sidebar.scrollTo(0, sidebar.scrollHeight);
                        }
                    }
                """)

                await asyncio.sleep(1)  # Wait for content to load
                last_height = current_height
                scroll_attempts += 1

                # Rate limiting
                await asyncio.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Error enumerating conversations: {e}")

        return conversations

    async def _extract_conversation_metadata(self, item) -> Optional[Dict[str, Any]]:
        """Extract conversation metadata from DOM element."""
        try:
            # Get conversation link and extract ID from href
            link = await item.query_selector('a')
            if not link:
                return None

            href = await link.get_attribute('href')
            if not href:
                return None

            # Extract conversation ID from URL
            conv_id = href.split('/')[-1]
            if not conv_id or conv_id == 'chat':
                return None

            # Get title
            title_elem = await item.query_selector('[data-testid="conversation-title"]')
            title = await title_elem.text_content() if title_elem else f"Conversation {conv_id[:8]}"

            # Get timestamp (may need to click to load)
            timestamp_elem = await item.query_selector('[data-testid="conversation-time"]')
            timestamp_text = await timestamp_elem.text_content() if timestamp_elem else ""

            # Convert relative time to timestamp
            create_time = self._parse_relative_time(timestamp_text)

            return {
                'id': conv_id,
                'title': title.strip(),
                'create_time': create_time,
                'url': f"https://chatgpt.com{href}"
            }

        except Exception as e:
            self.logger.warning(f"Error extracting metadata from conversation item: {e}")
            return None

    def _parse_relative_time(self, time_text: str) -> float:
        """Parse relative time string to Unix timestamp."""
        now = datetime.now()

        if not time_text:
            return now.timestamp()

        time_text = time_text.lower().strip()

        # Handle various formats
        if 'today' in time_text:
            # Assume morning if no time specified
            return now.replace(hour=9, minute=0, second=0).timestamp()
        elif 'yesterday' in time_text:
            yesterday = now - timedelta(days=1)
            return yesterday.replace(hour=9, minute=0, second=0).timestamp()
        elif 'hour' in time_text:
            hours_ago = int(time_text.split()[0]) if time_text.split()[0].isdigit() else 1
            return (now - timedelta(hours=hours_ago)).timestamp()
        elif 'minute' in time_text:
            mins_ago = int(time_text.split()[0]) if time_text.split()[0].isdigit() else 1
            return (now - timedelta(minutes=mins_ago)).timestamp()
        elif 'day' in time_text:
            days_ago = int(time_text.split()[0]) if time_text.split()[0].isdigit() else 1
            return (now - timedelta(days=days_ago)).timestamp()
        else:
            # Try to parse as absolute date
            try:
                return datetime.strptime(time_text, '%b %d').replace(year=now.year).timestamp()
            except:
                return now.timestamp()

    async def _fetch_conversation_data(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full conversation data via API or DOM scraping."""
        try:
            # Try API first
            api_data = await self._fetch_via_api(conv_id)
            if api_data:
                return api_data

            # Fallback to DOM scraping
            self.logger.info(f"API fetch failed for {conv_id}, trying DOM scraping")
            dom_data = await self._fetch_via_dom(conv_id)
            return dom_data

        except Exception as e:
            self.logger.error(f"Error fetching conversation {conv_id}: {e}")
            return None

    async def _fetch_via_api(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch conversation data via ChatGPT API."""
        try:
            # Get access token from browser storage
            access_token = await self.page.evaluate("""
                () => {
                    const token = localStorage.getItem('accessToken') ||
                                sessionStorage.getItem('accessToken');
                    return token;
                }
            """)

            if not access_token:
                self.logger.warning("No access token found")
                return None

            # Fetch conversation data
            api_url = f"https://chatgpt.com/backend-api/conversation/{conv_id}"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
            }

            response = await self.page.request.get(api_url, headers=headers)

            if response.status == 200:
                return await response.json()
            else:
                self.logger.warning(f"API request failed: {response.status}")
                return None

        except Exception as e:
            self.logger.warning(f"API fetch error: {e}")
            return None

    async def _fetch_via_dom(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch conversation data by navigating to the page and scraping DOM."""
        try:
            # Navigate to conversation
            conv_url = f"https://chatgpt.com/c/{conv_id}"
            await self.page.goto(conv_url)
            await self.page.wait_for_load_state('networkidle')

            # Extract conversation data from page
            # This is a fallback and may not get complete data
            conversation_data = await self.page.evaluate("""
                () => {
                    // Try to extract from React state or global variables
                    const keys = Object.keys(window);
                    for (const key of keys) {
                        if (key.includes('__NEXT_DATA__') || key.includes('conversation')) {
                            try {
                                const data = window[key];
                                if (data && typeof data === 'object' && data.conversation) {
                                    return data.conversation;
                                }
                            } catch (e) {}
                        }
                    }

                    // Fallback: extract visible messages from DOM
                    const messages = [];
                    const messageElements = document.querySelectorAll('[data-message-id]');

                    messageElements.forEach(elem => {
                        const role = elem.closest('[data-testid*="message"]')?.getAttribute('data-testid')?.includes('user') ? 'user' : 'assistant';
                        const content = elem.textContent || '';
                        if (content.trim()) {
                            messages.push({
                                role: role,
                                content: { parts: [content] },
                                create_time: Date.now() / 1000
                            });
                        }
                    });

                    return {
                        id: HOST_REDACTED_29('/').pop(),
                        title: document.title || 'Untitled',
                        mapping: {},
                        messages: messages
                    };
                }
            """)

            return conversation_data

        except Exception as e:
            self.logger.error(f"DOM scraping error: {e}")
            return None

    def _get_conversations_from_seed(self) -> List[Dict[str, Any]]:
        """Load conversations from local seed folder."""
        seed_path = Path(self.config['SEED_FOLDER'])
        conversations = []

        if not seed_path.exists():
            self.logger.warning(f"Seed folder not found: {seed_path}")
            return conversations

        self.logger.info(f"Loading conversations from seed folder: {seed_path}")

        # Look for JSON files in the seed folder
        json_files = list(seed_path.glob("**/*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)

                # Validate conversation structure
                if not isinstance(conv_data, dict) or 'id' not in conv_data:
                    self.logger.warning(f"Invalid conversation format in {json_file}")
                    continue

                # Ensure required fields
                conv_data.setdefault('title', f"Imported {conv_data['id'][:8]}")
                conv_data.setdefault('create_time', json_file.stat().st_mtime)
                conv_data.setdefault('update_time', conv_data.get('create_time', 0))

                conversations.append(conv_data)
                self.logger.debug(f"Loaded conversation: {conv_data['id']}")

            except json.JSONDecodeError as e:
                self.logger.warning(f"Error parsing JSON file {json_file}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Error loading conversation from {json_file}: {e}")
                continue

        self.logger.info(f"Loaded {len(conversations)} conversations from seed folder")
        return conversations

    async def _ensure_logged_in(self):
        """Ensure user is logged into ChatGPT."""
        # Check if already logged in
        await self.page.wait_for_load_state('networkidle')

        # Look for login indicators
        login_selectors = [
            'button[data-testid="login-button"]',
            'input[type="email"]',
            '[data-testid="email-input"]'
        ]

        needs_login = False
        for selector in login_selectors:
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                needs_login = True
                break
            except:
                continue

        if needs_login:
            await self._perform_login()

    async def _perform_login(self):
        """Perform login to ChatGPT."""
        self.logger.info("Performing login to ChatGPT")

        # Navigate to login page
        await self.page.goto('https://chatgpt.com/auth/login')

        # Fill email
        email_input = await self.page.wait_for_selector('input[type="email"]')
        await email_input.fill(self.config['CHATGPT_EMAIL'])

        # Click continue
        continue_button = await self.page.wait_for_selector('button[type="submit"]')
        await continue_button.click()

        # Fill password
        password_input = await self.page.wait_for_selector('input[type="password"]')
        await password_input.fill(self.config['CHATGPT_PASSWORD'])

        # Click continue
        continue_button = await self.page.wait_for_selector('button[type="submit"]')
        await continue_button.click()

        # Wait for 2FA or success
        self.logger.info("Login submitted. Check browser for 2FA if required.")
        self.logger.info("Press Enter in terminal when login is complete...")

        # Pause for manual 2FA
        input("Press Enter after completing 2FA in browser...")

        # Verify login success
        await self.page.wait_for_selector('.text-token-text-primary', timeout=30000)
        self.logger.info("Login successful")

    async def _process_conversations(self, conversations: List[Dict[str, Any]],
                                   dry_run: bool = False) -> List[Dict[str, Any]]:
        """Process and export conversations."""
        results = []
        total = len(conversations)

        for i, conv in enumerate(conversations):
            global_state['progress'] = {
                'current': i + 1,
                'total': total,
                'message': f'Processing: {conv.get("title", "Untitled")}'
            }

            if conv['id'] in self.processed_ids:
                self.logger.info(f"Skipping already processed: {conv['id']}")
                continue

            # Fetch full conversation data
            self.logger.info(f"Fetching full data for conversation: {conv['id']}")
            full_conv = await self._fetch_conversation_data(conv['id'])

            if not full_conv:
                self.logger.warning(f"Could not fetch data for conversation: {conv['id']}")
                continue

            # Merge metadata
            full_conv.update(conv)

            # Classify conversation
            classification = self._classify_conversation(full_conv)

            if classification == 'skip':
                self.logger.info(f"Skipping excluded conversation: {full_conv.get('title', 'Untitled')}")
                continue

            # Save raw data
            if not dry_run:
                saved_files = self._save_conversation(full_conv, classification)
                full_conv['_saved_files'] = saved_files
                full_conv['_classification'] = classification
                self._save_to_index(full_conv)

            results.append({
                'id': full_conv['id'],
                'title': full_conv.get('title', 'Untitled'),
                'classification': classification,
                'saved_files': full_conv.get('_saved_files', []) if not dry_run else [],
                'message_count': len(full_conv.get('mapping', {}))
            })

            # Rate limiting between conversations
            await asyncio.sleep(0.5)

        return results

    def _classify_conversation(self, conversation: Dict[str, Any]) -> str:
        """Classify conversation content."""
        # Extract text content
        text_content = self._extract_conversation_text(conversation).lower()

        # Check exclusion rules first
        exclude_keywords = self.classification_config.get('exclude_keywords', [])
        if any(keyword in text_content for keyword in exclude_keywords):
            return 'skip'

        # Check science disciplines
        disciplines = self.classification_config.get('science_disciplines', [])
        for discipline_config in disciplines:
            for discipline, keywords in discipline_config.items():
                if any(keyword in text_content for keyword in keywords):
                    return discipline

        # Check creative/spiritual content
        creative_keywords = self.classification_config.get('creative_spiritual_keywords', [])
        if any(keyword in text_content for keyword in creative_keywords):
            return 'philosophy_theory'

        # Default to personal if enabled
        if self.config['INCLUDE_PERSONAL_SANITIZED']:
            return 'personal_sanitized'

        return 'skip'

    def _extract_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Extract text content from conversation for classification."""
        texts = []

        mapping = conversation.get('mapping', {})
        for node_id, node in mapping.items():
            message = node.get('message', {})
            if message.get('author', {}).get('role') in ['user', 'assistant']:
                content = message.get('content', {})
                if isinstance(content, dict):
                    # Handle different content formats
                    parts = content.get('parts', [])
                    for part in parts:
                        if isinstance(part, str):
                            texts.append(part)
                elif isinstance(content, str):
                    texts.append(content)

        return ' '.join(texts)

    def _save_conversation(self, conversation: Dict[str, Any],
                          classification: str) -> List[str]:
        """Save conversation in various formats."""
        saved_files = []

        # Generate filename components
        created_at = datetime.fromtimestamp(conversation.get('create_time', 0))
        title_slug = self._slugify(conversation.get('title', 'Untitled'))
        conv_id = conversation['id']

        date_str = created_at.strftime('%Y-%m-%d_%H-%M-%S')
        base_filename = f"{date_str}__{title_slug}__{conv_id}"

        # Save raw JSON
        raw_json_path = self.raw_dir / created_at.strftime('%Y') / created_at.strftime('%m') / f"{base_filename}.json"
        raw_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_json_path, 'w') as f:
            json.dump(conversation, f, indent=2)
        saved_files.append(str(raw_json_path))

        # Save Markdown
        md_content = self._conversation_to_markdown(conversation)
        raw_md_path = self.raw_md_dir / created_at.strftime('%Y') / created_at.strftime('%m') / f"{base_filename}.md"
        raw_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_md_path, 'w') as f:
            f.write(md_content)
        saved_files.append(str(raw_md_path))

        # Save to classified folder
        if classification.startswith(('math', 'physics', 'ml', 'cryptography',
                                   'systems', 'philosophy_theory', 'application')):
            class_dir = self.science_dir / classification
        elif classification == 'personal_sanitized':
            class_dir = self.personal_dir
        else:
            return saved_files

        class_json_path = class_dir / created_at.strftime('%Y') / created_at.strftime('%m') / f"{base_filename}.json"
        class_md_path = class_dir / created_at.strftime('%Y') / created_at.strftime('%m') / f"{base_filename}.md"

        for path in [class_json_path, class_md_path]:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Copy files
        import shutil
        shutil.copy2(raw_json_path, class_json_path)
        shutil.copy2(raw_md_path, class_md_path)

        saved_files.extend([str(class_json_path), str(class_md_path)])

        return saved_files

    def _conversation_to_markdown(self, conversation: Dict[str, Any]) -> str:
        """Convert conversation to Markdown format."""
        lines = []
        lines.append(f"# {conversation.get('title', 'Untitled Conversation')}")
        lines.append("")

        mapping = conversation.get('mapping', {})
        current_node = conversation.get('current_node')

        # Sort nodes by creation time if available
        nodes = []
        for node_id, node in mapping.items():
            message = node.get('message', {})
            timestamp = message.get('create_time', 0)
            nodes.append((timestamp, node_id, node))

        nodes.sort(key=lambda x: x[0])

        for timestamp, node_id, node in nodes:
            message = node.get('message', {})
            author = message.get('author', {})
            role = author.get('role', 'unknown')

            if role not in ['user', 'assistant']:
                continue

            role_display = "User" if role == "user" else "Assistant"
            lines.append(f"## {role_display}")
            lines.append("")

            content = message.get('content', {})
            if isinstance(content, dict):
                parts = content.get('parts', [])
                for part in parts:
                    if isinstance(part, str):
                        lines.append(part)
            elif isinstance(content, str):
                lines.append(content)

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-').lower()[:50]

    def _update_aiva_memory(self, results: List[Dict[str, Any]]):
        """Update AIVA memory with exported conversations."""
        self.logger.info("Updating AIVA memory with conversation data")

        # Create backups of existing memory files
        self._backup_memory_files()

        try:
            # Update episodic memory
            self._update_episodic_memory(results)

            # Update timeline
            self._update_timeline(results)

            # Update artifacts
            self._update_artifacts(results)

            self.logger.info("AIVA memory updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update AIVA memory: {e}")
            # Restore backups on failure
            self._restore_memory_backups()
            raise

    def _backup_memory_files(self):
        """Create backups of memory files before modification."""
        import shutil

        backup_dir = self.run_data_dir / 'memory_backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)

        memory_files = [
            'episodic.json',
            'timeline.json',
            'artifacts.json'
        ]

        for filename in memory_files:
            src = self.aiva_memory_dir / filename
            if src.exists():
                dst = backup_dir / filename
                shutil.copy2(src, dst)
                self.logger.debug(f"Backed up {filename} to {backup_dir}")

    def _restore_memory_backups(self):
        """Restore memory files from most recent backup."""
        backup_dir = self.run_data_dir / 'memory_backups'
        if not backup_dir.exists():
            return

        # Find most recent backup
        backups = sorted(backup_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not backups:
            return

        latest_backup = backups[0]
        self.logger.info(f"Restoring memory from backup: {latest_backup}")

        memory_files = [
            'episodic.json',
            'timeline.json',
            'artifacts.json'
        ]

        for filename in memory_files:
            backup_file = latest_backup / filename
            if backup_file.exists():
                dst = self.aiva_memory_dir / filename
                import shutil
                shutil.copy2(backup_file, dst)
                self.logger.debug(f"Restored {filename} from backup")

    def _update_episodic_memory(self, results: List[Dict[str, Any]]):
        """Add conversation episodes to episodic memory."""
        episodic_file = self.aiva_memory_dir / 'episodic.json'

        # Load existing episodic memory
        if episodic_file.exists():
            with open(episodic_file, 'r') as f:
                episodic_data = json.load(f)
        else:
            episodic_data = {"episodes": []}

        existing_ids = {ep['id'] for ep in episodic_data['episodes']}

        # Add new episodes for conversations
        for result in results:
            conv_id = result['id']
            if conv_id in existing_ids:
                continue

            # Create episode entry
            episode = {
                'id': f'gpt_conv_{conv_id}',
                'time_utc': datetime.fromtimestamp(result.get('create_time', 0)).isoformat() + 'Z',
                'summary': f"GPT conversation: {result['title']}",
                'participants': ['Brad Wallace', 'GPT Assistant'],
                'artifacts': result.get('saved_files', []),
                'emotional_tone': ['intellectual', 'exploratory'],
                'outcomes': [f"Conversation documented and classified as {result['classification']}"],
                'key_insights': self._extract_key_insights(result),
                'resonance_score': 0.79,  # Default resonance
                'phase_state': 'research_documentation',
                'importance': self._determine_importance(result),
                'conversation_id': conv_id,
                'classification': result['classification'],
                'message_count': result['message_count']
            }

            episodic_data['episodes'].append(episode)

        # Update memory statistics
        episodic_data['memory_statistics'] = episodic_data.get('memory_statistics', {})
        episodic_data['memory_statistics']['total_episodes'] = len(episodic_data['episodes'])
        episodic_data['memory_statistics']['last_updated'] = datetime.now().isoformat()

        # Save updated episodic memory
        with open(episodic_file, 'w') as f:
            json.dump(episodic_data, f, indent=2)

    def _update_timeline(self, results: List[Dict[str, Any]]):
        """Add conversation events to timeline."""
        timeline_file = self.aiva_memory_dir / 'timeline.json'

        # Load existing timeline
        if timeline_file.exists():
            with open(timeline_file, 'r') as f:
                timeline_data = json.load(f)
        else:
            timeline_data = {"events": []}

        existing_ids = {event.get('conversation_id') for event in timeline_data['events']}

        # Add new events for conversations
        for result in results:
            conv_id = result['id']
            if conv_id in existing_ids:
                continue

            event = {
                'date': datetime.fromtimestamp(result.get('create_time', 0)).isoformat() + 'Z',
                'event': f"GPT conversation: {result['title']}",
                'type': 'research_discussion',
                'significance': f"Scientific discussion classified as {result['classification']}",
                'participants': ['Brad Wallace', 'GPT Assistant'],
                'outcome': f"Conversation documented with {result['message_count']} messages",
                'conversation_id': conv_id,
                'classification': result['classification'],
                'files_saved': len(result.get('saved_files', []))
            }

            timeline_data['events'].append(event)

        # Sort events by date
        timeline_data['events'].sort(key=lambda x: x['date'])

        # Save updated timeline
        with open(timeline_file, 'w') as f:
            json.dump(timeline_data, f, indent=2)

    def _update_artifacts(self, results: List[Dict[str, Any]]):
        """Add conversation files to artifacts registry."""
        artifacts_file = self.aiva_memory_dir / 'artifacts.json'

        # Load existing artifacts
        if artifacts_file.exists():
            with open(artifacts_file, 'r') as f:
                artifacts_data = json.load(f)
        else:
            artifacts_data = {"files": []}

        existing_files = {art['name'] for art in artifacts_data['files']}

        # Add new artifacts for saved files
        for result in results:
            for file_path in result.get('saved_files', []):
                file_name = Path(file_path).name
                if file_name in existing_files:
                    continue

                # Determine file type and description
                file_type, description = self._classify_file_artifact(file_path, result)

                artifact = {
                    'name': file_name,
                    'description': description,
                    'type': file_type,
                    'importance': 'medium',
                    'creation_date': datetime.now().strftime('%Y-%m-%d'),
                    'conversation_id': result['id'],
                    'classification': result['classification'],
                    'full_path': file_path,
                    'file_size': self._get_file_size(file_path),
                    'checksum': self._calculate_checksum(file_path)
                }

                artifacts_data['files'].append(artifact)

        # Save updated artifacts
        with open(artifacts_file, 'w') as f:
            json.dump(artifacts_data, f, indent=2)

    def _extract_key_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract key insights from conversation result."""
        insights = []

        classification = result.get('classification', '')
        if classification in ['math', 'physics', 'ml', 'cryptography']:
            insights.append(f"Technical discussion in {classification}")
        elif classification in ['philosophy_theory']:
            insights.append("Theoretical/philosophical exploration")
        elif classification == 'personal_sanitized':
            insights.append("Personal reflection documented")

        message_count = result.get('message_count', 0)
        insights.append(f"Conversation depth: {message_count} messages")

        return insights

    def _determine_importance(self, result: Dict[str, Any]) -> str:
        """Determine importance level of conversation."""
        classification = result.get('classification', '')
        message_count = result.get('message_count', 0)

        if classification in ['math', 'physics', 'ml', 'cryptography', 'philosophy_theory']:
            if message_count > 20:
                return 'high'
            else:
                return 'medium'
        else:
            return 'low'

    def _classify_file_artifact(self, file_path: str, result: Dict[str, Any]) -> Tuple[str, str]:
        """Classify a saved file for artifact registry."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        classification = result.get('classification', '')

        if suffix == '.json':
            if 'raw' in file_path:
                return 'conversation_data', f"Raw GPT conversation data - {classification}"
            else:
                return 'processed_conversation', f"Processed GPT conversation - {classification}"
        elif suffix == '.md':
            return 'conversation_transcript', f"Markdown transcript - {classification}"
        else:
            return 'conversation_artifact', f"GPT conversation file - {classification}"

    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return Path(file_path).stat().st_size
        except:
            return 0

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return 'checksum_failed'

    async def _cleanup(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()


def create_extension_api(exporter: GPTTeamsExporter, config_dict: Dict[str, Any]) -> Flask:
    """Create Flask API for browser extension."""
    from flask import Flask, request, jsonify
    import asyncio
    import threading

    app = Flask(__name__)

    @app.route('/status')
    def status():
        return jsonify({
            'status': 'ready',
            'backend_connected': True,
            'aiva_available': False
        })

    @app.route('/start', methods=['POST'])
    def start_export():
        try:
            data = request.get_json() or {}
            return jsonify({'status': 'not_implemented_yet'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/progress')
    def get_progress():
        return jsonify({'running': False, 'progress': 0})

    @app.route('/capture', methods=['POST'])
    def capture_conversation():
        try:
            data = request.get_json() or {}
            return jsonify({
                'status': 'received',
                'data_length': len(str(data)),
                'data_keys': list(data.keys()) if isinstance(data, dict) else 'not_dict'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/paste', methods=['POST'])
    def paste_conversation():
        try:
            data = request.get_json() or {}

            # Create a conversation-like object from pasted data
            conversation = {
                'id': f"pasted_{int(time_module.time())}_{hash(data.get('content', ''))[:8]}",
                'title': data.get('title', 'Pasted Conversation'),
                'create_time': time_module.time(),
                'update_time': time_module.time(),
                'pasted_at': data.get('pasted_at'),
                'content': data.get('content', ''),
                'messages': data.get('messages', []),
                'message_count': data.get('message_count', 1),
                'word_count': data.get('word_count', 0),
                'source': 'pasted',
                'mapping': {}
            }

            # Convert messages to mapping format if provided
            if data.get('messages'):
                for i, msg in enumerate(data['messages']):
                    conversation['mapping'][f'pasted_{i}'] = {
                        'message': {
                            'content': {'parts': [msg.get('content', '')]},
                            'author': {'role': msg.get('role', 'user')},
                            'create_time': time_module.time()
                        }
                    }

            # Process the conversation
            classification = exporter._classify_conversation(conversation)

            # Save the conversation
            saved_files = exporter._save_conversation(conversation, classification)
            exporter._save_to_index(conversation)

            # Update AIVA memory
            results = [{
                'id': conversation['id'],
                'title': conversation['title'],
                'classification': classification,
                'saved_files': saved_files,
                'message_count': conversation['message_count']
            }]
            exporter._update_aiva_memory(results)

            return jsonify({
                'status': 'pasted',
                'id': conversation['id'],
                'title': conversation['title'],
                'classification': classification,
                'message_count': conversation['message_count'],
                'word_count': conversation['word_count'],
                'pasted_at': conversation['pasted_at'],
                'file_path': saved_files[0] if saved_files else None,
                'source': 'pasted'
            })

        except Exception as e:
            print(f"Paste error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/stop', methods=['POST'])
    def stop_export():
        return jsonify({'status': 'stopped'})

    return app

def create_web_ui(exporter: GPTTeamsExporter) -> Flask:
    """Create Flask web UI for the exporter."""
    app = Flask(__name__)

    @app.route('/')
    def index():
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPT Teams Exporter</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .running { background: #e8f5e8; border: 1px solid #4caf50; }
                .stopped { background: #ffebee; border: 1px solid #f44336; }
                button { padding: 10px 20px; margin: 5px; cursor: pointer; }
                .progress { margin: 10px 0; }
                .error { color: #f44336; }
            </style>
        </head>
        <body>
            <h1>GPT Teams Archive Exporter</h1>
            <div class="status {{ 'running' if global_state['running'] else 'stopped' }}">
                Status: {{ 'Running' if global_state['running'] else 'Stopped' }}
            </div>

            <div class="progress">
                Progress: {{ global_state['progress']['current'] }}/{{ global_state['progress']['total'] }} - {{ global_state['progress']['message'] }}
            </div>

            <form action="/start" method="post">
                <button type="submit" {{ 'disabled' if global_state['running'] else '' }}>Start Export</button>
            </form>

            <form action="/dry-run" method="post">
                <button type="submit" {{ 'disabled' if global_state['running'] else '' }}>Dry Run</button>
            </form>

            {% if global_state['errors'] %}
            <div class="error">
                <h3>Errors:</h3>
                <ul>
                {% for error in global_state['errors'] %}
                    <li>{{ error }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}

            {% if global_state['last_run'] %}
            <div>
                <h3>Last Run Results:</h3>
                <pre>{{ global_state['last_run'] }}</pre>
            </div>
            {% endif %}
        </body>
        </html>
        """
        return render_template_string(html, global_state=global_state)

    @app.route('/start', methods=['POST'])
    def start_export():
        if global_state['running']:
            return jsonify({'error': 'Export already running'}), 400

        global_state['running'] = True
        global_state['errors'] = []

        # Run export in background
        def run_export():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(exporter.run())
                global_state['last_run'] = json.dumps(result, indent=2)
            except Exception as e:
                global_state['errors'].append(str(e))
            finally:
                global_state['running'] = False

        import threading
        thread = threading.Thread(target=run_export)
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'started'})

    @app.route('/dry-run', methods=['POST'])
    def dry_run():
        if global_state['running']:
            return jsonify({'error': 'Export already running'}), 400

        global_state['running'] = True
        global_state['errors'] = []

        def run_dry_run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(exporter.run(dry_run=True))
                global_state['last_run'] = json.dumps(result, indent=2)
            except Exception as e:
                global_state['errors'].append(str(e))
            finally:
                global_state['running'] = False

        import threading
        thread = threading.Thread(target=run_dry_run)
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'started'})

    return app


def main():
    parser = argparse.ArgumentParser(description='GPT Teams Archive Exporter')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be exported')
    parser.add_argument('--since', help='Only export after date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Maximum conversations to export')
    parser.add_argument('--headful', action='store_true', help='Run browser in headed mode')
    parser.add_argument('--only-team', help='Only export from specific team')
    parser.add_argument('--include-personal', action='store_true', help='Include personal conversations')
    parser.add_argument('--max-concurrency', type=int, default=3, help='Max concurrent requests')
    parser.add_argument('--retry', type=int, default=3, help='Number of retries')
    parser.add_argument('--seed-folder', help='Import from local folder')
    parser.add_argument('--web-ui', action='store_true', help='Start web UI')
    parser.add_argument('--extension-api', action='store_true', help='Start extension API server')

    args = parser.parse_args()

    # Override config with args
    config = DEFAULT_CONFIG.copy()
    if args.since:
        config['EXPORT_SINCE'] = args.since
    if args.limit:
        config['EXPORT_LIMIT'] = args.limit
    if args.headful:
        config['SCRAPE_HEADFUL'] = True
    if args.include_personal:
        config['INCLUDE_PERSONAL_SANITIZED'] = True
    if args.seed_folder:
        config['SEED_FOLDER'] = args.seed_folder

    # Validate required config (skip for dry-run or seed-folder mode)
    needs_credentials = not (args.dry_run or args.seed_folder)
    if needs_credentials and (not config['CHATGPT_EMAIL'] or not config['CHATGPT_PASSWORD']):
        print("ERROR: CHATGPT_EMAIL and CHATGPT_PASSWORD must be set in .env file")
        print("For dry-run or seed-folder mode, you can run without credentials.")
        sys.exit(1)

    exporter = GPTTeamsExporter(config)

    if args.extension_api:
        app = create_extension_api(exporter, {})  # Pass empty dict to avoid any config reference
        print("Starting extension API at http://localhost:8765")
        print("Load the Brave extension to connect")

        # Use Werkzeug directly to avoid Flask's reloader issues
        from werkzeug.serving import make_server
        server = make_server('127.0.0.1', 8765, app, threaded=True)
        print("Server starting...")
        server.serve_forever()
    elif args.web_ui:
        app = create_web_ui(exporter)
        print("Starting web UI at http://localhost:8765")
        app.run(host='0.0.0.0', port=8765, debug=True)
    else:
        # Run synchronously
        try:
            result = asyncio.run(exporter.run(
                dry_run=args.dry_run,
                limit=args.limit,
                since=args.since
            ))
            print(json.dumps(result, indent=2))
        except KeyboardInterrupt:
            print("\nExport interrupted")
        except Exception as e:
            print(f"Export failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
