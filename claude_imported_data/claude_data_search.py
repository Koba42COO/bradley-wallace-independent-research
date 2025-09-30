#!/usr/bin/env python3
"""
Claude Data Search Utility
Advanced search capabilities for imported Claude data
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class ClaudeDataSearch:
    """Advanced search utility for Claude data"""

    def __init__(self, data_path: str = "./claude_imported_data"):
        self.data_path = Path(data_path)
        self._load_data()

    def _load_data(self):
        """Load data for searching"""
        self.projects = []
        self.conversations = []

        # Load all projects
        projects_dir = self.data_path / "projects"
        if projects_dir.exists():
            for category_dir in projects_dir.iterdir():
                if category_dir.is_dir():
                    projects_file = category_dir / "projects.json"
                    if projects_file.exists():
                        with open(projects_file, 'r', encoding='utf-8') as f:
                            self.projects.extend(json.load(f))

        # Load conversations index
        conv_file = self.data_path / "conversations" / "conversations_index.json"
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)

    def search_projects(self, query: str, case_sensitive: bool = False,
                       regex: bool = False, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Advanced project search"""
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        # Filter by category if specified
        search_projects = [p for p in self.projects if category is None or p.get('category') == category]

        for project in search_projects:
            filename = project['filename']
            content = project['content']

            if regex:
                try:
                    if re.search(query, filename, flags) or re.search(query, content, flags):
                        results.append(project)
                except re.error:
                    continue  # Skip invalid regex
            else:
                search_query = query if case_sensitive else query.lower()
                search_filename = filename if case_sensitive else filename.lower()
                search_content = content if case_sensitive else content.lower()

                if search_query in search_filename or search_query in search_content:
                    results.append(project)

        return results

    def search_conversations(self, query: str, case_sensitive: bool = False,
                           regex: bool = False, sender: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search conversations"""
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for conv in self.conversations:
            match_found = False

            # Check if we should filter by sender
            if sender:
                messages = [msg for msg in conv['messages'] if msg['sender'] == sender]
            else:
                messages = conv['messages']

            for msg in messages:
                text = msg['text']
                content_text = ' '.join([c.get('text', '') for c in msg.get('content', []) if isinstance(c, dict)])

                if regex:
                    try:
                        if (re.search(query, text, flags) or
                            re.search(query, content_text, flags)):
                            match_found = True
                            break
                    except re.error:
                        continue
                else:
                    search_query = query if case_sensitive else query.lower()
                    search_text = text if case_sensitive else text.lower()
                    search_content = content_text if case_sensitive else content_text.lower()

                    if search_query in search_text or search_query in search_content:
                        match_found = True
                        break

            if match_found:
                results.append(conv)

        return results

    def find_similar_projects(self, content_sample: str, limit: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Find projects similar to provided content"""
        # Simple similarity based on word overlap
        sample_words = set(content_sample.lower().split())
        similarities = []

        for project in self.projects:
            project_words = set(project['content'].lower().split())
            intersection = len(sample_words.intersection(project_words))
            union = len(sample_words.union(project_words))

            if union > 0:
                similarity = intersection / union
                similarities.append((project, similarity))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def get_projects_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get projects created within date range"""
        results = []

        for project in self.projects:
            created_at = datetime.fromisoformat(project['created_at'])
            if start_date <= created_at <= end_date:
                results.append(project)

        return results

    def get_conversations_by_participants(self, sender_filter: str) -> List[Dict[str, Any]]:
        """Get conversations involving specific sender"""
        results = []

        for conv in self.conversations:
            if any(msg['sender'] == sender_filter for msg in conv['messages']):
                results.append(conv)

        return results

    def export_search_results(self, results: List[Dict[str, Any]], filename: str):
        """Export search results to JSON file"""
        export_path = self.data_path / "exports" / filename
        export_path.parent.mkdir(exist_ok=True)

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results exported to {{export_path}}")

def main():
    """Command line search interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Search Claude imported data")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--type", choices=["projects", "conversations", "all"],
                       default="all", help="Type of data to search")
    parser.add_argument("--category", help="Project category to search in")
    parser.add_argument("--regex", action="store_true", help="Use regex search")
    parser.add_argument("--case-sensitive", action="store_true", help="Case sensitive search")
    parser.add_argument("--export", help="Export results to file")

    args = parser.parse_args()

    search = ClaudeDataSearch()

    if args.type in ["projects", "all"]:
        print(f"Searching projects for: {{args.query}}")
        project_results = search.search_projects(
            args.query, args.case_sensitive, args.regex, args.category
        )
        print(f"Found {{len(project_results)}} projects")

        for i, project in enumerate(project_results[:5]):  # Show first 5
            print(f"  {{i+1}}. {{project['filename']}} ({{project.get('category', 'uncategorized')}})")

    if args.type in ["conversations", "all"]:
        print(f"\nSearching conversations for: {{args.query}}")
        conv_results = search.search_conversations(
            args.query, args.case_sensitive, args.regex
        )
        print(f"Found {{len(conv_results)}} conversations")

        for i, conv in enumerate(conv_results[:3]):  # Show first 3
            msg_count = conv.get('message_count', 0)
            print(f"  {{i+1}}. Conversation {{conv['uuid'][:8]}}... ({{msg_count}} messages)")

    if args.export and (args.type == "projects" or args.type == "all"):
        search.export_search_results(project_results, args.export)

if __name__ == "__main__":
    main()
