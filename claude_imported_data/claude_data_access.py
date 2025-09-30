#!/usr/bin/env python3
"""
Claude Data Access Utility
Provides easy access to all imported Claude data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ClaudeDataAccess:
    """Access utility for imported Claude data"""

    def __init__(self, data_path: str = "./claude_imported_data"):
        self.data_path = Path(data_path)
        self._metadata = None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Load import metadata"""
        if self._metadata is None:
            metadata_file = self.data_path / "metadata" / "import_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
        return self._metadata or {}

    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        users_file = self.data_path / "users" / "users.json"
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def get_projects_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get projects by category"""
        category_file = self.data_path / "projects" / category / "projects.json"
        if category_file.exists():
            with open(category_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects across all categories"""
        all_projects = []
        projects_dir = self.data_path / "projects"

        if projects_dir.exists():
            for category_dir in projects_dir.iterdir():
                if category_dir.is_dir():
                    category_file = category_dir / "projects.json"
                    if category_file.exists():
                        with open(category_file, 'r', encoding='utf-8') as f:
                            all_projects.extend(json.load(f))

        return all_projects

    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations index"""
        conv_file = self.data_path / "conversations" / "conversations_index.json"
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def get_conversation(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get specific conversation by UUID"""
        conv_file = self.data_path / "conversations" / f"conversation_{uuid}.json"
        if conv_file.exists():
            with open(conv_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def search_projects(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search projects by filename or content"""
        query_lower = query.lower()
        results = []

        projects = self.get_projects_by_category(category) if category else self.get_all_projects()

        for project in projects:
            if (query_lower in project['filename'].lower() or
                query_lower in project['content'].lower()):
                results.append(project)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get import statistics"""
        return self.metadata.get('statistics', {})

    def list_categories(self) -> List[str]:
        """List all project categories"""
        return self.metadata.get('categories', [])

# Convenience functions
def get_data_access(data_path: str = "./claude_imported_data") -> ClaudeDataAccess:
    """Get data access instance"""
    return ClaudeDataAccess(data_path)

def search_content(query: str) -> List[Dict[str, Any]]:
    """Quick search across all projects"""
    access = get_data_access()
    return access.search_projects(query)

def get_all_research_projects() -> List[Dict[str, Any]]:
    """Get all research-related projects"""
    access = get_data_access()
    return access.get_projects_by_category('research')

def get_ai_projects() -> List[Dict[str, Any]]:
    """Get all AI/ML projects"""
    access = get_data_access()
    return access.get_projects_by_category('ai_ml')

if __name__ == "__main__":
    # Example usage
    access = get_data_access()

    print("Claude Data Import Summary:")
    print(f"- Users: {len(access.get_users())}")
    print(f"- Projects: {len(access.get_all_projects())}")
    print(f"- Conversations: {len(access.get_conversations())}")
    print(f"- Categories: {', '.join(access.list_categories())}")

    # Show sample research projects
    research = access.get_projects_by_category('research')
    if research:
        print(f"\nSample Research Project: {research[0]['filename']}")
