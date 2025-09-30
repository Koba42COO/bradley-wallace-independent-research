#!/usr/bin/env python3
"""
Complete Claude Data Import System
Imports all data from Claude data dump -8-29-25 into project structure

Features:
- Imports users, projects, and conversations
- Maintains all metadata and relationships
- Creates searchable data structures
- Provides access utilities
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data structure"""
    uuid: str
    full_name: str
    email_address: str
    verified_phone_number: Optional[str] = None

@dataclass
class Project:
    """Project data structure"""
    uuid: str
    filename: str
    content: str
    created_at: datetime
    content_hash: str
    language: Optional[str] = None
    category: Optional[str] = None

@dataclass
class ConversationMessage:
    """Individual conversation message"""
    uuid: str
    text: str
    sender: str  # "human" or "assistant"
    created_at: datetime
    updated_at: datetime
    content: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    files: List[Dict[str, Any]]

@dataclass
class Conversation:
    """Complete conversation structure"""
    uuid: str
    messages: List[ConversationMessage]
    created_at: datetime
    updated_at: datetime
    message_count: int

class ClaudeDataImporter:
    """Complete importer for Claude data dump"""

    def __init__(self, dump_path: str, output_path: str = "./claude_imported_data"):
        self.dump_path = Path(dump_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # Data storage
        self.users: List[User] = []
        self.projects: List[Project] = []
        self.conversations: List[Conversation] = []

        # Statistics
        self.stats = {
            'users_imported': 0,
            'projects_imported': 0,
            'conversations_imported': 0,
            'messages_imported': 0,
            'errors': []
        }

    def import_all_data(self) -> Dict[str, Any]:
        """Import all data from the dump"""
        logger.info("Starting complete Claude data import...")

        try:
            # Import each data type
            self._import_users()
            self._import_projects()
            self._import_conversations()

            # Save processed data
            self._save_imported_data()

            # Create access utilities
            self._create_access_utilities()

            # Generate summary
            summary = self._generate_import_summary()

            logger.info("Claude data import completed successfully!")
            return summary

        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return self._generate_import_summary()

    def _import_users(self):
        """Import users data"""
        logger.info("Importing users data...")
        users_file = self.dump_path / "users.json"

        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)

            for user_data in users_data:
                user = User(
                    uuid=user_data['uuid'],
                    full_name=user_data['full_name'],
                    email_address=user_data['email_address'],
                    verified_phone_number=user_data.get('verified_phone_number')
                )
                self.users.append(user)

            self.stats['users_imported'] = len(self.users)
            logger.info(f"Imported {len(self.users)} users")

        except Exception as e:
            error_msg = f"Failed to import users: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

    def _import_projects(self):
        """Import projects data"""
        logger.info("Importing projects data...")
        projects_file = self.dump_path / "projects.json"

        try:
            with open(projects_file, 'r', encoding='utf-8') as f:
                projects_data = json.load(f)

            for project_data in projects_data:
                # Each project can have multiple documents in the 'docs' array
                docs = project_data.get('docs', [])
                for doc in docs:
                    # Parse created_at timestamp
                    created_at = datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00'))

                    # Generate content hash for integrity checking
                    content_hash = hashlib.sha256(doc['content'].encode('utf-8')).hexdigest()

                    # Detect language/category from filename and content
                    language = self._detect_language(doc['filename'], doc['content'])
                    category = self._categorize_project(doc['filename'], doc['content'])

                    # Create a unique UUID combining project and doc UUIDs
                    combined_uuid = f"{project_data['uuid']}_{doc['uuid']}"

                    project = Project(
                        uuid=combined_uuid,
                        filename=doc['filename'],
                        content=doc['content'],
                        created_at=created_at,
                        content_hash=content_hash,
                        language=language,
                        category=category
                    )
                    self.projects.append(project)

            self.stats['projects_imported'] = len(self.projects)
            logger.info(f"Imported {len(self.projects)} projects")

        except Exception as e:
            error_msg = f"Failed to import projects: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

    def _import_conversations(self):
        """Import conversations data"""
        logger.info("Importing conversations data...")
        conversations_file = self.dump_path / "conversations.json"

        try:
            with open(conversations_file, 'r', encoding='utf-8') as f:
                conversations_data = json.load(f)

            for conv_data in conversations_data:
                messages = []
                for msg_data in conv_data.get('messages', []):
                    # Parse timestamps
                    created_at = datetime.fromisoformat(msg_data['created_at'].replace('Z', '+00:00'))
                    updated_at = datetime.fromisoformat(msg_data['updated_at'].replace('Z', '+00:00'))

                    message = ConversationMessage(
                        uuid=msg_data['uuid'],
                        text=msg_data['text'],
                        sender=msg_data['sender'],
                        created_at=created_at,
                        updated_at=updated_at,
                        content=msg_data.get('content', []),
                        attachments=msg_data.get('attachments', []),
                        files=msg_data.get('files', [])
                    )
                    messages.append(message)

                # Conversation metadata
                created_at = datetime.fromisoformat(conv_data['created_at'].replace('Z', '+00:00'))
                updated_at = datetime.fromisoformat(conv_data['updated_at'].replace('Z', '+00:00'))

                conversation = Conversation(
                    uuid=conv_data['uuid'],
                    messages=messages,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=len(messages)
                )
                self.conversations.append(conversation)
                self.stats['messages_imported'] += len(messages)

            self.stats['conversations_imported'] = len(self.conversations)
            logger.info(f"Imported {len(self.conversations)} conversations with {self.stats['messages_imported']} messages")

        except Exception as e:
            error_msg = f"Failed to import conversations: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

    def _detect_language(self, filename: str, content: str) -> Optional[str]:
        """Detect programming language from filename and content"""
        filename_lower = filename.lower()

        # File extension detection
        if filename_lower.endswith('.py'):
            return 'python'
        elif filename_lower.endswith('.js'):
            return 'javascript'
        elif filename_lower.endswith('.ts'):
            return 'typescript'
        elif filename_lower.endswith('.java'):
            return 'java'
        elif filename_lower.endswith('.cpp') or filename_lower.endswith('.cc') or filename_lower.endswith('.cxx'):
            return 'cpp'
        elif filename_lower.endswith('.c'):
            return 'c'
        elif filename_lower.endswith('.go'):
            return 'go'
        elif filename_lower.endswith('.rs'):
            return 'rust'
        elif filename_lower.endswith('.rb'):
            return 'ruby'
        elif filename_lower.endswith('.php'):
            return 'php'
        elif filename_lower.endswith('.md'):
            return 'markdown'
        elif filename_lower.endswith('.txt'):
            return 'text'
        elif filename_lower.endswith('.json'):
            return 'json'
        elif filename_lower.endswith('.yaml') or filename_lower.endswith('.yml'):
            return 'yaml'
        elif filename_lower.endswith('.xml'):
            return 'xml'
        elif filename_lower.endswith('.html'):
            return 'html'
        elif filename_lower.endswith('.css'):
            return 'css'
        elif filename_lower.endswith('.sql'):
            return 'sql'
        elif filename_lower.endswith('.sh'):
            return 'shell'
        elif filename_lower.endswith('.tex'):
            return 'latex'

        # Content-based detection for files without extensions
        content_sample = content[:1000].lower()

        if 'python' in content_sample or 'import ' in content_sample or 'def ' in content_sample:
            return 'python'
        elif 'function' in content_sample or 'const ' in content_sample or 'let ' in content_sample:
            return 'javascript'
        elif 'class ' in content_sample and ('public' in content_sample or 'private' in content_sample):
            return 'java'
        elif '#include' in content_sample or 'int main' in content_sample:
            return 'c'
        elif 'fn ' in content_sample or 'let mut' in content_sample:
            return 'rust'

        return None

    def _categorize_project(self, filename: str, content: str) -> Optional[str]:
        """Categorize project type based on filename and content"""
        filename_lower = filename.lower()
        content_sample = content[:2000].lower()

        # Research and academic
        if any(term in filename_lower or term in content_sample for term in
               ['research', 'paper', 'thesis', 'mathematical', 'physics', 'quantum', 'consciousness', 'wallace']):
            return 'research'

        # AI and machine learning
        if any(term in filename_lower or term in content_sample for term in
               ['ai', 'ml', 'machine learning', 'neural', 'deep learning', 'llm', 'gpt', 'claude', 'model']):
            return 'ai_ml'

        # Software development
        if any(term in filename_lower or term in content_sample for term in
               ['api', 'server', 'client', 'framework', 'library', 'sdk', 'tool']):
            return 'software'

        # Data science and analytics
        if any(term in filename_lower or term in content_sample for term in
               ['data', 'analytics', 'visualization', 'plot', 'chart', 'benchmark']):
            return 'data_science'

        # Web development
        if any(term in filename_lower or term in content_sample for term in
               ['web', 'html', 'css', 'javascript', 'frontend', 'backend', 'http']):
            return 'web'

        # System and infrastructure
        if any(term in filename_lower or term in content_sample for term in
               ['system', 'infrastructure', 'deployment', 'docker', 'kubernetes', 'cloud']):
            return 'infrastructure'

        # Documentation
        if any(term in filename_lower or term in content_sample for term in
               ['readme', 'documentation', 'guide', 'tutorial', 'manual']):
            return 'documentation'

        # Configuration and scripts
        if any(term in filename_lower or term in content_sample for term in
               ['config', 'script', 'automation', 'pipeline', 'ci', 'cd']):
            return 'configuration'

        return 'general'

    def _save_imported_data(self):
        """Save all imported data to disk"""
        logger.info("Saving imported data to disk...")

        # Create subdirectories
        users_dir = self.output_path / "users"
        projects_dir = self.output_path / "projects"
        conversations_dir = self.output_path / "conversations"
        metadata_dir = self.output_path / "metadata"

        for dir_path in [users_dir, projects_dir, conversations_dir, metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # Save users
        users_data = [asdict(user) for user in self.users]
        with open(users_dir / "users.json", 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, default=str)

        # Save projects (organized by category)
        projects_by_category = {}
        for project in self.projects:
            category = project.category or 'uncategorized'
            if category not in projects_by_category:
                projects_by_category[category] = []
            projects_by_category[category].append(asdict(project))

        for category, projects in projects_by_category.items():
            category_dir = projects_dir / category
            category_dir.mkdir(exist_ok=True)

            # Save category index
            with open(category_dir / "projects.json", 'w', encoding='utf-8') as f:
                json.dump(projects, f, indent=2, default=str)

            # Save individual project files
            for project in projects:
                safe_filename = "".join(c for c in project['filename'] if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                project_file = category_dir / f"{safe_filename}_{project['uuid'][:8]}.json"
                with open(project_file, 'w', encoding='utf-8') as f:
                    json.dump(project, f, indent=2, default=str)

        # Save conversations
        conversations_data = []
        for conv in self.conversations:
            conv_dict = asdict(conv)
            conversations_data.append(conv_dict)

            # Save individual conversation
            conv_file = conversations_dir / f"conversation_{conv.uuid}.json"
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conv_dict, f, indent=2, default=str)

        # Save conversations index
        with open(conversations_dir / "conversations_index.json", 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, indent=2, default=str)

        # Save metadata and statistics
        metadata = {
            'import_timestamp': datetime.now().isoformat(),
            'source_path': str(self.dump_path),
            'statistics': self.stats,
            'categories': list(projects_by_category.keys()),
            'users_count': len(self.users),
            'projects_count': len(self.projects),
            'conversations_count': len(self.conversations),
            'total_messages': self.stats['messages_imported']
        }

        with open(metadata_dir / "import_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Data saved successfully")

    def _create_access_utilities(self):
        """Create utility scripts for data access"""
        logger.info("Creating data access utilities...")

        # Create main access utility
        access_utility = f'''#!/usr/bin/env python3
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
        return self._metadata or {{}}

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
        conv_file = self.data_path / "conversations" / f"conversation_{{uuid}}.json"
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
        return self.metadata.get('statistics', {{}})

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
    print(f"- Users: {{len(access.get_users())}}")
    print(f"- Projects: {{len(access.get_all_projects())}}")
    print(f"- Conversations: {{len(access.get_conversations())}}")
    print(f"- Categories: {{', '.join(access.list_categories())}}")

    # Show sample research projects
    research = access.get_projects_by_category('research')
    if research:
        print(f"\\nSample Research Project: {{research[0]['filename']}}")
'''

        # Save access utility
        with open(self.output_path / "claude_data_access.py", 'w', encoding='utf-8') as f:
            f.write(access_utility)

        # Create search utility
        search_utility = '''#!/usr/bin/env python3
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
        print(f"\\nSearching conversations for: {{args.query}}")
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
'''

        # Save search utility
        with open(self.output_path / "claude_data_search.py", 'w', encoding='utf-8') as f:
            f.write(search_utility)

        logger.info("Access utilities created")

    def _generate_import_summary(self) -> Dict[str, Any]:
        """Generate comprehensive import summary"""
        summary = {
            'import_timestamp': datetime.now().isoformat(),
            'source_path': str(self.dump_path),
            'output_path': str(self.output_path),
            'statistics': self.stats,
            'data_overview': {
                'users': len(self.users),
                'projects': len(self.projects),
                'conversations': len(self.conversations),
                'total_messages': self.stats['messages_imported']
            },
            'project_categories': {},
            'language_breakdown': {},
            'errors': self.stats['errors']
        }

        # Project category breakdown
        for project in self.projects:
            category = project.category or 'uncategorized'
            summary['project_categories'][category] = summary['project_categories'].get(category, 0) + 1

            language = project.language
            if language:
                summary['language_breakdown'][language] = summary['language_breakdown'].get(language, 0) + 1

        # Save summary
        with open(self.output_path / "IMPORT_SUMMARY.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

def main():
    """Main import function"""
    if len(sys.argv) != 2:
        print("Usage: python claude_data_import.py <path_to_claude_dump>")
        print("Example: python claude_data_import.py '/path/to/Claude data dump -8-29-25'")
        sys.exit(1)

    dump_path = sys.argv[1]

    if not os.path.exists(dump_path):
        print(f"Error: Path '{dump_path}' does not exist")
        sys.exit(1)

    # Run the import
    importer = ClaudeDataImporter(dump_path)
    summary = importer.import_all_data()

    # Print summary
    print("\n" + "="*60)
    print("CLAUDE DATA IMPORT COMPLETED")
    print("="*60)
    print(f"Source: {summary['source_path']}")
    print(f"Output: {summary['output_path']}")
    print(f"Imported: {summary['statistics']['users_imported']} users")
    print(f"Imported: {summary['statistics']['projects_imported']} projects")
    print(f"Imported: {summary['statistics']['conversations_imported']} conversations")
    print(f"Imported: {summary['statistics']['messages_imported']} messages")

    if summary['statistics']['errors']:
        print(f"Errors: {len(summary['statistics']['errors'])}")
        for error in summary['statistics']['errors']:
            print(f"  - {error}")

    print(f"\nProject Categories: {', '.join(summary['project_categories'].keys())}")
    print(f"Top Languages: {', '.join(sorted(summary['language_breakdown'].items(), key=lambda x: x[1], reverse=True)[:5])}")

    print("\nAccess utilities created:")
    print(f"  - claude_data_access.py (main access utility)")
    print(f"  - claude_data_search.py (advanced search)")
    print(f"  - IMPORT_SUMMARY.json (detailed summary)")

    print("\nTo use the data:")
    print("  python -c \"from claude_data_access import get_data_access; access = get_data_access(); print(access.get_statistics())\"")

if __name__ == "__main__":
    main()
