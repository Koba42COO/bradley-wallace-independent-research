#!/usr/bin/env python3
"""
FINE TOOTH COMB ANALYSIS
Comprehensive examination of every file and directory in the workspace
"""

import os
import json
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

class FineToothAnalyzer:
    def __init__(self, workspace_dir="/Users/coo-koba42/dev"):
        self.workspace_dir = Path(workspace_dir)
        self.analysis = {
            "timestamp": datetime.now().isoformat(),
            "files_by_type": defaultdict(list),
            "files_by_size": defaultdict(list),
            "directories_analysis": {},
            "duplicates": {},
            "outdated_files": [],
            "problematic_files": [],
            "missing_dependencies": [],
            "security_concerns": [],
            "code_quality_issues": [],
            "documentation_gaps": [],
            "build_deployment_issues": [],
            "performance_concerns": [],
            "summary": {}
        }
        self.file_hashes = {}
        self.imports_found = defaultdict(set)
        self.exports_found = defaultdict(set)

    def analyze_file(self, filepath):
        """Analyze individual file comprehensively"""
        try:
            stat = filepath.stat()
            file_info = {
                "path": str(filepath.relative_to(self.workspace_dir)),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "type": self.get_file_type(filepath),
                "hash": self.get_file_hash(filepath),
                "issues": []
            }

            # Analyze file content
            if filepath.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.jl']:
                content_issues = self.analyze_code_file(filepath)
                file_info["issues"].extend(content_issues)

            elif filepath.suffix in ['.md', '.txt', '.rst']:
                doc_issues = self.analyze_documentation_file(filepath)
                file_info["issues"].extend(doc_issues)

            elif filepath.suffix in ['.json', '.yaml', '.yml']:
                config_issues = self.analyze_config_file(filepath)
                file_info["issues"].extend(config_issues)

            # Check for duplicates
            if file_info["hash"] in self.file_hashes:
                self.analysis["duplicates"][file_info["hash"]] = [
                    self.file_hashes[file_info["hash"]],
                    file_info["path"]
                ]
            else:
                self.file_hashes[file_info["hash"]] = file_info["path"]

            # Check file size concerns
            if stat.st_size > 50 * 1024 * 1024:  # 50MB
                file_info["issues"].append("VERY_LARGE_FILE")
                self.analysis["performance_concerns"].append(f"Large file: {file_info['path']} ({stat.st_size/1024/1024:.1f}MB)")

            # Check for outdated files
            if self.is_outdated_file(filepath):
                self.analysis["outdated_files"].append(file_info["path"])

            # Check for security issues
            security_issues = self.check_security_concerns(filepath)
            file_info["issues"].extend(security_issues)

            return file_info

        except Exception as e:
            return {
                "path": str(filepath.relative_to(self.workspace_dir)),
                "error": str(e),
                "issues": ["READ_ERROR"]
            }

    def get_file_type(self, filepath):
        """Determine file type"""
        if filepath.is_dir():
            return "directory"
        mime_type, _ = mimetypes.guess_type(str(filepath))
        if mime_type:
            return mime_type
        return "unknown"

    def get_file_hash(self, filepath, chunk_size=8192):
        """Calculate file hash efficiently"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "unreadable"

    def analyze_code_file(self, filepath):
        """Analyze code file for issues"""
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check for TODO/FIXME comments
            todo_count = len(re.findall(r'(?i)(todo|fixme|hack|xxx)', content))
            if todo_count > 0:
                issues.append(f"TODO_COMMENTS_{todo_count}")

            # Check for print statements in production code
            print_count = len(re.findall(r'\bprint\s*\(', content))
            if print_count > 10 and 'test' not in str(filepath).lower():
                issues.append(f"DEBUG_PRINTS_{print_count}")

            # Check for hardcoded secrets
            secret_patterns = [
                r'password\s*[:=]\s*["\'][^"\']+["\']',
                r'secret\s*[:=]\s*["\'][^"\']+["\']',
                r'key\s*[:=]\s*["\'][^"\']+["\']',
                r'token\s*[:=]\s*["\'][^"\']+["\']'
            ]
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append("HARDCODED_SECRETS")
                    break

            # Check for imports
            if filepath.suffix == '.py':
                import_lines = re.findall(r'^(?:from|import)\s+.+', content, re.MULTILINE)
                for line in import_lines:
                    self.imports_found[str(filepath)].add(line.strip())

            # Check for syntax issues
            if filepath.suffix == '.py':
                try:
                    compile(content, str(filepath), 'exec')
                except SyntaxError as e:
                    issues.append(f"SYNTAX_ERROR_{e.lineno}")

        except Exception as e:
            issues.append(f"ANALYSIS_ERROR_{str(e)}")

        return issues

    def analyze_documentation_file(self, filepath):
        """Analyze documentation files"""
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check for TODO items
            if 'TODO' in content.upper():
                issues.append("UNRESOLVED_TODO")

            # Check for incomplete sections
            if 'TBD' in content.upper() or 'TO BE DETERMINED' in content.upper():
                issues.append("INCOMPLETE_SECTIONS")

            # Check for broken links (basic check)
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, content)
            for text, url in links:
                if url.startswith('http') and 'localhost' not in url:
                    # Could check if links are valid, but skip for now
                    pass

        except Exception as e:
            issues.append(f"DOC_ANALYSIS_ERROR_{str(e)}")

        return issues

    def analyze_config_file(self, filepath):
        """Analyze configuration files"""
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check JSON validity
            if filepath.suffix == '.json':
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    issues.append(f"INVALID_JSON_{e.lineno}")

            # Check for hardcoded values that should be environment variables
            hardcoded_patterns = [
                r'localhost:\d+',
                r'127\.0\.0\.1:\d+',
                r'/hardcoded/path',
                r'C:\\hardcoded\\path'
            ]
            for pattern in hardcoded_patterns:
                if re.search(pattern, content):
                    issues.append("HARDCODED_VALUES")

        except Exception as e:
            issues.append(f"CONFIG_ANALYSIS_ERROR_{str(e)}")

        return issues

    def is_outdated_file(self, filepath):
        """Check if file appears to be outdated"""
        filename = str(filepath).lower()

        # Old naming patterns
        old_patterns = [
            'old', 'backup', 'bak', 'orig', 'temp', 'tmp',
            'deprecated', 'obsolete', 'legacy', 'archive',
            'draft', 'work_in_progress', 'wip'
        ]

        for pattern in old_patterns:
            if pattern in filename:
                return True

        # Files older than 6 months
        try:
            mtime = filepath.stat().st_mtime
            age_days = (datetime.now().timestamp() - mtime) / (24 * 3600)
            if age_days > 180:  # 6 months
                return True
        except:
            pass

        return False

    def check_security_concerns(self, filepath):
        """Check for security concerns in file"""
        issues = []
        filename = str(filepath).lower()

        # Check filename for sensitive information
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'auth', 'credential',
            'private', 'confidential', 'sensitive'
        ]

        for pattern in sensitive_patterns:
            if pattern in filename:
                issues.append(f"SENSITIVE_FILENAME_{pattern.upper()}")

        # Check file permissions
        try:
            stat = filepath.stat()
            permissions = oct(stat.st_mode)[-3:]

            # Check if file is world-writable
            if permissions[-1] in ['6', '7']:
                issues.append("WORLD_WRITABLE")

            # Check if file is executable when it shouldn't be
            if filepath.suffix in ['.txt', '.md', '.json', '.yaml', '.yml'] and permissions[-1] in ['5', '7']:
                issues.append("UNNECESSARY_EXECUTABLE")

        except:
            pass

        return issues

    def analyze_directory_structure(self):
        """Analyze directory structure and organization"""
        self.analysis["directories_analysis"] = {}

        for root, dirs, files in os.walk(self.workspace_dir):
            if '.git' in root or '__pycache__' in root:
                continue

            rel_root = Path(root).relative_to(self.workspace_dir)
            dir_info = {
                "path": str(rel_root),
                "subdirs": len(dirs),
                "files": len(files),
                "total_size": 0,
                "issues": []
            }

            # Calculate total size
            for file in files:
                try:
                    filepath = Path(root) / file
                    if filepath.exists():
                        dir_info["total_size"] += filepath.stat().st_size
                except:
                    pass

            # Check for problematic directory structures
            if len(dirs) > 50:
                dir_info["issues"].append("TOO_MANY_SUBDIRS")
            if len(files) > 200:
                dir_info["issues"].append("TOO_MANY_FILES")
            if dir_info["total_size"] > 1024 * 1024 * 1024:  # 1GB
                dir_info["issues"].append("VERY_LARGE_DIRECTORY")

            # Check for empty directories
            if not dirs and not files:
                dir_info["issues"].append("EMPTY_DIRECTORY")

            # Check for mixed content types
            extensions = [Path(f).suffix for f in files if f]
            if len(set(extensions)) > 10:
                dir_info["issues"].append("MIXED_CONTENT_TYPES")

            self.analysis["directories_analysis"][str(rel_root)] = dir_info

    def analyze_dependencies(self):
        """Analyze dependency relationships"""
        try:
            # Check Python requirements
            req_file = self.workspace_dir / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as f:
                    requirements = f.read().split('\n')

                # Check for outdated packages (basic check)
                for req in requirements:
                    if req.strip() and not req.startswith('#'):
                        if '==' in req:
                            package, version = req.split('==', 1)
                            # Could check version against latest, but skip for now
                            pass

            # Check for missing __init__.py files in Python packages
            for root, dirs, files in os.walk(self.workspace_dir):
                if any(f.endswith('.py') for f in files):
                    init_file = Path(root) / "__init__.py"
                    if not init_file.exists():
                        self.analysis["missing_dependencies"].append(f"Missing __init__.py in {Path(root).relative_to(self.workspace_dir)}")

        except Exception as e:
            self.analysis["missing_dependencies"].append(f"Dependency analysis error: {e}")

    def run_full_analysis(self):
        """Run complete fine-tooth analysis"""
        print("🔍 FINE TOOTH COMB ANALYSIS")
        print("=" * 80)

        print("📊 Analyzing directory structure...")
        self.analyze_directory_structure()

        print("🔗 Analyzing dependencies...")
        self.analyze_dependencies()

        print("📁 Analyzing all files...")
        file_count = 0

        for root, dirs, files in os.walk(self.workspace_dir):
            # Skip problematic directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', '.git']]

            for file in files:
                filepath = Path(root) / file
                file_info = self.analyze_file(filepath)

                # Categorize by type
                file_type = file_info.get("type", "unknown")
                self.analysis["files_by_type"][file_type].append(file_info)

                # Categorize by size
                size_mb = file_info.get("size", 0) / (1024 * 1024)
                if size_mb < 1:
                    size_category = "small"
                elif size_mb < 10:
                    size_category = "medium"
                elif size_mb < 100:
                    size_category = "large"
                else:
                    size_category = "huge"
                self.analysis["files_by_size"][size_category].append(file_info["path"])

                file_count += 1
                if file_count % 1000 == 0:
                    print(f"   Analyzed {file_count} files...")

        print(f"✅ Analyzed {file_count} files total")

        # Generate summary
        self.generate_summary()

        return self.analysis

    def generate_summary(self):
        """Generate comprehensive analysis summary"""
        print("\n" + "=" * 80)
        print("🎯 ANALYSIS SUMMARY")
        print("=" * 80)

        # File type breakdown
        print("\n📊 FILES BY TYPE:")
        total_files = sum(len(files) for files in self.analysis["files_by_type"].values())
        for file_type, files in sorted(self.analysis["files_by_type"].items(), key=lambda x: len(x[1]), reverse=True):
            percentage = (len(files) / total_files) * 100 if total_files > 0 else 0
            print(f"   {file_type}: {len(files)} files ({percentage:.1f}%)")

        # Size distribution
        print("\n📏 FILES BY SIZE:")
        for size_cat in ["small", "medium", "large", "huge"]:
            files = self.analysis["files_by_size"][size_cat]
            print(f"   {size_cat.title()}: {len(files)} files")

        # Issues summary
        all_issues = []
        for file_list in self.analysis["files_by_type"].values():
            for file_info in file_list:
                all_issues.extend(file_info.get("issues", []))

        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1

        print("\n⚠️  ISSUES FOUND:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {issue}: {count} occurrences")

        # Critical issues
        critical_issues = [
            "HARDCODED_SECRETS", "WORLD_WRITABLE", "SYNTAX_ERROR",
            "INVALID_JSON", "READ_ERROR"
        ]

        critical_found = []
        for issue in critical_issues:
            if issue in issue_counts:
                critical_found.append(f"{issue}: {issue_counts[issue]}")

        if critical_found:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in critical_found:
                print(f"   ❌ {issue}")
        else:
            print("\n✅ NO CRITICAL ISSUES FOUND")

        # Directory issues
        dir_issues = []
        for dir_path, dir_info in self.analysis["directories_analysis"].items():
            if dir_info.get("issues"):
                dir_issues.extend([f"{dir_path}: {issue}" for issue in dir_info["issues"]])

        if dir_issues:
            print("\n📁 DIRECTORY ISSUES:")
            for issue in dir_issues[:10]:  # Show first 10
                print(f"   📁 {issue}")
            if len(dir_issues) > 10:
                print(f"   ... and {len(dir_issues) - 10} more")

        # Duplicates
        if self.analysis["duplicates"]:
            print("\n🔄 DUPLICATE FILES:")
            print(f"   Found {len(self.analysis['duplicates'])} duplicate groups")
            for hash_val, files in list(self.analysis["duplicates"].items())[:5]:
                print(f"   📋 {files[0]} ↔ {files[1]}")

        # Outdated files
        if self.analysis["outdated_files"]:
            print("\n📅 POTENTIALLY OUTDATED FILES:")
            print(f"   Found {len(self.analysis['outdated_files'])} potentially outdated files")
            for file in self.analysis["outdated_files"][:5]:
                print(f"   📅 {file}")

        # Performance concerns
        if self.analysis["performance_concerns"]:
            print("\n⚡ PERFORMANCE CONCERNS:")
            for concern in self.analysis["performance_concerns"][:5]:
                print(f"   ⚡ {concern}")

        # Security concerns
        security_issues = [issue for file_list in self.analysis["files_by_type"].values()
                          for file_info in file_list
                          for issue in file_info.get("issues", [])
                          if "SECURITY" in issue or "SECRET" in issue or "WRITABLE" in issue]

        if security_issues:
            print("\n🔐 SECURITY CONCERNS:")
            unique_security = list(set(security_issues))
            for issue in unique_security[:5]:
                count = security_issues.count(issue)
                print(f"   🔐 {issue}: {count} occurrences")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print(f"   1. Review {len(self.analysis['outdated_files'])} potentially outdated files")
        print(f"   2. Address {len(dir_issues)} directory structure issues")
        print(f"   3. Remove {len(self.analysis['duplicates'])} duplicate file groups")
        print(f"   4. Fix {sum(issue_counts.values())} code quality issues")
        print(f"   5. Review {len(self.analysis['performance_concerns'])} performance concerns")

        # Overall assessment
        critical_count = len([issue for issue in all_issues if any(crit in issue for crit in ["SECRET", "SYNTAX", "SECURITY"])])
        issue_percentage = (len(all_issues) / total_files) * 100 if total_files > 0 else 0

        print("\n🏆 OVERALL ASSESSMENT:")
        if critical_count == 0 and issue_percentage < 5:
            print("   🌟 EXCELLENT: Clean, production-ready codebase")
        elif critical_count == 0 and issue_percentage < 15:
            print("   ✅ GOOD: Well-maintained codebase with minor issues")
        elif critical_count < 10:
            print("   ⚠️  ACCEPTABLE: Functional codebase needing cleanup")
        else:
            print("   ❌ NEEDS ATTENTION: Significant issues requiring fixes")

        print(f"\n📄 Detailed analysis saved to fine_tooth_analysis.json")

        # Save detailed analysis
        with open("/Users/coo-koba42/dev/fine_tooth_analysis.json", "w") as f:
            json.dump(self.analysis, f, indent=2, default=str)

def main():
    """Main analysis execution"""
    analyzer = FineToothAnalyzer()
    results = analyzer.run_full_analysis()

    # Return summary for potential automation
    critical_issues = sum(1 for file_list in results["files_by_type"].values()
                         for file_info in file_list
                         for issue in file_info.get("issues", [])
                         if any(crit in issue for crit in ["SECRET", "SYNTAX", "SECURITY"]))

    return critical_issues

if __name__ == "__main__":
    critical_count = main()
    print(f"\n🎯 Analysis complete. Found {critical_count} critical issues.")
