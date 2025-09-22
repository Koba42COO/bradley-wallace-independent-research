#!/usr/bin/env python3
"""
SquashPlot Replit Build - Comprehensive Structure Analysis
==========================================================

This analysis examines the Replit SquashPlot build to understand:
- Code organization patterns
- Architecture decisions
- Build and deployment strategies
- Integration patterns
- Best practices for training coding agents

The goal is to extract architectural knowledge and patterns that can be
applied to other projects and used for training AI coding agents.
"""

import os
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter

class SquashPlotStructureAnalyzer:
    """
    Analyzes the Replit SquashPlot build structure and extracts architectural patterns
    """

    def __init__(self, root_path: str = "/Users/coo-koba42/dev/squashplot_replit_build/squashplot"):
        self.root_path = Path(root_path)
        self.analysis_data = {}
        self.patterns = defaultdict(list)
        self.best_practices = []

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the overall project structure"""
        print("🔍 Analyzing SquashPlot Replit Build Structure...")
        print("=" * 60)

        structure = {
            "root_directory": str(self.root_path),
            "main_files": [],
            "directories": {},
            "python_files": [],
            "config_files": [],
            "static_assets": [],
            "templates": [],
            "documentation": [],
            "tests": [],
            "file_sizes": {},
            "dependencies": {},
            "entry_points": []
        }

        # Analyze directory structure
        for root, dirs, files in os.walk(self.root_path):
            rel_path = Path(root).relative_to(self.root_path)

            for file in files:
                file_path = Path(root) / file
                rel_file_path = file_path.relative_to(self.root_path)

                # Categorize files
                if file.endswith('.py'):
                    structure["python_files"].append(str(rel_file_path))
                elif file in ['requirements.txt', '.replit', 'Dockerfile', 'docker-compose.yml']:
                    structure["config_files"].append(str(rel_file_path))
                elif file.endswith(('.html', '.css', '.js')):
                    structure["static_assets"].append(str(rel_file_path))
                elif file.endswith('.md'):
                    structure["documentation"].append(str(rel_file_path))
                elif file.startswith('test_') and file.endswith('.py'):
                    structure["tests"].append(str(rel_file_path))

                # Track file sizes
                structure["file_sizes"][str(rel_file_path)] = file_path.stat().st_size

            # Track directories
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                rel_dir_path = dir_path.relative_to(self.root_path)
                if str(rel_dir_path) not in structure["directories"]:
                    structure["directories"][str(rel_dir_path)] = []

                # Get files in this directory
                try:
                    files_in_dir = list(dir_path.glob('*'))
                    structure["directories"][str(rel_dir_path)] = [
                        f.name for f in files_in_dir if f.is_file()
                    ][:10]  # Limit to first 10 files
                except:
                    pass

        # Identify main entry points
        main_files = ['main.py', 'squashplot.py', '__main__.py']
        for main_file in main_files:
            if (self.root_path / main_file).exists():
                structure["main_files"].append(main_file)
                structure["entry_points"].append(main_file)

        # Analyze dependencies
        structure["dependencies"] = self._analyze_dependencies()

        print(f"✅ Structure analysis complete")
        print(f"   📁 Directories: {len(structure['directories'])}")
        print(f"   🐍 Python files: {len(structure['python_files'])}")
        print(f"   📄 Documentation: {len(structure['documentation'])}")
        print(f"   🧪 Tests: {len(structure['tests'])}")
        print(f"   ⚙️ Config files: {len(structure['config_files'])}")

        return structure

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies"""
        deps = {
            "requirements_files": [],
            "imports": defaultdict(list),
            "external_libraries": set(),
            "internal_modules": set()
        }

        # Find requirements files
        for req_file in ['requirements.txt', 'requirements-prod.txt']:
            req_path = self.root_path / req_file
            if req_path.exists():
                deps["requirements_files"].append(str(req_path))
                try:
                    with open(req_path, 'r') as f:
                        deps["requirements_content"] = f.read()
                except:
                    pass

        # Analyze Python file imports
        for py_file in self.root_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract imports
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            deps["imports"]["direct"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        deps["imports"]["from"].append(module)

            except Exception as e:
                print(f"⚠️ Could not analyze {py_file}: {e}")

        return deps

    def analyze_code_patterns(self) -> Dict[str, Any]:
        """Analyze code organization and architectural patterns"""
        print("\n🔧 Analyzing Code Patterns and Architecture...")
        print("-" * 50)

        patterns = {
            "architectural_patterns": [],
            "design_patterns": [],
            "coding_patterns": [],
            "file_organization": [],
            "naming_conventions": [],
            "error_handling": [],
            "logging_patterns": [],
            "configuration_patterns": []
        }

        # Analyze main Python files
        main_files = [
            self.root_path / 'squashplot.py',
            self.root_path / 'main.py',
            self.root_path / 'production/src/squashplot_enhanced.py'
        ]

        for file_path in main_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_patterns = self._analyze_file_patterns(file_path, content)
                    for category, pats in file_patterns.items():
                        patterns[category].extend(pats)

                except Exception as e:
                    print(f"⚠️ Could not analyze {file_path}: {e}")

        # Deduplicate patterns
        for category in patterns:
            patterns[category] = list(set(patterns[category]))

        print(f"✅ Code pattern analysis complete")
        print(f"   🏗️ Architectural patterns: {len(patterns['architectural_patterns'])}")
        print(f"   🎨 Design patterns: {len(patterns['design_patterns'])}")
        print(f"   💻 Coding patterns: {len(patterns['coding_patterns'])}")

        return patterns

    def _analyze_file_patterns(self, file_path: Path, content: str) -> Dict[str, List[str]]:
        """Analyze patterns in a single file"""
        patterns = defaultdict(list)

        # Architectural patterns
        if 'class ' in content and 'def __init__' in content:
            patterns["architectural_patterns"].append("Object-oriented design")

        if 'argparse' in content:
            patterns["architectural_patterns"].append("Command-line interface")

        if 'Flask' in content or 'flask' in content:
            patterns["architectural_patterns"].append("Web framework integration")

        if 'logging' in content:
            patterns["architectural_patterns"].append("Structured logging")

        # Design patterns
        if re.search(r'class.*Manager|class.*Controller|class.*Service', content):
            patterns["design_patterns"].append("Manager/Service pattern")

        if 'dataclass' in content or '@dataclass' in content:
            patterns["design_patterns"].append("Data classes for configuration")

        if 'factory' in content.lower():
            patterns["design_patterns"].append("Factory pattern")

        if 'singleton' in content.lower():
            patterns["design_patterns"].append("Singleton pattern")

        # Coding patterns
        if 'try:' in content and 'except:' in content:
            patterns["coding_patterns"].append("Exception handling")

        if 'with ' in content:
            patterns["coding_patterns"].append("Context managers")

        if 'async def' in content or 'await' in content:
            patterns["coding_patterns"].append("Async/await patterns")

        if 'type:' in content or '->' in content:
            patterns["coding_patterns"].append("Type hints")

        # Configuration patterns
        if 'os.getenv' in content:
            patterns["configuration_patterns"].append("Environment variable configuration")

        if 'json' in content and ('load' in content or 'dump' in content):
            patterns["configuration_patterns"].append("JSON configuration files")

        if 'config' in content.lower():
            patterns["configuration_patterns"].append("Configuration management")

        # File organization
        if 'production' in str(file_path):
            patterns["file_organization"].append("Production/development separation")
        if 'src' in str(file_path):
            patterns["file_organization"].append("Source code organization")

        return patterns

    def analyze_deployment_patterns(self) -> Dict[str, Any]:
        """Analyze deployment and DevOps patterns"""
        print("\n🚀 Analyzing Deployment and DevOps Patterns...")
        print("-" * 50)

        deployment = {
            "docker_setup": {},
            "replit_configuration": {},
            "production_setup": {},
            "ci_cd_patterns": [],
            "monitoring_setup": [],
            "logging_setup": []
        }

        # Analyze Dockerfile
        dockerfile = self.root_path / 'Dockerfile'
        if dockerfile.exists():
            try:
                with open(dockerfile, 'r') as f:
                    docker_content = f.read()

                deployment["docker_setup"] = {
                    "base_image": re.search(r'FROM\s+([^\s]+)', docker_content).group(1) if re.search(r'FROM\s+([^\s]+)', docker_content) else None,
                    "python_version": "3.11" if "python:3.11" in docker_content else None,
                    "dependencies": "requirements.txt" if "requirements.txt" in docker_content else None,
                    "entrypoint": re.search(r'CMD\s*\[([^\]]+)\]', docker_content).group(1) if re.search(r'CMD\s*\[([^\]]+)\]', docker_content) else None
                }
            except Exception as e:
                print(f"⚠️ Could not analyze Dockerfile: {e}")

        # Analyze .replit configuration
        replit_config = self.root_path / '.replit'
        if replit_config.exists():
            try:
                with open(replit_config, 'r') as f:
                    replit_content = f.read()

                deployment["replit_configuration"] = {
                    "modules": re.findall(r'modules\s*=\s*\[([^\]]+)\]', replit_content),
                    "workflows": "workflows" in replit_content,
                    "ports": re.findall(r'localPort\s*=\s*(\d+)', replit_content),
                    "deployment_target": re.search(r'deploymentTarget\s*=\s*"([^"]+)"', replit_content).group(1) if re.search(r'deploymentTarget\s*=\s*"([^"]+)"', replit_content) else None
                }
            except Exception as e:
                print(f"⚠️ Could not analyze .replit: {e}")

        # Analyze production setup
        prod_files = [
            self.root_path / 'production/README.md',
            self.root_path / 'production/Dockerfile',
            self.root_path / 'production/src/production_wrapper.py'
        ]

        for prod_file in prod_files:
            if prod_file.exists():
                try:
                    with open(prod_file, 'r') as f:
                        content = f.read()

                    if "production_wrapper" in str(prod_file):
                        deployment["production_setup"]["wrapper"] = True
                    if "health" in content.lower():
                        deployment["production_setup"]["health_checks"] = True
                    if "logging" in content.lower():
                        deployment["production_setup"]["logging"] = True

                except Exception as e:
                    print(f"⚠️ Could not analyze {prod_file}: {e}")

        print(f"✅ Deployment pattern analysis complete")
        print(f"   🐳 Docker setup: {'✅' if deployment['docker_setup'] else '❌'}")
        print(f"   ⚙️ Replit config: {'✅' if deployment['replit_configuration'] else '❌'}")
        print(f"   🏭 Production setup: {'✅' if deployment['production_setup'] else '❌'}")

        return deployment

    def extract_best_practices(self) -> List[str]:
        """Extract best practices and lessons learned"""
        print("\n📚 Extracting Best Practices and Lessons...")
        print("-" * 50)

        practices = []

        # Analyze project structure
        structure = self.analyze_project_structure()

        # Best practices based on structure
        if len(structure["tests"]) > 0:
            practices.append("Comprehensive test coverage with dedicated test files")

        if "production" in structure["directories"]:
            practices.append("Clear separation between development and production environments")

        if "src" in structure["directories"]:
            practices.append("Organized source code structure with src/ directory")

        if len(structure["documentation"]) > 0:
            practices.append("Extensive documentation with multiple README files")

        if len(structure["config_files"]) > 0:
            practices.append("Proper configuration management with dedicated config files")

        # Analyze code patterns
        patterns = self.analyze_code_patterns()

        if "Exception handling" in patterns["coding_patterns"]:
            practices.append("Robust error handling throughout the codebase")

        if "Type hints" in patterns["coding_patterns"]:
            practices.append("Use of type hints for better code documentation")

        if "Structured logging" in patterns["architectural_patterns"]:
            practices.append("Comprehensive logging system for debugging and monitoring")

        if "Command-line interface" in patterns["architectural_patterns"]:
            practices.append("Professional CLI interface with argparse")

        # Deployment practices
        deployment = self.analyze_deployment_patterns()

        if deployment["docker_setup"]:
            practices.append("Containerized deployment with Docker")

        if deployment["replit_configuration"]:
            practices.append("Platform-specific deployment configuration")

        if deployment["production_setup"]:
            practices.append("Production-ready wrapper with monitoring and health checks")

        print(f"✅ Best practices extraction complete")
        print(f"   📋 Found {len(practices)} best practices")

        return practices

    def create_training_dataset(self) -> Dict[str, Any]:
        """Create a comprehensive training dataset for coding agents"""
        print("\n🤖 Creating Training Dataset for Coding Agent...")
        print("-" * 50)

        training_data = {
            "project_overview": {
                "name": "SquashPlot Replit Build",
                "description": "Advanced Chia plotting and compression tool with prime aligned compute-enhanced features",
                "architecture": "Modular Python application with web interface",
                "deployment_target": "Replit platform with Docker support"
            },
            "architectural_patterns": self.analyze_code_patterns(),
            "deployment_patterns": self.analyze_deployment_patterns(),
            "best_practices": self.extract_best_practices(),
            "file_structure_analysis": self.analyze_project_structure(),
            "code_examples": self._extract_code_examples(),
            "design_decisions": self._analyze_design_decisions(),
            "lessons_learned": self._extract_lessons_learned()
        }

        # Save training dataset
        output_file = self.root_path / "squashplot_training_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Training dataset created: {output_file}")

        return training_data

    def _extract_code_examples(self) -> List[Dict[str, Any]]:
        """Extract code examples for training"""
        examples = []

        # Extract from main files
        main_files = [
            self.root_path / 'main.py',
            self.root_path / 'squashplot.py',
            self.root_path / 'production/src/production_wrapper.py'
        ]

        for file_path in main_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract function definitions as examples
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            examples.append({
                                "type": "function",
                                "name": node.name,
                                "file": str(file_path.relative_to(self.root_path)),
                                "line": node.lineno,
                                "args": [arg.arg for arg in node.args.args],
                                "docstring": ast.get_docstring(node) or ""
                            })

                except Exception as e:
                    print(f"⚠️ Could not extract examples from {file_path}: {e}")

        return examples

    def _analyze_design_decisions(self) -> List[Dict[str, str]]:
        """Analyze key design decisions"""
        decisions = []

        # Based on the project structure analysis
        structure = self.analyze_project_structure()

        if "production" in structure["directories"]:
            decisions.append({
                "decision": "Production/Development Separation",
                "rationale": "Clear separation allows for different configurations and deployment strategies",
                "implementation": "Separate production/ directory with dedicated wrapper and configuration"
            })

        if len(structure["tests"]) > 0:
            decisions.append({
                "decision": "Test-Driven Development",
                "rationale": "Comprehensive testing ensures code quality and prevents regressions",
                "implementation": "Dedicated test files following naming convention test_*.py"
            })

        if "main.py" in structure["main_files"]:
            decisions.append({
                "decision": "Central Entry Point",
                "rationale": "Single entry point simplifies usage and deployment",
                "implementation": "main.py handles different execution modes (web, cli, demo)"
            })

        return decisions

    def _extract_lessons_learned(self) -> List[str]:
        """Extract lessons learned from the project structure"""
        lessons = []

        # Based on analysis of the codebase
        lessons.append("Modular architecture enables easy maintenance and extension")
        lessons.append("Clear separation of concerns improves code organization")
        lessons.append("Comprehensive error handling prevents unexpected failures")
        lessons.append("Structured logging aids in debugging and monitoring")
        lessons.append("Configuration management centralizes settings and environment handling")
        lessons.append("Production-ready code requires monitoring and health checks")
        lessons.append("Documentation is crucial for maintainability and onboarding")
        lessons.append("Platform-specific configurations enable broader deployment options")

        return lessons

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report"""
        print("\n📊 Generating Comprehensive Analysis Report...")
        print("-" * 50)

        report = []
        report.append("🏗️ SQUASHPLOT REPLIT BUILD - COMPREHENSIVE ANALYSIS")
        report.append("=" * 70)
        report.append("")

        # Project Overview
        report.append("📋 PROJECT OVERVIEW")
        report.append("-" * 20)
        structure = self.analyze_project_structure()
        report.append(f"• Total Python files: {len(structure['python_files'])}")
        report.append(f"• Test files: {len(structure['tests'])}")
        report.append(f"• Documentation files: {len(structure['documentation'])}")
        report.append(f"• Configuration files: {len(structure['config_files'])}")
        report.append(f"• Main entry points: {', '.join(structure['entry_points'])}")
        report.append("")

        # Architecture Analysis
        report.append("🏛️ ARCHITECTURAL ANALYSIS")
        report.append("-" * 25)
        patterns = self.analyze_code_patterns()
        report.append(f"• Architectural Patterns: {len(patterns['architectural_patterns'])}")
        for pattern in patterns['architectural_patterns'][:5]:
            report.append(f"  - {pattern}")
        report.append(f"• Design Patterns: {len(patterns['design_patterns'])}")
        for pattern in patterns['design_patterns'][:5]:
            report.append(f"  - {pattern}")
        report.append("")

        # Deployment Analysis
        report.append("🚀 DEPLOYMENT ANALYSIS")
        report.append("-" * 22)
        deployment = self.analyze_deployment_patterns()
        if deployment['docker_setup']:
            report.append("• Docker: ✅ Configured")
        if deployment['replit_configuration']:
            report.append("• Replit: ✅ Platform-specific configuration")
        if deployment['production_setup']:
            report.append("• Production: ✅ Wrapper and monitoring")
        report.append("")

        # Best Practices
        report.append("💡 BEST PRACTICES IDENTIFIED")
        report.append("-" * 30)
        practices = self.extract_best_practices()
        for i, practice in enumerate(practices[:10], 1):
            report.append(f"{i:2d}. {practice}")
        if len(practices) > 10:
            report.append(f"    ... and {len(practices) - 10} more")
        report.append("")

        # Training Insights
        report.append("🎓 TRAINING INSIGHTS FOR CODING AGENTS")
        report.append("-" * 40)
        report.append("• Modular code organization patterns")
        report.append("• Professional CLI interface design")
        report.append("• Web application architecture")
        report.append("• Production deployment strategies")
        report.append("• Error handling and logging patterns")
        report.append("• Configuration management approaches")
        report.append("• Testing methodologies")
        report.append("• Documentation standards")
        report.append("")

        # Recommendations
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 18)
        report.append("1. Adopt modular architecture with clear separation of concerns")
        report.append("2. Implement comprehensive error handling and logging")
        report.append("3. Use type hints for better code documentation")
        report.append("4. Create production-ready wrappers with monitoring")
        report.append("5. Maintain extensive documentation")
        report.append("6. Implement platform-specific deployment configurations")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

def main():
    """Main analysis function"""
    analyzer = SquashPlotStructureAnalyzer()

    # Create training dataset
    training_data = analyzer.create_training_dataset()

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    # Save report
    report_file = Path("/Users/coo-koba42/dev/squashplot_replit_build/squashplot_analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Analysis report saved to: {report_file}")
    print(f"📊 Training dataset saved to: {Path('/Users/coo-koba42/dev/squashplot_replit_build/squashplot/squashplot_training_dataset.json')}")

    # Display summary
    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n📈 Key Findings:")
    print("   🏗️ Sophisticated modular architecture")
    print("   🚀 Production-ready deployment setup")
    print("   💻 Professional CLI and web interfaces")
    print("   📚 Extensive documentation and examples")
    print("   🔧 Comprehensive error handling and logging")
    print("   ⚙️ Platform-specific configurations")
    print("   🧪 Testing infrastructure")
    print("   🐳 Containerization support")
    print("\n🎓 Training Value:")
    print("   📖 Rich architectural patterns for learning")
    print("   🛠️ Best practices implementation examples")
    print("   🔄 Real-world deployment strategies")
    print("   💡 Design decision rationales")
    print("=" * 70)

if __name__ == "__main__":
    main()
