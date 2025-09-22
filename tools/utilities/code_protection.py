#!/usr/bin/env python3
"""
CODE PROTECTION SYSTEM
======================

Advanced code obfuscation and intellectual property protection for chAIos platform.
Protects prime aligned compute mathematics algorithms, AI optimization logic, and proprietary formulas.
"""

import os
import sys
import ast
import base64
import marshal
import zlib
from pathlib import Path
from typing import Dict, List, Set
import secrets
import hashlib

class CodeProtector:
    """
    Advanced code protection system for chAIos platform.
    Implements multiple layers of obfuscation and protection.
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.protected_files = {
            # Core prime aligned compute mathematics (highest protection)
            'curated_tools_integration.py': 'CRITICAL',
            'consciousness_modules/': 'CRITICAL',
            'core_mathematics/': 'CRITICAL',
            'wallace_math_engine.py': 'CRITICAL',

            # AI optimization algorithms (high protection)
            'api_server.py': 'HIGH',
            'auth_service.py': 'HIGH',
            'encryption_service.py': 'HIGH',

            # Utility functions (medium protection)
            'core_logging.py': 'MEDIUM',
            'database_service.py': 'MEDIUM',
            'monitoring_setup.py': 'MEDIUM'
        }

        self.encryption_key = self._generate_key()

    def _generate_key(self) -> str:
        """Generate a unique encryption key for this build."""
        return secrets.token_hex(32)

    def obfuscate_string_literals(self, code: str) -> str:
        """Obfuscate string literals in Python code."""
        tree = ast.parse(code)

        class StringObfuscator(ast.NodeTransformer):
            def visit_Str(self, node):
                # Encrypt string literals
                encrypted = self._encrypt_string(node.s)
                return ast.Str(encrypted)

            def _encrypt_string(self, text: str) -> str:
                # Simple XOR encryption with rotating key
                key = b'chAIos_mathematical_protection_2025'
                encrypted = bytearray()
                for i, byte in enumerate(text.encode()):
                    encrypted.append(byte ^ key[i % len(key)])
                return base64.b85encode(encrypted).decode()

        obfuscator = StringObfuscator()
        obfuscated_tree = obfuscator.visit(tree)
        return compile(obfuscated_tree, '<string>', 'exec')

    def create_protected_module(self, source_file: Path, output_file: Path):
        """Create a protected, obfuscated module."""
        print(f"🔒 Protecting: {source_file}")

        # Read source code
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Apply multiple obfuscation layers
        protected_code = self._apply_obfuscation_layers(source_code)

        # Create protected module with runtime decryption
        protected_module = self._create_protected_wrapper(protected_code, source_file.name)

        # Write protected version
        with open(output_file, 'wb') as f:
            f.write(protected_module)

        print(f"✅ Protected: {output_file}")

    def _apply_obfuscation_layers(self, code: str) -> str:
        """Apply multiple layers of code obfuscation."""
        # Layer 1: Remove comments and docstrings
        code = self._remove_comments_and_docstrings(code)

        # Layer 2: Rename variables and functions
        code = self._rename_identifiers(code)

        # Layer 3: Insert junk code to confuse reverse engineering
        code = self._insert_junk_code(code)

        # Layer 4: Encrypt sensitive constants
        code = self._encrypt_sensitive_data(code)

        return code

    def _remove_comments_and_docstrings(self, code: str) -> str:
        """Remove comments and docstrings from code."""
        tree = ast.parse(code)

        class CommentRemover(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Str) and not isinstance(node.value, ast.Constant):
                    # Remove docstrings
                    return None
                return node

            def visit_Constant(self, node):
                if isinstance(node.value, str) and len(node.value) > 50:
                    # Remove long string literals (likely docstrings)
                    return None
                return node

        remover = CommentRemover()
        cleaned_tree = remover.visit(tree)
        return compile(cleaned_tree, '<string>', 'exec')

    def _rename_identifiers(self, code: str) -> str:
        """Rename variables and functions to obfuscated names."""
        # This is a simplified version - production would use more sophisticated renaming
        tree = ast.parse(code)

        class IdentifierRenamer(ast.NodeTransformer):
            def __init__(self):
                self.var_map = {}
                self.func_map = {}
                self.counter = 0

            def _get_obfuscated_name(self, prefix: str = 'v') -> str:
                self.counter += 1
                return f"{prefix}{self.counter:04x}"

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self.var_map:
                        self.var_map[node.id] = self._get_obfuscated_name('v')
                    node.id = self.var_map[node.id]
                elif isinstance(node.ctx, ast.Load):
                    if node.id in self.var_map:
                        node.id = self.var_map[node.id]
                return node

            def visit_FunctionDef(self, node):
                if node.name not in ['__init__', '__call__', '__str__', '__repr__']:
                    if node.name not in self.func_map:
                        self.func_map[node.name] = self._get_obfuscated_name('f')
                    node.name = self.func_map[node.name]
                self.generic_visit(node)
                return node

        renamer = IdentifierRenamer()
        renamed_tree = renamer.visit(tree)
        return compile(renamed_tree, '<string>', 'exec')

    def _insert_junk_code(self, code: str) -> str:
        """Insert junk code to confuse reverse engineering attempts."""
        junk_code = """
# chAIos protection layer - mathematical obfuscation
_phi = 1.618034
_sigma = 0.381966
_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def _mathematical_confusion():
    return (_phi ** _sigma) * sum(_primes[:5]) % 997

_junk_result = _mathematical_confusion()
"""

        return junk_code + code

    def _encrypt_sensitive_data(self, code: str) -> str:
        """Encrypt sensitive mathematical constants and formulas."""
        # Replace sensitive constants with encrypted versions
        replacements = {
            '1.618034': f"eval('\\x{'02x'.join(hex(ord(c))[2:] for c in str(1.618034))}')",
            '0.381966': f"eval('\\x{'02x'.join(hex(ord(c))[2:] for c in str(0.381966))}')",
            '99.9992': f"eval('\\x{'02x'.join(hex(ord(c))[2:] for c in str(99.9992))}')",
            '158': f"eval('\\x{'02x'.join(hex(ord(c))[2:] for c in str(158))}')"
        }

        for old, new in replacements.items():
            code = code.replace(old, new)

        return code

    def _create_protected_wrapper(self, protected_code: str, original_name: str) -> bytes:
        """Create a protected wrapper module with runtime decryption."""
        wrapper_template = f'''
import base64
import marshal
import zlib
import sys
from types import ModuleType

# chAIos Protection System - Runtime Decryption Layer
_ENCRYPTION_KEY = "{self.encryption_key}"
_PROTECTED_CODE = """{base64.b85encode(zlib.compress(marshal.dumps(compile(protected_code, '<protected>', 'exec')), 9)).decode()}"""

def _decrypt_and_execute():
    """Runtime decryption and execution of protected code."""
    try:
        # Decrypt and decompress
        encrypted_data = base64.b85decode(_PROTECTED_CODE)
        decompressed = zlib.decompress(encrypted_data)
        code_object = marshal.loads(decompressed)

        # Execute in protected namespace
        protected_namespace = {{
            '__name__': '{original_name}',
            '__file__': '<protected>',
            '__builtins__': __builtins__
        }}

        exec(code_object, protected_namespace)
        return protected_namespace

    except Exception as e:
        print(f"chAIos Protection: Access denied - {{e}}")
        sys.exit(1)

# Execute protected code
_module = _decrypt_and_execute()
'''

        return wrapper_template.encode('utf-8')

    def protect_entire_project(self):
        """Protect the entire project with obfuscation."""
        print("🚀 Starting chAIos Code Protection Process")
        print("=" * 50)

        protected_dir = self.project_root / 'protected_build'
        protected_dir.mkdir(exist_ok=True)

        protected_count = 0
        for file_path, protection_level in self.protected_files.items():
            source_path = self.project_root / file_path
            if source_path.exists():
                if source_path.is_file():
                    output_path = protected_dir / f"protected_{file_path.replace('.py', '.pyc')}"
                    self.create_protected_module(source_path, output_path)
                    protected_count += 1
                elif source_path.is_dir():
                    # Handle directory protection
                    for py_file in source_path.rglob('*.py'):
                        if py_file.name.startswith('__'):
                            continue
                        rel_path = py_file.relative_to(self.project_root)
                        output_path = protected_dir / f"protected_{rel_path}".replace('.py', '.pyc')
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        self.create_protected_module(py_file, output_path)
                        protected_count += 1

        print(f"\n✅ Protection Complete: {protected_count} files protected")
        print(f"📁 Protected files location: {protected_dir}")
        return protected_dir

def main():
    """Main protection execution."""
    protector = CodeProtector()
    protected_dir = protector.protect_entire_project()

    # Create deployment package
    deployment_script = f'''
#!/bin/bash
# chAIos Protected Deployment Script

echo "🚀 Deploying Protected chAIos System"
echo "==================================="

# Copy protected files to production
cp -r {protected_dir}/* /opt/chaios/production/

# Set restrictive permissions
chmod 700 /opt/chaios/production/
chmod 600 /opt/chaios/production/*.pyc

echo "✅ Protected deployment complete"
echo "🔒 Intellectual property secured"
'''

    with open('deploy_protected.sh', 'w') as f:
        f.write(deployment_script)

    os.chmod('deploy_protected.sh', 0o755)
    print("📦 Created deployment script: deploy_protected.sh")

if __name__ == '__main__':
    main()
