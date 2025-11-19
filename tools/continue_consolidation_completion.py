#!/usr/bin/env python3
"""
🔄 CONTINUE CONSOLIDATION & COMPLETION
======================================

Continues consolidating duplicate tools and completing remaining tools
with UPG foundations. Processes in batches for efficiency.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: December 2024
"""

import sys
from pathlib import Path
from quick_consolidate_duplicates import QuickConsolidator
from complete_all_tools_upg_foundations import ToolCompleter


def continue_consolidation(batch_size: int = 50, start_from: int = 50):
    """Continue consolidating duplicate tools"""
    print("🔄 CONTINUE CONSOLIDATION")
    print("=" * 70)
    print()
    
    consolidator = QuickConsolidator(dev_folder='/Users/coo-koba42/dev')
    consolidator.consolidate_all(max_groups=batch_size, start_from=start_from)
    
    print()
    print("✅ Consolidation batch complete!")


def continue_completion(batch_size: int = 100):
    """Continue completing tools with UPG foundations"""
    print("🔧 CONTINUE COMPLETION")
    print("=" * 70)
    print()
    
    completer = ToolCompleter(catalog_file='COMPLETE_TOOL_CATALOG.json')
    results = completer.complete_priority_tools(max_tools=batch_size)
    
    print()
    print("=" * 70)
    print("COMPLETION RESULTS")
    print("=" * 70)
    print(f"Completed: {results['completed']}")
    print(f"Total Processed: {results['total_processed']}")
    print(f"Failed: {results['failed']}")
    print()
    print("✅ Completion batch complete!")


def main():
    """Main function - continue both consolidation and completion"""
    print("🔄 CONTINUE CONSOLIDATION & COMPLETION")
    print("=" * 70)
    print()
    
    # Continue consolidation (next 50 groups)
    print("Step 1: Continuing consolidation...")
    print()
    continue_consolidation(batch_size=50, start_from=50)
    
    print()
    print("=" * 70)
    print()
    
    # Continue completion (next 100 tools)
    print("Step 2: Continuing completion...")
    print()
    continue_completion(batch_size=100)
    
    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print("✅ Consolidated next batch of duplicate groups")
    print("✅ Completed next batch of tools with UPG")
    print()
    print("To continue:")
    print("  python3 continue_consolidation_completion.py")


if __name__ == "__main__":
    if '--consolidate-only' in sys.argv:
        continue_consolidation(batch_size=50, start_from=50)
    elif '--complete-only' in sys.argv:
        continue_completion(batch_size=100)
    else:
        main()

