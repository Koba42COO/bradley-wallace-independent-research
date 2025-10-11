#!/usr/bin/env python3
"""
Research Timestamp Utility for Consciousness Mathematics Framework

This utility provides standardized timestamping for all research records,
findings, and documentation in the consciousness mathematics framework.
Ensures chronological integrity and reproducibility of research artifacts.

Author: Christopher Wallace
Created: 2025-10-11 16:45:00 UTC
Framework: Consciousness Mathematics Research Vault
License: Research Framework
"""

import datetime
import pytz
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid

class ResearchTimestampUtility:
    """
    Comprehensive timestamping system for research integrity and reproducibility.

    This class provides standardized timestamping across all research domains,
    ensuring chronological consistency and enabling research artifact tracking.
    """

    def __init__(self, timezone: str = 'UTC', vault_path: Optional[str] = None):
        """
        Initialize the timestamp utility.

        Parameters:
        -----------
        timezone : str
            Default timezone for timestamps (default: UTC)
        vault_path : str, optional
            Path to the research vault for artifact tracking
        """
        self.timezone = pytz.timezone(timezone)
        self.vault_path = Path(vault_path) if vault_path else None

        # Timestamp format standards
        self.formats = {
            'iso': '%Y-%m-%dT%H:%M:%S%z',           # ISO 8601 with timezone
            'compact': '%Y-%m-%d_%H-%M-%S',          # Compact format for filenames
            'readable': '%Y-%m-%d %H:%M:%S %Z',      # Human-readable format
            'date_only': '%Y-%m-%d',                  # Date only
            'time_only': '%H:%M:%S',                  # Time only
        }

        print(f"Research Timestamp Utility initialized")
        print(f"Timezone: {timezone}")
        print(f"Vault Path: {self.vault_path}")

    def get_current_timestamp(self, format_type: str = 'iso') -> str:
        """
        Get current timestamp in specified format.

        Parameters:
        -----------
        format_type : str
            Format type ('iso', 'compact', 'readable', 'date_only', 'time_only')

        Returns:
        --------
        str: Formatted timestamp
        """
        now = datetime.datetime.now(self.timezone)
        return now.strftime(self.formats[format_type])

    def create_research_record(self,
                             domain: str,
                             topic: str,
                             title: str,
                             content_type: str = 'analysis',
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized research record with comprehensive timestamping.

        Parameters:
        -----------
        domain : str
            Research domain (mathematics, physics, consciousness, etc.)
        topic : str
            Specific research topic
        title : str
            Research title or finding description
        content_type : str
            Type of content ('analysis', 'breakthrough', 'visualization', etc.)
        metadata : dict, optional
            Additional metadata for the research record

        Returns:
        --------
        dict: Complete research record with timestamps and identifiers
        """
        timestamp = self.get_current_timestamp('iso')

        record = {
            # Core identifiers
            'id': str(uuid.uuid4()),
            'title': title,
            'domain': domain,
            'topic': topic,
            'content_type': content_type,

            # Timestamping (multiple formats for different uses)
            'created': {
                'iso': timestamp,
                'compact': self.get_current_timestamp('compact'),
                'readable': self.get_current_timestamp('readable'),
                'date': self.get_current_timestamp('date_only'),
                'time': self.get_current_timestamp('time_only'),
                'unix': int(datetime.datetime.now(self.timezone).timestamp())
            },

            # Version control
            'version': '1.0',
            'last_modified': timestamp,

            # Integrity and reproducibility
            'checksum': None,  # To be calculated when content is added
            'size': None,      # File size in bytes

            # Research status
            'status': 'draft',
            'validation_status': 'pending',
            'breakthrough_level': 'analysis',

            # Metadata
            'tags': [domain, topic, content_type],
            'references': [],
            'related_records': [],

            # Additional metadata
            'custom_metadata': metadata or {},

            # Archival information
            'archived': False,
            'retention_period': 'permanent',
            'backup_locations': []
        }

        return record

    def generate_filename(self,
                         domain: str,
                         topic: str,
                         title: str,
                         extension: str = 'md',
                         timestamp: Optional[str] = None) -> str:
        """
        Generate standardized filename with timestamp.

        Parameters:
        -----------
        domain : str
            Research domain
        topic : str
            Research topic
        title : str
            Research title
        extension : str
            File extension (default: 'md')
        timestamp : str, optional
            Specific timestamp to use

        Returns:
        --------
        str: Standardized filename
        """
        if timestamp is None:
            timestamp = self.get_current_timestamp('compact')

        # Sanitize strings for filename
        safe_domain = self._sanitize_filename(domain)
        safe_topic = self._sanitize_filename(topic)
        safe_title = self._sanitize_filename(title)

        filename = f"{timestamp}_{safe_domain}_{safe_topic}_{safe_title}.{extension}"
        return filename

    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filenames."""
        # Replace spaces and special characters with underscores
        import re
        sanitized = re.sub(r'[^\w\-_\.]', '_', text)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    def calculate_checksum(self, content: str) -> str:
        """
        Calculate SHA-256 checksum for content integrity.

        Parameters:
        -----------
        content : str
            Content to checksum

        Returns:
        --------
        str: SHA-256 checksum
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def update_record_checksum(self, record: Dict[str, Any], content: str) -> Dict[str, Any]:
        """
        Update research record with content checksum and size.

        Parameters:
        -----------
        record : dict
            Research record dictionary
        content : str
            Content to analyze

        Returns:
        --------
        dict: Updated record with checksum and size
        """
        record['checksum'] = self.calculate_checksum(content)
        record['size'] = len(content.encode('utf-8'))
        record['last_modified'] = self.get_current_timestamp('iso')
        return record

    def create_version_record(self,
                            original_record: Dict[str, Any],
                            changes: str,
                            new_content: str) -> Dict[str, Any]:
        """
        Create a version record for research record updates.

        Parameters:
        -----------
        original_record : dict
            Original research record
        changes : str
            Description of changes made
        new_content : str
            New content

        Returns:
        --------
        dict: Version record
        """
        version_record = {
            'original_id': original_record['id'],
            'version': self._increment_version(original_record.get('version', '1.0')),
            'timestamp': self.get_current_timestamp('iso'),
            'changes': changes,
            'previous_checksum': original_record.get('checksum'),
            'new_checksum': self.calculate_checksum(new_content),
            'size_change': len(new_content.encode('utf-8')) - original_record.get('size', 0)
        }

        return version_record

    def _increment_version(self, version: str) -> str:
        """Increment version number (semantic versioning)."""
        parts = version.split('.')
        if len(parts) >= 2:
            # Increment minor version
            parts[1] = str(int(parts[1]) + 1)
            # Reset patch version
            if len(parts) >= 3:
                parts[2] = '0'
        else:
            parts = [version, '1', '0']

        return '.'.join(parts)

    def validate_timestamp_chronology(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate chronological consistency of research records.

        Parameters:
        -----------
        records : list
            List of research records to validate

        Returns:
        --------
        dict: Validation results
        """
        validation_results = {
            'total_records': len(records),
            'chronological_consistency': True,
            'anomalies': [],
            'oldest_record': None,
            'newest_record': None,
            'time_span': None
        }

        if not records:
            return validation_results

        # Sort records by creation time
        sorted_records = sorted(records, key=lambda r: r['created']['unix'])

        validation_results['oldest_record'] = sorted_records[0]['created']['iso']
        validation_results['newest_record'] = sorted_records[-1]['created']['iso']

        # Calculate time span
        oldest_time = datetime.datetime.fromtimestamp(sorted_records[0]['created']['unix'], self.timezone)
        newest_time = datetime.datetime.fromtimestamp(sorted_records[-1]['created']['unix'], self.timezone)
        validation_results['time_span'] = str(newest_time - oldest_time)

        # Check for chronological anomalies (future timestamps, etc.)
        now = datetime.datetime.now(self.timezone)
        for record in records:
            record_time = datetime.datetime.fromtimestamp(record['created']['unix'], self.timezone)
            if record_time > now:
                validation_results['chronological_consistency'] = False
                validation_results['anomalies'].append({
                    'record_id': record['id'],
                    'timestamp': record['created']['iso'],
                    'anomaly': 'Future timestamp'
                })

        return validation_results

    def export_timestamp_report(self, records: List[Dict[str, Any]], output_path: str) -> str:
        """
        Export comprehensive timestamp report for research integrity.

        Parameters:
        -----------
        records : list
            Research records to include in report
        output_path : str
            Path to save the report

        Returns:
        --------
        str: Path to generated report
        """
        report = {
            'generated_at': self.get_current_timestamp('iso'),
            'total_records': len(records),
            'validation_results': self.validate_timestamp_chronology(records),
            'records_summary': [],

            # Domain breakdown
            'domain_breakdown': {},
            'topic_breakdown': {},
            'content_type_breakdown': {},

            # Chronological analysis
            'chronological_distribution': {},
            'creation_frequency': {},

            # Integrity metrics
            'integrity_metrics': {
                'records_with_checksums': 0,
                'total_checksums': 0,
                'average_record_size': 0,
                'version_distribution': {}
            }
        }

        # Analyze records
        total_size = 0
        for record in records:
            # Basic summary
            summary = {
                'id': record['id'],
                'title': record['title'],
                'domain': record['domain'],
                'created': record['created']['iso'],
                'status': record['status'],
                'has_checksum': record.get('checksum') is not None
            }
            report['records_summary'].append(summary)

            # Domain breakdown
            domain = record['domain']
            if domain not in report['domain_breakdown']:
                report['domain_breakdown'][domain] = 0
            report['domain_breakdown'][domain] += 1

            # Integrity metrics
            if record.get('checksum'):
                report['integrity_metrics']['records_with_checksums'] += 1
                report['integrity_metrics']['total_checksums'] += 1

            if record.get('size'):
                total_size += record['size']

            # Version distribution
            version = record.get('version', '1.0')
            if version not in report['integrity_metrics']['version_distribution']:
                report['integrity_metrics']['version_distribution'][version] = 0
            report['integrity_metrics']['version_distribution'][version] += 1

        # Calculate averages
        if records:
            report['integrity_metrics']['average_record_size'] = total_size / len(records)

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Timestamp report exported to: {output_path}")
        return output_path

    def create_backup_manifest(self,
                             source_path: str,
                             backup_path: str,
                             records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create backup manifest with timestamp integrity verification.

        Parameters:
        -----------
        source_path : str
            Source vault path
        backup_path : str
            Backup destination path
        records : list
            Research records to include in manifest

        Returns:
        --------
        dict: Backup manifest
        """
        manifest = {
            'backup_created': self.get_current_timestamp('iso'),
            'source_path': source_path,
            'backup_path': backup_path,
            'total_records': len(records),
            'backup_integrity': {},
            'file_manifest': [],

            # Timestamp validation
            'timestamp_validation': self.validate_timestamp_chronology(records),

            # Backup metadata
            'backup_metadata': {
                'compression': None,
                'encryption': None,
                'retention_policy': 'permanent',
                'backup_frequency': 'weekly'
            }
        }

        # Create file manifest
        for record in records:
            file_entry = {
                'record_id': record['id'],
                'title': record['title'],
                'original_path': f"{record['domain']}/{record['topic']}",
                'filename': self.generate_filename(
                    record['domain'],
                    record['topic'],
                    record['title'],
                    record.get('created', {}).get('compact', self.get_current_timestamp('compact'))
                ),
                'checksum': record.get('checksum'),
                'size': record.get('size'),
                'created': record['created']['iso']
            }
            manifest['file_manifest'].append(file_entry)

        # Integrity check
        manifest['backup_integrity'] = {
            'all_records_have_checksums': all(r.get('checksum') for r in records),
            'chronological_consistency': manifest['timestamp_validation']['chronological_consistency'],
            'total_files': len(manifest['file_manifest']),
            'total_size_bytes': sum(r.get('size', 0) for r in records)
        }

        return manifest


def main():
    """Main execution function for timestamp utility demonstration."""
    print("=== Research Timestamp Utility ===")
    print("Initializing timestamping system for consciousness mathematics framework...")

    # Initialize utility
    timestamp_util = ResearchTimestampUtility(
        timezone='UTC',
        vault_path='/Users/coo-koba42/dev/Math_Research_Obsidian_Vault'
    )

    # Demonstrate timestamp creation
    print(f"\nCurrent timestamp (ISO): {timestamp_util.get_current_timestamp('iso')}")
    print(f"Current timestamp (Compact): {timestamp_util.get_current_timestamp('compact')}")
    print(f"Current timestamp (Readable): {timestamp_util.get_current_timestamp('readable')}")

    # Create sample research record
    sample_record = timestamp_util.create_research_record(
        domain='consciousness',
        topic='skyrmion_physics',
        title='Topological Charge Analysis',
        content_type='breakthrough',
        metadata={'validation_level': 'experimental', 'impact_factor': 'high'}
    )

    print(f"\nSample Research Record Created:")
    print(f"ID: {sample_record['id']}")
    print(f"Title: {sample_record['title']}")
    print(f"Created: {sample_record['created']['iso']}")
    print(f"Domain: {sample_record['domain']}")

    # Generate filename
    filename = timestamp_util.generate_filename(
        sample_record['domain'],
        sample_record['topic'],
        sample_record['title']
    )
    print(f"Generated Filename: {filename}")

    # Demonstrate checksum calculation
    sample_content = "# Topological Charge Analysis\n\nThis is a breakthrough in skyrmion physics..."
    checksum = timestamp_util.calculate_checksum(sample_content)
    print(f"Content Checksum: {checksum[:16]}...")

    # Update record with content information
    updated_record = timestamp_util.update_record_checksum(sample_record, sample_content)
    print(f"Record Size: {updated_record['size']} bytes")

    print("\n=== Timestamp Utility Ready for Research Integrity ===")
    print("Features implemented:")
    print("- Standardized timestamping across all formats")
    print("- Research record creation with metadata")
    print("- File naming conventions")
    print("- Content integrity verification")
    print("- Version control support")
    print("- Chronological validation")
    print("- Backup manifest generation")


if __name__ == "__main__":
    main()
