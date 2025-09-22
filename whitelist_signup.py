#!/usr/bin/env python3
"""
SquashPlot Pro Whitelist Sign-up System
========================================

Manage early access to Pro version features with advanced compression.
"""

import json
import hashlib
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional

class WhitelistManager:
    """Manage Pro version whitelist and early access"""

    def __init__(self, whitelist_file: str = None):
        if whitelist_file is None:
            self.whitelist_file = Path.home() / ".squashplot" / "whitelist.json"
        else:
            self.whitelist_file = Path(whitelist_file)

        self.whitelist_file.parent.mkdir(exist_ok=True)
        self.whitelist = self._load_whitelist()

    def _load_whitelist(self) -> Dict:
        """Load whitelist from file"""
        if self.whitelist_file.exists():
            try:
                with open(self.whitelist_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Corrupted whitelist file, starting fresh")
                return {}
        return {}

    def _save_whitelist(self):
        """Save whitelist to file"""
        with open(self.whitelist_file, 'w') as f:
            json.dump(self.whitelist, f, indent=2, default=str)

    def add_to_whitelist(self, email: str, user_info: Dict = None) -> Dict:
        """Add user to whitelist"""

        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]

        if user_id in self.whitelist:
            return {
                'success': False,
                'message': 'Email already registered',
                'user_id': user_id,
                'status': self.whitelist[user_id]['status']
            }

        # Auto-approve certain domains/emails for demo
        auto_approve = self._should_auto_approve(email)

        entry = {
            'user_id': user_id,
            'email': email,
            'status': 'approved' if auto_approve else 'pending',
            'registered_at': datetime.now().isoformat(),
            'approved_at': datetime.now().isoformat() if auto_approve else None,
            'tier': 'pro_beta',
            'features': [
                'advanced_compression_99_5',
                'consciousness_enhancement',
                'real_time_optimization',
                'priority_support'
            ]
        }

        if user_info:
            entry.update(user_info)

        self.whitelist[user_id] = entry
        self._save_whitelist()

        # Send confirmation email
        self._send_confirmation_email(email, entry)

        return {
            'success': True,
            'user_id': user_id,
            'status': entry['status'],
            'message': f'Whitelist {"approved" if auto_approve else "request submitted"}',
            'auto_approved': auto_approve
        }

    def _should_auto_approve(self, email: str) -> bool:
        """Determine if email should be auto-approved"""

        # Auto-approve for demo purposes
        auto_approve_domains = [
            'squashplot.com',
            'chia.net',
            'protonmail.com',
            'tutanota.com'
        ]

        domain = email.split('@')[-1] if '@' in email else ''

        # Auto-approve certain emails for testing
        if any(keyword in email.lower() for keyword in ['admin', 'beta', 'test', 'pro']):
            return True

        if domain in auto_approve_domains:
            return True

        # Auto-approve first 100 users (for demo)
        if len(self.whitelist) < 100:
            return True

        return False

    def check_status(self, email: str = None, user_id: str = None) -> Dict:
        """Check whitelist status"""

        if not email and not user_id:
            return {'error': 'Email or user_id required'}

        if email:
            user_id = hashlib.sha256(email.encode()).hexdigest()[:16]

        if user_id not in self.whitelist:
            return {
                'found': False,
                'message': 'Not found in whitelist',
                'user_id': user_id
            }

        entry = self.whitelist[user_id]

        return {
            'found': True,
            'user_id': user_id,
            'email': entry['email'],
            'status': entry['status'],
            'registered_at': entry['registered_at'],
            'approved_at': entry['approved_at'],
            'tier': entry['tier'],
            'features': entry['features']
        }

    def approve_user(self, user_id: str) -> Dict:
        """Manually approve a user"""

        if user_id not in self.whitelist:
            return {'success': False, 'message': 'User not found'}

        if self.whitelist[user_id]['status'] == 'approved':
            return {'success': False, 'message': 'Already approved'}

        self.whitelist[user_id]['status'] = 'approved'
        self.whitelist[user_id]['approved_at'] = datetime.now().isoformat()
        self._save_whitelist()

        # Send approval email
        self._send_approval_email(self.whitelist[user_id]['email'], self.whitelist[user_id])

        return {
            'success': True,
            'message': 'User approved',
            'user_id': user_id
        }

    def _send_confirmation_email(self, email: str, entry: Dict):
        """Send confirmation email"""

        subject = "SquashPlot Pro Whitelist Registration"
        body = f"""
        Thank you for your interest in SquashPlot Pro!

        Your registration details:
        - Email: {email}
        - User ID: {entry['user_id']}
        - Status: {entry['status']}
        - Tier: {entry['tier']}

        Pro Features Unlocked:
        {'- ' + chr(10).join(entry['features'])}

        {'🎉 Congratulations! You have been auto-approved for Pro access!' if entry['status'] == 'approved' else '⏳ Your request is being reviewed. You will receive an email when approved.'}

        To use Pro features:
        python squashplot.py --pro --input your_plot.dat --output compressed_plot.dat

        Best regards,
        SquashPlot Team
        """

        print(f"📧 Confirmation email would be sent to: {email}")
        print(f"   Subject: {subject}")
        print(f"   Status: {'APPROVED' if entry['status'] == 'approved' else 'PENDING'}")

        # In production, this would send actual email
        # self._send_email(email, subject, body)

    def _send_approval_email(self, email: str, entry: Dict):
        """Send approval email"""

        subject = "🎉 SquashPlot Pro Access Approved!"
        body = f"""
        Congratulations! Your SquashPlot Pro access has been approved!

        Access Details:
        - Email: {email}
        - User ID: {entry['user_id']}
        - Approved: {entry['approved_at']}
        - Tier: {entry['tier']}

        Pro Features Now Available:
        {'- ' + chr(10).join(entry['features'])}

        Start using Pro features:
        python squashplot.py --pro --input your_plot.dat --output compressed_plot.dat

        Advanced compression (99.5%) and prime aligned compute enhancement are now active!

        Welcome to the Pro tier!

        Best regards,
        SquashPlot Team
        """

        print(f"📧 Approval email would be sent to: {email}")
        print(f"   Subject: {subject}")

    def get_stats(self) -> Dict:
        """Get whitelist statistics"""

        total_users = len(self.whitelist)
        approved_users = sum(1 for u in self.whitelist.values() if u['status'] == 'approved')
        pending_users = sum(1 for u in self.whitelist.values() if u['status'] == 'pending')

        return {
            'total_users': total_users,
            'approved_users': approved_users,
            'pending_users': pending_users,
            'approval_rate': approved_users / total_users if total_users > 0 else 0
        }

    def list_users(self, status_filter: str = None) -> List[Dict]:
        """List whitelist users"""

        users = []
        for user_id, entry in self.whitelist.items():
            if status_filter and entry['status'] != status_filter:
                continue

            users.append({
                'user_id': user_id,
                'email': entry['email'],
                'status': entry['status'],
                'registered_at': entry['registered_at'],
                'tier': entry['tier']
            })

        return users

def main():
    """Command-line interface for whitelist management"""

    import argparse

    parser = argparse.ArgumentParser(description="SquashPlot Pro Whitelist Manager")
    parser.add_argument('--add', type=str, help='Add email to whitelist')
    parser.add_argument('--check', type=str, help='Check email status')
    parser.add_argument('--approve', type=str, help='Approve user ID')
    parser.add_argument('--list', action='store_true', help='List all users')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--pending', action='store_true', help='List pending users')

    args = parser.parse_args()

    whitelist = WhitelistManager()

    if args.add:
        result = whitelist.add_to_whitelist(args.add)
        print(f"✅ Result: {result}")

    elif args.check:
        result = whitelist.check_status(email=args.check)
        print(f"📋 Status: {result}")

    elif args.approve:
        result = whitelist.approve_user(args.approve)
        print(f"✅ Result: {result}")

    elif args.list:
        users = whitelist.list_users()
        print("📋 All Users:")
        for user in users:
            print(f"   {user['user_id']}: {user['email']} ({user['status']})")

    elif args.pending:
        users = whitelist.list_users('pending')
        print("⏳ Pending Users:")
        for user in users:
            print(f"   {user['user_id']}: {user['email']}")

    elif args.stats:
        stats = whitelist.get_stats()
        print("📊 Whitelist Statistics:")
        print(f"   Total Users: {stats['total_users']}")
        print(f"   Approved: {stats['approved_users']}")
        print(f"   Pending: {stats['pending_users']}")
        print(".1f")
    else:
        parser.print_help()
        print("\n📚 Examples:")
        print("   Add user: python whitelist_signup.py --add user@domain.com")
        print("   Check status: python whitelist_signup.py --check user@domain.com")
        print("   Approve user: python whitelist_signup.py --approve user_id")
        print("   List pending: python whitelist_signup.py --pending")
        print("   Show stats: python whitelist_signup.py --stats")

if __name__ == "__main__":
    main()
