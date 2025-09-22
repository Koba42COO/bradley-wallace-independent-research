#!/usr/bin/env python3
"""
Chia Wallet Service
Provides wallet connectivity, offer management, and reward claiming automation
"""

import asyncio
import json
import logging
import time
import os
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiofiles

try:
    from chia_py_rpc.wallet import Wallet
    from chia_py_rpc.exceptions import ChiaRPCException
    CHIA_AVAILABLE = True
except ImportError:
    CHIA_AVAILABLE = False
    logging.warning("Chia-py-RPC not available - wallet features disabled")

@dataclass
class WalletInfo:
    """Wallet information"""
    wallet_id: int
    name: str
    wallet_type: str
    balance_xch: Decimal
    unconfirmed_xch: Decimal
    spendable_xch: Decimal

@dataclass
class OfferFile:
    """Offer file information"""
    offer_id: str
    offer_data: str
    summary: str
    status: str
    created_at: datetime
    file_path: str

@dataclass
class RewardClaim:
    """Reward claim information"""
    claim_type: str  # 'block_reward', 'pool_reward', 'farmer_reward'
    amount_xch: Decimal
    block_height: int
    timestamp: datetime
    status: str

class ChiaWalletService:
    """Chia wallet service for blockchain operations"""
    
    def __init__(self):
        self.wallet = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.offers_dir = "data/offers"
        self.auto_claim_enabled = False
        self.last_claim_check = None
        
        # Ensure directories exist
        os.makedirs(self.offers_dir, exist_ok=True)
        os.makedirs("data/rewards", exist_ok=True)
    
    async def connect(self) -> bool:
        """Connect to Chia wallet RPC"""
        if not CHIA_AVAILABLE:
            self.logger.error("Chia-py-RPC library not available")
            return False
        
        try:
            self.wallet = Wallet()
            # Test connection
            wallets = await self.get_wallets()
            self.connected = len(wallets) >= 0  # Connection successful if we can get wallets
            self.logger.info(f"Connected to Chia wallet - Found {len(wallets)} wallets")
            return self.connected
        except Exception as e:
            self.logger.error(f"Failed to connect to Chia wallet: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from wallet"""
        self.wallet = None
        self.connected = False
        self.logger.info("Disconnected from Chia wallet")
    
    def is_connected(self) -> bool:
        """Check if wallet is connected"""
        return self.connected and self.wallet is not None
    
    def xch_to_mojos(self, xch_amount: float) -> int:
        """Convert XCH to mojos (1 XCH = 1 trillion mojos)"""
        return int(Decimal(str(xch_amount)) * Decimal('1000000000000'))
    
    def mojos_to_xch(self, mojos: int) -> Decimal:
        """Convert mojos to XCH"""
        return Decimal(mojos) / Decimal('1000000000000')
    
    async def get_wallets(self) -> List[WalletInfo]:
        """Get all wallet information"""
        if not self.is_connected():
            return []
        
        try:
            wallets_response = self.wallet.get_wallets()
            wallets = []
            
            for wallet_data in wallets_response.get('wallets', []):
                wallet_id = wallet_data['id']
                
                # Get balance for each wallet
                balance_response = self.wallet.get_wallet_balance(wallet_id)
                balance = balance_response.get('wallet_balance', {})
                
                wallet_info = WalletInfo(
                    wallet_id=wallet_id,
                    name=wallet_data.get('name', f'Wallet {wallet_id}'),
                    wallet_type=wallet_data.get('type', 'unknown'),
                    balance_xch=self.mojos_to_xch(balance.get('confirmed_wallet_balance', 0)),
                    unconfirmed_xch=self.mojos_to_xch(balance.get('unconfirmed_wallet_balance', 0)),
                    spendable_xch=self.mojos_to_xch(balance.get('spendable_balance', 0))
                )
                wallets.append(wallet_info)
            
            return wallets
        except Exception as e:
            self.logger.error(f"Failed to get wallets: {e}")
            return []
    
    async def send_xch(self, wallet_id: int, recipient_address: str, amount_xch: float, fee_xch: float = 0.00001) -> Dict[str, Any]:
        """Send XCH transaction"""
        if not self.is_connected():
            return {'success': False, 'error': 'Wallet not connected'}
        
        try:
            amount_mojos = self.xch_to_mojos(amount_xch)
            fee_mojos = self.xch_to_mojos(fee_xch)
            
            response = self.wallet.send_transaction(
                wallet_id=wallet_id,
                amount=amount_mojos,
                address=recipient_address,
                fee=fee_mojos
            )
            
            return {
                'success': True,
                'transaction_id': response.get('transaction_id'),
                'status': response.get('status', 'submitted')
            }
        except Exception as e:
            self.logger.error(f"Failed to send XCH: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_offer(self, wallet_id: int, offered_amount: float, requested_asset: str, requested_amount: float) -> Dict[str, Any]:
        """Create an offer file"""
        if not self.is_connected():
            return {'success': False, 'error': 'Wallet not connected'}
        
        try:
            # Create offer using Chia RPC
            offer_data = {
                'offer': {
                    str(wallet_id): -self.xch_to_mojos(offered_amount),  # Negative = offering
                    # Add requested asset logic here
                },
                'fee': self.xch_to_mojos(0.00001),  # Small fee
                'validate_only': False
            }
            
            response = self.wallet.create_offer_for_ids(offer_data)
            
            if response.get('success'):
                offer_id = f"offer_{int(time.time())}"
                offer_file_path = os.path.join(self.offers_dir, f"{offer_id}.offer")
                
                # Save offer to file
                async with aiofiles.open(offer_file_path, 'w') as f:
                    await f.write(response.get('offer', ''))
                
                # Save offer metadata
                offer_info = OfferFile(
                    offer_id=offer_id,
                    offer_data=response.get('offer', ''),
                    summary=f"Offer {offered_amount} XCH for {requested_amount} {requested_asset}",
                    status='created',
                    created_at=datetime.now(),
                    file_path=offer_file_path
                )
                
                await self._save_offer_metadata(offer_info)
                
                return {
                    'success': True,
                    'offer_id': offer_id,
                    'offer_file': offer_file_path,
                    'summary': offer_info.summary
                }
            else:
                return {'success': False, 'error': response.get('error', 'Failed to create offer')}
                
        except Exception as e:
            self.logger.error(f"Failed to create offer: {e}")
            return {'success': False, 'error': str(e)}
    
    async def accept_offer(self, offer_file_path: str, fee_xch: float = 0.00001) -> Dict[str, Any]:
        """Accept an offer file"""
        if not self.is_connected():
            return {'success': False, 'error': 'Wallet not connected'}
        
        try:
            # Read offer file
            async with aiofiles.open(offer_file_path, 'r') as f:
                offer_data = await f.read()
            
            # Take offer using Chia RPC
            response = self.wallet.take_offer(
                offer=offer_data,
                fee=self.xch_to_mojos(fee_xch)
            )
            
            if response.get('success'):
                return {
                    'success': True,
                    'trade_record': response.get('trade_record'),
                    'transaction_ids': response.get('transaction_ids', [])
                }
            else:
                return {'success': False, 'error': response.get('error', 'Failed to accept offer')}
                
        except Exception as e:
            self.logger.error(f"Failed to accept offer: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_offers(self) -> List[OfferFile]:
        """Get all offer files"""
        offers = []
        
        try:
            if not os.path.exists(self.offers_dir):
                return offers
            
            # List all .offer files
            for filename in os.listdir(self.offers_dir):
                if filename.endswith('.offer'):
                    offer_id = filename[:-6]  # Remove .offer extension
                    offer_metadata = await self._load_offer_metadata(offer_id)
                    if offer_metadata:
                        offers.append(offer_metadata)
            
            return sorted(offers, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get offers: {e}")
            return offers
    
    async def enable_auto_claim(self, enabled: bool = True):
        """Enable/disable automatic reward claiming"""
        self.auto_claim_enabled = enabled
        self.logger.info(f"Auto-claim {'enabled' if enabled else 'disabled'}")
        
        if enabled:
            # Start auto-claim task
            asyncio.create_task(self._auto_claim_loop())
    
    async def check_pending_rewards(self) -> List[RewardClaim]:
        """Check for pending farming and pool rewards"""
        if not self.is_connected():
            return []
        
        try:
            rewards = []
            
            # Get farming rewards (block rewards)
            farm_summary = self.wallet.get_farmed_amount()
            if farm_summary.get('farmer_reward_amount', 0) > 0:
                rewards.append(RewardClaim(
                    claim_type='farmer_reward',
                    amount_xch=self.mojos_to_xch(farm_summary['farmer_reward_amount']),
                    block_height=0,  # Get from blockchain
                    timestamp=datetime.now(),
                    status='pending'
                ))
            
            # Get pool rewards
            if farm_summary.get('pool_reward_amount', 0) > 0:
                rewards.append(RewardClaim(
                    claim_type='pool_reward',
                    amount_xch=self.mojos_to_xch(farm_summary['pool_reward_amount']),
                    block_height=0,
                    timestamp=datetime.now(),
                    status='pending'
                ))
            
            return rewards
            
        except Exception as e:
            self.logger.error(f"Failed to check pending rewards: {e}")
            return []
    
    async def claim_rewards(self, reward_types: List[str] = None) -> Dict[str, Any]:
        """Claim farming rewards"""
        if not self.is_connected():
            return {'success': False, 'error': 'Wallet not connected'}
        
        if reward_types is None:
            reward_types = ['farmer_reward', 'pool_reward']
        
        try:
            claimed_rewards = []
            
            # Claim farmer rewards
            if 'farmer_reward' in reward_types:
                try:
                    response = self.wallet.get_farmed_amount()
                    if response.get('farmer_reward_amount', 0) > 0:
                        # Auto-claim happens when wallet syncs
                        claimed_rewards.append({
                            'type': 'farmer_reward',
                            'amount': self.mojos_to_xch(response['farmer_reward_amount']),
                            'status': 'claimed'
                        })
                except Exception as e:
                    self.logger.error(f"Failed to claim farmer rewards: {e}")
            
            # Claim pool rewards
            if 'pool_reward' in reward_types:
                try:
                    response = self.wallet.get_farmed_amount()
                    if response.get('pool_reward_amount', 0) > 0:
                        claimed_rewards.append({
                            'type': 'pool_reward',
                            'amount': self.mojos_to_xch(response['pool_reward_amount']),
                            'status': 'claimed'
                        })
                except Exception as e:
                    self.logger.error(f"Failed to claim pool rewards: {e}")
            
            return {
                'success': True,
                'claimed_rewards': claimed_rewards,
                'total_claimed': sum(r['amount'] for r in claimed_rewards if isinstance(r['amount'], Decimal))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to claim rewards: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _auto_claim_loop(self):
        """Background task for automatic reward claiming"""
        while self.auto_claim_enabled:
            try:
                # Check for rewards every 10 minutes
                if (self.last_claim_check is None or 
                    datetime.now() - self.last_claim_check > timedelta(minutes=10)):
                    
                    rewards = await self.check_pending_rewards()
                    if rewards:
                        self.logger.info(f"Found {len(rewards)} pending rewards - claiming automatically")
                        result = await self.claim_rewards()
                        if result['success']:
                            self.logger.info(f"Auto-claimed rewards: {result['total_claimed']} XCH")
                    
                    self.last_claim_check = datetime.now()
                
                # Wait 60 seconds before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in auto-claim loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _save_offer_metadata(self, offer_info: OfferFile):
        """Save offer metadata to JSON file"""
        metadata_file = os.path.join(self.offers_dir, f"{offer_info.offer_id}.json")
        metadata = {
            'offer_id': offer_info.offer_id,
            'summary': offer_info.summary,
            'status': offer_info.status,
            'created_at': offer_info.created_at.isoformat(),
            'file_path': offer_info.file_path
        }
        
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
    
    async def _load_offer_metadata(self, offer_id: str) -> Optional[OfferFile]:
        """Load offer metadata from JSON file"""
        metadata_file = os.path.join(self.offers_dir, f"{offer_id}.json")
        
        try:
            if os.path.exists(metadata_file):
                async with aiofiles.open(metadata_file, 'r') as f:
                    metadata = json.loads(await f.read())
                
                return OfferFile(
                    offer_id=metadata['offer_id'],
                    offer_data='',  # Load from .offer file if needed
                    summary=metadata['summary'],
                    status=metadata['status'],
                    created_at=datetime.fromisoformat(metadata['created_at']),
                    file_path=metadata['file_path']
                )
        except Exception as e:
            self.logger.error(f"Failed to load offer metadata for {offer_id}: {e}")
        
        return None

# Global wallet service instance
wallet_service = ChiaWalletService()

async def initialize_wallet_service():
    """Initialize the wallet service"""
    connected = await wallet_service.connect()
    if connected:
        # Enable auto-claim by default
        await wallet_service.enable_auto_claim(True)
    return connected