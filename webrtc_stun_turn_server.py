#!/usr/bin/env python3
"""
Self-Hosted STUN/TURN Server for WebRTC P2P Connections
Implements ICE credential management without external relays
"""

import asyncio
import socket
import struct
import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# STUN/TURN Protocol Constants
STUN_MAGIC_COOKIE = 0x2112A442
STUN_HEADER_SIZE = 20
STUN_BINDING_REQUEST = 0x0001
STUN_BINDING_RESPONSE = 0x0101
STUN_SHARED_SECRET_REQUEST = 0x0002
STUN_SHARED_SECRET_RESPONSE = 0x0102
STUN_ALLOCATE_REQUEST = 0x0003
STUN_ALLOCATE_RESPONSE = 0x0103
STUN_SEND_INDICATION = 0x0006
STUN_DATA_INDICATION = 0x0007

# TURN Attributes
TURN_CHANNEL_BIND_REQUEST = 0x0009
TURN_CHANNEL_BIND_RESPONSE = 0x0109
TURN_CONNECT_REQUEST = 0x000A
TURN_CONNECT_RESPONSE = 0x010A
TURN_CONNECTION_BIND_REQUEST = 0x000B
TURN_CONNECTION_BIND_RESPONSE = 0x010B

@dataclass
class STUNMessage:
    """STUN message structure"""
    message_type: int
    transaction_id: bytes
    attributes: Dict[int, bytes]

@dataclass
class ICECredentials:
    """ICE server credentials"""
    username: str
    password: str
    expires_at: float

@dataclass
class TURNAllocation:
    """TURN allocation state"""
    client_addr: Tuple[str, int]
    relayed_addr: Tuple[str, int]
    permissions: Dict[Tuple[str, int], float]  # client_addr -> expiration
    channels: Dict[int, Tuple[str, int]]      # channel -> peer_addr
    created_at: float
    expires_at: float

class SelfHostedSTUNTURNServer:
    """
    Self-hosted STUN/TURN server for WebRTC peer-to-peer connections
    Handles ICE credentials and relay functionality without external services
    """

    def __init__(self, host: str = "0.0.0.0", stun_port: int = 3478, turn_port: int = 3479):
        self.host = host
        self.stun_port = stun_port
        self.turn_port = turn_port

        # ICE credentials management
        self.credentials: Dict[str, ICECredentials] = {}
        self.realm = "self-hosted-webrtc.example.com"

        # TURN allocations
        self.allocations: Dict[Tuple[str, int], TURNAllocation] = {}

        # Server sockets
        self.stun_socket: Optional[socket.socket] = None
        self.turn_socket: Optional[socket.socket] = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Relay address pool (for TURN functionality)
        self.relay_addresses: List[Tuple[str, int]] = []
        self._initialize_relay_addresses()

    def _initialize_relay_addresses(self):
        """Initialize available relay addresses for TURN allocations"""
        # For demonstration, use localhost addresses
        # In production, use actual public IP addresses
        self.relay_addresses = [
            ("127.0.0.1", 50000 + i) for i in range(100)
        ]

    def generate_ice_credentials(self, username: Optional[str] = None) -> ICECredentials:
        """Generate ICE server credentials for WebRTC clients"""
        if username is None:
            username = f"peer_{secrets.token_hex(8)}"

        # Generate a secure password
        password = secrets.token_urlsafe(32)

        # Credentials expire in 1 hour
        expires_at = time.time() + 3600

        credentials = ICECredentials(
            username=username,
            password=password,
            expires_at=expires_at
        )

        self.credentials[username] = credentials
        self.logger.info(f"Generated ICE credentials for user: {username}")

        return credentials

    def get_ice_servers_config(self) -> Dict:
        """Get ICE server configuration for WebRTC clients"""
        return {
            "iceServers": [
                {
                    "urls": [
                        f"stun:{self.host}:{self.stun_port}",
                        f"turn:{self.host}:{self.turn_port}"
                    ],
                    "username": "peer_default",
                    "credential": "default_password_2024"
                }
            ]
        }

    def _parse_stun_message(self, data: bytes) -> Optional[STUNMessage]:
        """Parse STUN message from raw bytes"""
        if len(data) < STUN_HEADER_SIZE:
            return None

        message_type, length = struct.unpack("!HH", data[:4])
        magic_cookie, transaction_id = struct.unpack("!I12s", data[4:20])

        if magic_cookie != STUN_MAGIC_COOKIE:
            return None

        # Parse attributes
        attributes = {}
        offset = STUN_HEADER_SIZE

        while offset < len(data):
            if offset + 4 > len(data):
                break

            attr_type, attr_length = struct.unpack("!HH", data[offset:offset+4])
            offset += 4

            if offset + attr_length > len(data):
                break

            attr_value = data[offset:offset + attr_length]
            attributes[attr_type] = attr_value
            offset += attr_length

            # Handle padding
            while offset % 4 != 0:
                offset += 1

        return STUNMessage(message_type, transaction_id, attributes)

    def _create_stun_response(self, message: STUNMessage, attributes: Dict[int, bytes]) -> bytes:
        """Create STUN response message"""
        # Calculate total length
        attr_length = sum(len(value) + (4 - len(value) % 4) % 4 + 4 for value in attributes.values())
        total_length = attr_length

        # Create response
        response = bytearray()
        response.extend(struct.pack("!HH", message.message_type | 0x0100, total_length))
        response.extend(struct.pack("!I", STUN_MAGIC_COOKIE))
        response.extend(message.transaction_id)

        # Add attributes
        for attr_type, attr_value in attributes.items():
            response.extend(struct.pack("!HH", attr_type, len(attr_value)))
            response.extend(attr_value)

            # Add padding
            padding = (4 - len(attr_value) % 4) % 4
            response.extend(b'\x00' * padding)

        return bytes(response)

    def _handle_stun_binding_request(self, message: STUNMessage, client_addr: Tuple[str, int]) -> bytes:
        """Handle STUN binding request"""
        # XOR the client's IP and port for privacy
        ip_bytes = socket.inet_aton(client_addr[0])
        port_bytes = struct.pack("!H", client_addr[1])

        # XOR with magic cookie
        xored_ip = bytes(a ^ b for a, b in zip(ip_bytes, struct.pack("!I", STUN_MAGIC_COOKIE)[:4]))
        xored_port = bytes(a ^ b for a, b in zip(port_bytes, struct.pack("!H", STUN_MAGIC_COOKIE >> 16)))

        # Create XOR-MAPPED-ADDRESS attribute (0x0020)
        xor_addr_attr = b'\x00\x01' + xored_port + xored_ip

        response_attrs = {
            0x0020: xor_addr_attr,  # XOR-MAPPED-ADDRESS
            0x8028: b'\x00\x04\x00\x00\x00\x00'  # FINGERPRINT (placeholder)
        }

        return self._create_stun_response(message, response_attrs)

    def _handle_turn_allocate_request(self, message: STUNMessage, client_addr: Tuple[str, int]) -> bytes:
        """Handle TURN allocate request"""
        # Create new allocation
        if self.relay_addresses:
            relayed_addr = self.relay_addresses.pop(0)
        else:
            # Return error if no relay addresses available
            return self._create_stun_response(message, {0x0112: b'\x00\x00'})  # 420 Insufficient Capacity

        allocation = TURNAllocation(
            client_addr=client_addr,
            relayed_addr=relayed_addr,
            permissions={},
            channels={},
            created_at=time.time(),
            expires_at=time.time() + 600  # 10 minutes
        )

        self.allocations[client_addr] = allocation

        # Create RELAYED-ADDRESS attribute (0x0016)
        ip_bytes = socket.inet_aton(relayed_addr[0])
        relayed_attr = b'\x00\x01' + struct.pack("!H", relayed_addr[1]) + ip_bytes

        # Create LIFETIME attribute (0x000D)
        lifetime_attr = struct.pack("!I", 600)

        response_attrs = {
            0x0016: relayed_attr,  # RELAYED-ADDRESS
            0x000D: lifetime_attr   # LIFETIME
        }

        self.logger.info(f"Created TURN allocation for {client_addr} -> {relayed_addr}")
        return self._create_stun_response(message, response_attrs)

    def _handle_stun_message(self, data: bytes, client_addr: Tuple[str, int]) -> Optional[bytes]:
        """Handle incoming STUN message"""
        message = self._parse_stun_message(data)
        if not message:
            return None

        if message.message_type == STUN_BINDING_REQUEST:
            return self._handle_stun_binding_request(message, client_addr)
        elif message.message_type == STUN_ALLOCATE_REQUEST:
            return self._handle_turn_allocate_request(message, client_addr)

        return None

    async def _handle_stun_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle STUN client connection"""
        client_addr = writer.get_extra_info('peername')
        self.logger.info(f"STUN client connected: {client_addr}")

        try:
            while True:
                data = await reader.read(2048)
                if not data:
                    break

                response = self._handle_stun_message(data, client_addr)
                if response:
                    writer.write(response)
                    await writer.drain()

        except Exception as e:
            self.logger.error(f"STUN client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_turn_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle TURN client connection"""
        client_addr = writer.get_extra_info('peername')
        self.logger.info(f"TURN client connected: {client_addr}")

        try:
            while True:
                data = await reader.read(2048)
                if not data:
                    break

                response = self._handle_stun_message(data, client_addr)
                if response:
                    writer.write(response)
                    await writer.drain()

        except Exception as e:
            self.logger.error(f"TURN client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start_servers(self):
        """Start STUN and TURN servers"""
        # Start STUN server
        stun_server = await asyncio.start_server(
            self._handle_stun_client,
            self.host,
            self.stun_port
        )

        # Start TURN server
        turn_server = await asyncio.start_server(
            self._handle_turn_client,
            self.host,
            self.turn_port
        )

        self.logger.info(f"STUN server started on {self.host}:{self.stun_port}")
        self.logger.info(f"TURN server started on {self.host}:{self.turn_port}")

        # Generate default credentials
        self.generate_ice_credentials("peer_default")

        try:
            # Run both servers concurrently
            await asyncio.gather(
                stun_server.serve_forever(),
                turn_server.serve_forever()
            )
        except KeyboardInterrupt:
            self.logger.info("Shutting down servers...")
        finally:
            stun_server.close()
            turn_server.close()
            await stun_server.wait_closed()
            await turn_server.wait_closed()

    def cleanup_expired_allocations(self):
        """Clean up expired TURN allocations"""
        current_time = time.time()
        expired_clients = []

        for client_addr, allocation in self.allocations.items():
            if current_time > allocation.expires_at:
                expired_clients.append(client_addr)
                # Return relay address to pool
                self.relay_addresses.append(allocation.relayed_addr)

        for client_addr in expired_clients:
            del self.allocations[client_addr]

        if expired_clients:
            self.logger.info(f"Cleaned up {len(expired_clients)} expired TURN allocations")

async def main():
    """Main server entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    server = SelfHostedSTUNTURNServer()
    await server.start_servers()

if __name__ == "__main__":
    asyncio.run(main())
