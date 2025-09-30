#!/usr/bin/env python3
"""
Unified WebRTC P2P Server
Combines STUN/TURN and signaling servers for complete self-hosted P2P communication
"""

import asyncio
import logging
from webrtc_stun_turn_server import SelfHostedSTUNTURNServer
from webrtc_signaling_server import WebRTCSignalingServer

class WebRTCP2PServer:
    """
    Complete self-hosted WebRTC peer-to-peer server
    Provides STUN/TURN servers and signaling without external dependencies
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 stun_port: int = 3478,
                 turn_port: int = 3479,
                 signaling_port: int = 8080):
        self.host = host
        self.stun_port = stun_port
        self.turn_port = turn_port
        self.signaling_port = signaling_port

        # Initialize servers
        self.stun_turn_server = SelfHostedSTUNTURNServer(host, stun_port, turn_port)
        self.signaling_server = WebRTCSignalingServer(host, signaling_port)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def get_server_info(self) -> dict:
        """Get server configuration information"""
        return {
            "stun_server": f"{self.host}:{self.stun_port}",
            "turn_server": f"{self.host}:{self.turn_port}",
            "signaling_server": f"ws://{self.host}:{self.signaling_port}",
            "ice_servers": self.stun_turn_server.get_ice_servers_config()
        }

    async def start_servers(self):
        """Start all WebRTC servers"""
        self.logger.info("Starting WebRTC P2P Server...")
        self.logger.info(f"STUN Server: {self.host}:{self.stun_port}")
        self.logger.info(f"TURN Server: {self.host}:{self.turn_port}")
        self.logger.info(f"Signaling Server: ws://{self.host}:{self.signaling_port}")

        try:
            # Start both servers concurrently
            await asyncio.gather(
                self.stun_turn_server.start_servers(),
                self.signaling_server.start_server()
            )
        except KeyboardInterrupt:
            self.logger.info("Shutting down WebRTC P2P Server...")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise

    def generate_room_credentials(self, room_id: str) -> dict:
        """Generate credentials for a specific room"""
        credentials = self.stun_turn_server.generate_ice_credentials(f"room_{room_id}")
        return {
            "room_id": room_id,
            "ice_credentials": {
                "username": credentials.username,
                "password": credentials.password
            },
            "servers": self.get_server_info()
        }

async def main():
    """Main server entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    server = WebRTCP2PServer()
    await server.start_servers()

if __name__ == "__main__":
    asyncio.run(main())
