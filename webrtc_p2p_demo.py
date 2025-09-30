#!/usr/bin/env python3
"""
WebRTC P2P Demo Application
Demonstrates self-hosted peer-to-peer communication without external relays
"""

import asyncio
import logging
import json
import time
from webrtc_p2p_server import WebRTCP2PServer
from webrtc_peer_client import WebRTCPeerClient

class P2PDemoApp:
    """
    Demo application showing WebRTC peer-to-peer communication
    using self-hosted STUN/TURN and signaling servers
    """

    def __init__(self):
        self.server = WebRTCP2PServer()
        self.clients = []

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def create_demo_room(self, room_id: str = "demo_room") -> dict:
        """Create a demo room and return connection info"""
        room_credentials = self.server.generate_room_credentials(room_id)

        self.logger.info(f"Created demo room: {room_id}")
        self.logger.info(f"ICE Credentials: {room_credentials['ice_credentials']}")

        return room_credentials

    async def create_demo_client(self, room_id: str, client_name: str) -> WebRTCPeerClient:
        """Create a demo client for the room"""
        # Get server info
        server_info = self.server.get_server_info()

        # Create client
        client = WebRTCPeerClient(
            signaling_url=server_info["signaling_server"] + "/ws",
            stun_server=f"stun:{server_info['stun_server']}",
            turn_server=f"turn:{server_info['turn_server']}"
        )

        # Set up event handlers
        client.on_peer_joined = lambda peer_id: self.logger.info(f"[{client_name}] Peer joined: {peer_id}")
        client.on_peer_left = lambda peer_id: self.logger.info(f"[{client_name}] Peer left: {peer_id}")
        client.on_message_received = lambda peer_id, message: self._handle_message(client_name, peer_id, message)
        client.on_connection_established = lambda peer_id: self.logger.info(f"[{client_name}] Connected to {peer_id}")
        client.on_connection_failed = lambda peer_id: self.logger.warning(f"[{client_name}] Failed to connect to {peer_id}")

        self.clients.append(client)
        return client

    def _handle_message(self, client_name: str, from_peer: str, message: any):
        """Handle incoming messages"""
        self.logger.info(f"[{client_name}] Message from {from_peer}: {message}")

        # Echo messages back (simple demo)
        if isinstance(message, dict) and message.get("type") == "echo":
            asyncio.create_task(self._echo_message(client_name, from_peer, message))

    async def _echo_message(self, client_name: str, to_peer: str, original_message: dict):
        """Echo a message back to sender"""
        client = next((c for c in self.clients if c.peer_id.startswith(client_name)), None)
        if client:
            echo_response = {
                "type": "echo_response",
                "original": original_message,
                "timestamp": time.time()
            }
            await client.send_message(to_peer, echo_response)

    async def run_demo(self):
        """Run the complete P2P demo"""
        self.logger.info("🚀 Starting WebRTC P2P Demo")
        self.logger.info("This demo shows self-hosted peer-to-peer communication")
        self.logger.info("No external STUN/TURN servers or signaling services required!")

        # Start server in background
        server_task = asyncio.create_task(self.server.start_servers())

        # Wait a moment for servers to start
        await asyncio.sleep(2)

        # Create demo room
        room_info = await self.create_demo_room("tangtalk_demo")
        self.logger.info(f"📡 Room Info: {json.dumps(room_info, indent=2)}")

        # Create multiple demo clients
        client_tasks = []

        for i in range(3):
            client_name = f"client_{i+1}"
            client = await self.create_demo_client("tangtalk_demo", client_name)

            # Create task for each client
            task = asyncio.create_task(self._run_client_demo(client, client_name, i))
            client_tasks.append(task)

        # Let demo run for a while
        self.logger.info("🎭 Demo running... Clients will join and exchange messages")

        try:
            await asyncio.wait_for(asyncio.gather(*client_tasks), timeout=60)
        except asyncio.TimeoutError:
            self.logger.info("Demo completed successfully!")

        # Cleanup
        self.logger.info("🧹 Cleaning up demo...")
        for client in self.clients:
            await client.leave_room()

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    async def _run_client_demo(self, client: WebRTCPeerClient, client_name: str, delay: int):
        """Run demo scenario for a client"""
        # Stagger client joins
        await asyncio.sleep(delay * 2)

        try:
            self.logger.info(f"[{client_name}] Joining room...")
            await client.join_room("tangtalk_demo")

            # Wait for connections
            await asyncio.sleep(5)

            # Send some demo messages
            connected_peers = client.get_connected_peers()
            if connected_peers:
                # Send hello message to first connected peer
                hello_message = {
                    "type": "hello",
                    "from": client_name,
                    "message": f"Hello from {client_name}!",
                    "timestamp": time.time()
                }
                await client.send_message(connected_peers[0], hello_message)

                # Send echo test
                await asyncio.sleep(2)
                echo_message = {
                    "type": "echo",
                    "message": f"Echo test from {client_name}",
                    "timestamp": time.time()
                }
                await client.send_message(connected_peers[0], echo_message)

            # Show connection stats
            stats = client.get_connection_stats()
            self.logger.info(f"[{client_name}] Connection stats: {json.dumps(stats, indent=2)}")

            # Keep client alive
            await asyncio.sleep(50)

        except Exception as e:
            self.logger.error(f"[{client_name}] Demo error: {e}")

async def run_interactive_demo():
    """Run an interactive demo where users can choose what to do"""
    logging.basicConfig(level=logging.INFO)
    demo = P2PDemoApp()

    print("🎯 WebRTC Self-Hosted P2P Demo")
    print("=" * 50)
    print("1. Run automated demo (3 clients)")
    print("2. Start server only (for manual testing)")
    print("3. Show server configuration")
    print("4. Exit")

    while True:
        try:
            choice = input("\nChoose option (1-4): ").strip()

            if choice == "1":
                await demo.run_demo()
                break
            elif choice == "2":
                print("Starting servers... (Press Ctrl+C to stop)")
                server = WebRTCP2PServer()
                await server.start_servers()
            elif choice == "3":
                server = WebRTCP2PServer()
                config = server.get_server_info()
                print("\n📋 Server Configuration:")
                print(json.dumps(config, indent=2))

                room_info = server.generate_room_credentials("test_room")
                print("\n🏠 Sample Room Credentials:")
                print(json.dumps(room_info, indent=2))
            elif choice == "4":
                break
            else:
                print("Invalid choice. Please select 1-4.")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main entry point"""
    await run_interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())
