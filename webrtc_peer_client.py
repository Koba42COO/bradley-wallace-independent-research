#!/usr/bin/env python3
"""
WebRTC Peer-to-Peer Client
Connects to self-hosted STUN/TURN and signaling servers for P2P communication
"""

import asyncio
import json
import logging
import secrets
import time
from typing import Dict, Optional, Callable, Any
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import websockets
from dataclasses import dataclass

@dataclass
class PeerConnection:
    """WebRTC peer connection state"""
    peer_id: str
    pc: RTCPeerConnection
    data_channel: Optional[Any] = None
    is_connected: bool = False

class WebRTCPeerClient:
    """
    WebRTC peer-to-peer client using self-hosted servers
    Handles signaling, ICE negotiation, and data channels
    """

    def __init__(self,
                 signaling_url: str = "ws://localhost:8080/ws",
                 stun_server: str = "stun:localhost:3478",
                 turn_server: str = "turn:localhost:3479"):
        self.signaling_url = signaling_url
        self.stun_server = stun_server
        self.turn_server = turn_server

        # Peer connections
        self.peer_connections: Dict[str, PeerConnection] = {}
        self.peer_id = f"peer_{secrets.token_hex(8)}"

        # Signaling websocket
        self.signaling_ws: Optional[websockets.WebSocketServerProtocol] = None

        # Room state
        self.room_id: Optional[str] = None
        self.is_host = False

        # Event callbacks
        self.on_peer_joined: Optional[Callable[[str], None]] = None
        self.on_peer_left: Optional[Callable[[str], None]] = None
        self.on_message_received: Optional[Callable[[str, Any], None]] = None
        self.on_connection_established: Optional[Callable[[str], None]] = None
        self.on_connection_failed: Optional[Callable[[str], None]] = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def join_room(self, room_id: str, password: Optional[str] = None):
        """Join a WebRTC room"""
        self.room_id = room_id

        try:
            # Connect to signaling server
            uri = f"{self.signaling_url}/{self.peer_id}"
            async with websockets.connect(uri) as websocket:
                self.signaling_ws = websocket

                # Send join message
                join_message = {
                    "type": "join",
                    "room_id": room_id,
                    "password": password
                }
                await websocket.send(json.dumps(join_message))

                # Wait for join confirmation
                response = await websocket.recv()
                join_response = json.loads(response)

                if join_response.get("type") == "error":
                    raise Exception(join_response.get("message", "Failed to join room"))

                self.is_host = join_response.get("is_host", False)
                self.logger.info(f"Joined room {room_id} as {'host' if self.is_host else 'peer'}")

                # Handle signaling messages
                await self._handle_signaling_messages(websocket)

        except Exception as e:
            self.logger.error(f"Failed to join room {room_id}: {e}")
            raise

    async def create_peer_connection(self, peer_id: str) -> PeerConnection:
        """Create a new WebRTC peer connection"""
        pc = RTCPeerConnection()

        # Configure ICE servers
        pc.iceServers = [
            {"urls": [self.stun_server]},
            {
                "urls": [self.turn_server],
                "username": "peer_default",
                "credential": "default_password_2024"
            }
        ]

        # Create data channel for messaging
        data_channel = pc.createDataChannel("chat")
        data_channel.onmessage = lambda event: self._handle_data_channel_message(peer_id, event.data)
        data_channel.onopen = lambda: self._handle_data_channel_open(peer_id)
        data_channel.onclose = lambda: self._handle_data_channel_close(peer_id)

        # Set up ICE event handlers
        @pc.on("icecandidate")
        async def on_icecandidate(event):
            if event.candidate:
                await self._send_signaling_message({
                    "type": "ice_candidate",
                    "to_peer": peer_id,
                    "data": {
                        "candidate": event.candidate.candidate,
                        "sdpMid": event.candidate.sdpMid,
                        "sdpMLineIndex": event.candidate.sdpMLineIndex
                    }
                })

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.logger.info(f"Connection state with {peer_id}: {pc.connectionState}")
            if pc.connectionState == "connected":
                peer_conn = self.peer_connections.get(peer_id)
                if peer_conn:
                    peer_conn.is_connected = True
                    if self.on_connection_established:
                        self.on_connection_established(peer_id)
            elif pc.connectionState in ["failed", "closed", "disconnected"]:
                if self.on_connection_failed:
                    self.on_connection_failed(peer_id)

        peer_conn = PeerConnection(
            peer_id=peer_id,
            pc=pc,
            data_channel=data_channel
        )

        self.peer_connections[peer_id] = peer_conn
        return peer_conn

    async def initiate_connection(self, peer_id: str):
        """Initiate WebRTC connection with a peer"""
        peer_conn = await self.create_peer_connection(peer_id)

        # Create offer
        offer = await peer_conn.pc.createOffer()
        await peer_conn.pc.setLocalDescription(offer)

        # Send offer through signaling
        await self._send_signaling_message({
            "type": "offer",
            "to_peer": peer_id,
            "data": {
                "sdp": offer.sdp,
                "type": offer.type
            }
        })

    async def _handle_signaling_messages(self, websocket):
        """Handle incoming signaling messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "room_state":
                    # Handle initial room state
                    peers = data.get("peers", [])
                    for peer_info in peers:
                        peer_id = peer_info["peer_id"]
                        if self.is_host:
                            # Host initiates connections
                            await self.initiate_connection(peer_id)

                elif message_type == "peer_joined":
                    peer_id = data.get("peer_id")
                    self.logger.info(f"Peer joined: {peer_id}")
                    if self.on_peer_joined:
                        self.on_peer_joined(peer_id)

                    if not self.is_host:
                        # Non-host peers wait for offers
                        pass

                elif message_type == "peer_left":
                    peer_id = data.get("peer_id")
                    self.logger.info(f"Peer left: {peer_id}")
                    if peer_id in self.peer_connections:
                        await self.peer_connections[peer_id].pc.close()
                        del self.peer_connections[peer_id]
                    if self.on_peer_left:
                        self.on_peer_left(peer_id)

                elif message_type == "offer":
                    from_peer = data.get("from_peer")
                    offer_data = data.get("data")
                    await self._handle_offer(from_peer, offer_data)

                elif message_type == "answer":
                    from_peer = data.get("from_peer")
                    answer_data = data.get("data")
                    await self._handle_answer(from_peer, answer_data)

                elif message_type == "ice_candidate":
                    from_peer = data.get("from_peer")
                    candidate_data = data.get("data")
                    await self._handle_ice_candidate(from_peer, candidate_data)

            except Exception as e:
                self.logger.error(f"Error handling signaling message: {e}")

    async def _handle_offer(self, from_peer: str, offer_data: dict):
        """Handle incoming WebRTC offer"""
        peer_conn = await self.create_peer_connection(from_peer)

        # Set remote description
        offer = RTCSessionDescription(
            sdp=offer_data["sdp"],
            type=offer_data["type"]
        )
        await peer_conn.pc.setRemoteDescription(offer)

        # Create answer
        answer = await peer_conn.pc.createAnswer()
        await peer_conn.pc.setLocalDescription(answer)

        # Send answer through signaling
        await self._send_signaling_message({
            "type": "answer",
            "to_peer": from_peer,
            "data": {
                "sdp": answer.sdp,
                "type": answer.type
            }
        })

    async def _handle_answer(self, from_peer: str, answer_data: dict):
        """Handle incoming WebRTC answer"""
        peer_conn = self.peer_connections.get(from_peer)
        if not peer_conn:
            return

        answer = RTCSessionDescription(
            sdp=answer_data["sdp"],
            type=answer_data["type"]
        )
        await peer_conn.pc.setRemoteDescription(answer)

    async def _handle_ice_candidate(self, from_peer: str, candidate_data: dict):
        """Handle incoming ICE candidate"""
        peer_conn = self.peer_connections.get(from_peer)
        if not peer_conn:
            return

        candidate = RTCIceCandidate(
            candidate=candidate_data["candidate"],
            sdpMid=candidate_data["sdpMid"],
            sdpMLineIndex=candidate_data["sdpMLineIndex"]
        )
        await peer_conn.pc.addIceCandidate(candidate)

    async def _send_signaling_message(self, message: dict):
        """Send message through signaling server"""
        if self.signaling_ws:
            await self.signaling_ws.send(json.dumps(message))

    def _handle_data_channel_message(self, peer_id: str, message: str):
        """Handle incoming data channel message"""
        self.logger.info(f"Message from {peer_id}: {message}")
        if self.on_message_received:
            try:
                data = json.loads(message)
                self.on_message_received(peer_id, data)
            except json.JSONDecodeError:
                self.on_message_received(peer_id, message)

    def _handle_data_channel_open(self, peer_id: str):
        """Handle data channel open"""
        self.logger.info(f"Data channel opened with {peer_id}")
        peer_conn = self.peer_connections.get(peer_id)
        if peer_conn:
            peer_conn.is_connected = True

    def _handle_data_channel_close(self, peer_id: str):
        """Handle data channel close"""
        self.logger.info(f"Data channel closed with {peer_id}")

    async def send_message(self, peer_id: str, message: Any):
        """Send message to a peer through data channel"""
        peer_conn = self.peer_connections.get(peer_id)
        if not peer_conn or not peer_conn.data_channel:
            raise Exception(f"No data channel available for peer {peer_id}")

        if isinstance(message, (dict, list)):
            message = json.dumps(message)

        peer_conn.data_channel.send(message)

    async def broadcast_message(self, message: Any):
        """Broadcast message to all connected peers"""
        for peer_id, peer_conn in self.peer_connections.items():
            if peer_conn.is_connected and peer_conn.data_channel:
                try:
                    await self.send_message(peer_id, message)
                except Exception as e:
                    self.logger.error(f"Failed to send message to {peer_id}: {e}")

    async def leave_room(self):
        """Leave the current room"""
        if self.signaling_ws:
            try:
                await self._send_signaling_message({"type": "leave"})
            except Exception as e:
                self.logger.error(f"Error sending leave message: {e}")

        # Close all peer connections
        for peer_conn in self.peer_connections.values():
            await peer_conn.pc.close()

        self.peer_connections.clear()
        self.room_id = None
        self.is_host = False

    def get_connected_peers(self) -> list:
        """Get list of currently connected peers"""
        return [
            peer_id for peer_id, peer_conn in self.peer_connections.items()
            if peer_conn.is_connected
        ]

    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        stats = {
            "peer_id": self.peer_id,
            "room_id": self.room_id,
            "is_host": self.is_host,
            "total_peers": len(self.peer_connections),
            "connected_peers": len(self.get_connected_peers()),
            "connections": {}
        }

        for peer_id, peer_conn in self.peer_connections.items():
            stats["connections"][peer_id] = {
                "connected": peer_conn.is_connected,
                "connection_state": peer_conn.pc.connectionState,
                "ice_connection_state": peer_conn.pc.iceConnectionState,
                "ice_gathering_state": peer_conn.pc.iceGatheringState
            }

        return stats

# Example usage
async def demo_client():
    """Demo client usage"""
    logging.basicConfig(level=logging.INFO)

    client = WebRTCPeerClient()

    # Set up event handlers
    client.on_peer_joined = lambda peer_id: print(f"Peer joined: {peer_id}")
    client.on_peer_left = lambda peer_id: print(f"Peer left: {peer_id}")
    client.on_message_received = lambda peer_id, message: print(f"Message from {peer_id}: {message}")
    client.on_connection_established = lambda peer_id: print(f"Connected to {peer_id}")

    try:
        # Join a room
        await client.join_room("demo_room")

        # Keep running
        await asyncio.sleep(300)  # Run for 5 minutes

    except KeyboardInterrupt:
        print("Shutting down client...")
    finally:
        await client.leave_room()

if __name__ == "__main__":
    asyncio.run(demo_client())
