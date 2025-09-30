#!/usr/bin/env python3
"""
WebRTC Signaling Server for Peer-to-Peer Connections
Handles offer/answer exchange and peer discovery without external signaling
"""

import asyncio
import json
import logging
import secrets
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

@dataclass
class PeerInfo:
    """Information about a connected peer"""
    peer_id: str
    room_id: str
    connected_at: float
    websocket: WebSocket
    is_host: bool = False

@dataclass
class SignalingMessage:
    """WebRTC signaling message structure"""
    type: str
    from_peer: str
    to_peer: Optional[str] = None
    room_id: Optional[str] = None
    data: Optional[dict] = None
    timestamp: Optional[float] = None

@dataclass
class Room:
    """WebRTC room for peer connections"""
    room_id: str
    peers: Dict[str, PeerInfo]
    created_at: float
    max_peers: int = 10
    password: Optional[str] = None

class WebRTCSignalingServer:
    """
    Self-hosted WebRTC signaling server for peer-to-peer connections
    Manages rooms, peer discovery, and signaling message exchange
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port

        # Room management
        self.rooms: Dict[str, Room] = {}

        # Connected peers
        self.peers: Dict[str, PeerInfo] = {}

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # FastAPI app
        self.app = FastAPI(title="WebRTC Signaling Server", version="1.0.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            """Health check endpoint"""
            return {
                "status": "running",
                "server": "WebRTC Signaling Server",
                "active_rooms": len(self.rooms),
                "connected_peers": len(self.peers)
            }

        @self.app.get("/rooms")
        async def list_rooms():
            """List available rooms"""
            return {
                "rooms": [
                    {
                        "room_id": room.room_id,
                        "peer_count": len(room.peers),
                        "max_peers": room.max_peers,
                        "created_at": room.created_at,
                        "has_password": room.password is not None
                    }
                    for room in self.rooms.values()
                ]
            }

        @self.app.post("/rooms")
        async def create_room(room_data: dict):
            """Create a new room"""
            room_id = room_data.get("room_id") or secrets.token_urlsafe(8)
            max_peers = min(room_data.get("max_peers", 10), 50)  # Max 50 peers
            password = room_data.get("password")

            if room_id in self.rooms:
                raise HTTPException(status_code=400, detail="Room already exists")

            room = Room(
                room_id=room_id,
                peers={},
                created_at=time.time(),
                max_peers=max_peers,
                password=password
            )

            self.rooms[room_id] = room
            self.logger.info(f"Created room: {room_id} (max_peers: {max_peers})")

            return {"room_id": room_id, "status": "created"}

        @self.app.websocket("/ws/{peer_id}")
        async def websocket_endpoint(websocket: WebSocket, peer_id: str):
            """WebRTC signaling WebSocket endpoint"""
            await websocket.accept()

            try:
                # Wait for join message
                join_message = await websocket.receive_json()

                if join_message.get("type") != "join":
                    await websocket.send_json({
                        "type": "error",
                        "message": "Expected join message first"
                    })
                    return

                room_id = join_message.get("room_id")
                password = join_message.get("password")

                # Validate room
                if room_id not in self.rooms:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Room does not exist"
                    })
                    return

                room = self.rooms[room_id]

                # Check password
                if room.password and room.password != password:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid room password"
                    })
                    return

                # Check room capacity
                if len(room.peers) >= room.max_peers:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Room is full"
                    })
                    return

                # Add peer to room
                peer_info = PeerInfo(
                    peer_id=peer_id,
                    room_id=room_id,
                    connected_at=time.time(),
                    websocket=websocket,
                    is_host=len(room.peers) == 0  # First peer is host
                )

                room.peers[peer_id] = peer_info
                self.peers[peer_id] = peer_info

                # Send success message
                await websocket.send_json({
                    "type": "joined",
                    "room_id": room_id,
                    "peer_id": peer_id,
                    "is_host": peer_info.is_host,
                    "peer_count": len(room.peers)
                })

                # Notify other peers about new peer
                await self._notify_room_peers(room_id, peer_id, {
                    "type": "peer_joined",
                    "peer_id": peer_id,
                    "is_host": peer_info.is_host
                }, exclude_peer=peer_id)

                # Send list of existing peers to new peer
                existing_peers = [
                    {
                        "peer_id": p.peer_id,
                        "is_host": p.is_host
                    }
                    for p in room.peers.values()
                    if p.peer_id != peer_id
                ]

                await websocket.send_json({
                    "type": "room_state",
                    "peers": existing_peers
                })

                self.logger.info(f"Peer {peer_id} joined room {room_id}")

                # Handle signaling messages
                await self._handle_signaling_messages(peer_id, websocket, room_id)

            except WebSocketDisconnect:
                await self._handle_peer_disconnect(peer_id)
            except Exception as e:
                self.logger.error(f"WebSocket error for peer {peer_id}: {e}")
                await self._handle_peer_disconnect(peer_id)

    async def _handle_signaling_messages(self, peer_id: str, websocket: WebSocket, room_id: str):
        """Handle WebRTC signaling messages"""
        while True:
            try:
                message = await websocket.receive_json()
                message_type = message.get("type")

                if message_type == "offer":
                    await self._handle_offer(peer_id, message, room_id)
                elif message_type == "answer":
                    await self._handle_answer(peer_id, message, room_id)
                elif message_type == "ice_candidate":
                    await self._handle_ice_candidate(peer_id, message, room_id)
                elif message_type == "leave":
                    break
                else:
                    self.logger.warning(f"Unknown message type: {message_type}")

            except Exception as e:
                self.logger.error(f"Error handling message from {peer_id}: {e}")
                break

    async def _handle_offer(self, from_peer: str, message: dict, room_id: str):
        """Handle WebRTC offer"""
        to_peer = message.get("to_peer")
        if not to_peer:
            return

        room = self.rooms.get(room_id)
        if not room or to_peer not in room.peers:
            return

        # Forward offer to target peer
        offer_message = {
            "type": "offer",
            "from_peer": from_peer,
            "to_peer": to_peer,
            "data": message.get("data"),
            "timestamp": time.time()
        }

        await room.peers[to_peer].websocket.send_json(offer_message)

    async def _handle_answer(self, from_peer: str, message: dict, room_id: str):
        """Handle WebRTC answer"""
        to_peer = message.get("to_peer")
        if not to_peer:
            return

        room = self.rooms.get(room_id)
        if not room or to_peer not in room.peers:
            return

        # Forward answer to target peer
        answer_message = {
            "type": "answer",
            "from_peer": from_peer,
            "to_peer": to_peer,
            "data": message.get("data"),
            "timestamp": time.time()
        }

        await room.peers[to_peer].websocket.send_json(answer_message)

    async def _handle_ice_candidate(self, from_peer: str, message: dict, room_id: str):
        """Handle ICE candidate"""
        to_peer = message.get("to_peer")
        if not to_peer:
            return

        room = self.rooms.get(room_id)
        if not room or to_peer not in room.peers:
            return

        # Forward ICE candidate to target peer
        candidate_message = {
            "type": "ice_candidate",
            "from_peer": from_peer,
            "to_peer": to_peer,
            "data": message.get("data"),
            "timestamp": time.time()
        }

        await room.peers[to_peer].websocket.send_json(candidate_message)

    async def _notify_room_peers(self, room_id: str, from_peer: str, message: dict, exclude_peer: Optional[str] = None):
        """Notify all peers in a room"""
        room = self.rooms.get(room_id)
        if not room:
            return

        for peer_id, peer_info in room.peers.items():
            if peer_id != exclude_peer:
                try:
                    await peer_info.websocket.send_json(message)
                except Exception as e:
                    self.logger.error(f"Failed to notify peer {peer_id}: {e}")

    async def _handle_peer_disconnect(self, peer_id: str):
        """Handle peer disconnection"""
        if peer_id not in self.peers:
            return

        peer_info = self.peers[peer_id]
        room_id = peer_info.room_id

        # Remove from room
        if room_id in self.rooms:
            room = self.rooms[room_id]
            if peer_id in room.peers:
                del room.peers[peer_id]

            # If room is empty, remove it
            if not room.peers:
                del self.rooms[room_id]
                self.logger.info(f"Removed empty room: {room_id}")
            else:
                # Notify remaining peers
                await self._notify_room_peers(room_id, peer_id, {
                    "type": "peer_left",
                    "peer_id": peer_id
                })

        # Remove from peers
        del self.peers[peer_id]

        self.logger.info(f"Peer {peer_id} disconnected from room {room_id}")

    def cleanup_expired_rooms(self):
        """Clean up empty or expired rooms"""
        current_time = time.time()
        expired_rooms = []

        for room_id, room in self.rooms.items():
            # Remove rooms that have been empty for more than 1 hour
            if not room.peers and (current_time - room.created_at) > 3600:
                expired_rooms.append(room_id)

        for room_id in expired_rooms:
            del self.rooms[room_id]
            self.logger.info(f"Cleaned up expired room: {room_id}")

    async def start_server(self):
        """Start the signaling server"""
        self.logger.info(f"Starting WebRTC Signaling Server on {self.host}:{self.port}")

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

        # Start FastAPI server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _cleanup_task(self):
        """Periodic cleanup task"""
        while True:
            await asyncio.sleep(300)  # Clean up every 5 minutes
            self.cleanup_expired_rooms()

async def main():
    """Main server entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    server = WebRTCSignalingServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())
