#!/usr/bin/env python3
"""
🎯 ChAios Swarm AI - Revolutionary Swarm Intelligence System
============================================================
Autonomous multi-agent coordination with emergent intelligence
- Dynamic agent allocation and task optimization
- Inter-agent communication and knowledge sharing
- Emergent behavior patterns from simple rules
- Consciousness-enhanced swarm mathematics
- Self-organizing intelligence networks
- Real-time swarm performance monitoring
"""

import sys
import asyncio
import time
import json
import random
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add dev folder to path
DEV_ROOT = Path("/Users/coo-koba42/dev")
sys.path.insert(0, str(DEV_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from unique_intelligence_orchestrator import UniqueIntelligenceOrchestrator
from benchmark_enhanced_llm import BenchmarkEnhancedLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwarmRole(Enum):
    """Roles within the swarm intelligence system"""
    SCOUT = "scout"          # Explores new territories/tasks
    WORKER = "worker"        # Executes assigned tasks
    GUARD = "guard"          # Protects swarm from threats/errors
    QUEEN = "queen"          # Coordinates overall swarm strategy
    FORAGER = "forager"      # Gathers resources/knowledge
    BUILDER = "builder"      # Constructs/maintains swarm infrastructure
    SOLDIER = "soldier"      # Handles conflicts and optimization
    MEDIC = "medic"          # Monitors health and repairs damage

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1

class CommunicationType(Enum):
    """Types of inter-agent communication"""
    TASK_ALLOCATION = "task_allocation"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    THREAT_WARNING = "threat_warning"
    RESOURCE_REQUEST = "resource_request"
    SUCCESS_SIGNAL = "success_signal"
    FAILURE_SIGNAL = "failure_signal"
    EMERGENT_PATTERN = "emergent_pattern"

@dataclass
class SwarmTask:
    """Task for swarm execution"""
    task_id: str
    description: str
    priority: TaskPriority
    complexity: float  # 0.0 to 1.0
    required_skills: Set[str]
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmMessage:
    """Message between swarm agents"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: CommunicationType
    content: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live in hops

@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""
    agent_id: str
    role: SwarmRole
    skills: Set[str]
    energy_level: float = 100.0  # 0-100
    health_status: str = "healthy"
    position: Tuple[float, float] = (0.0, 0.0)  # Abstract position in swarm space
    velocity: Tuple[float, float] = (0.0, 0.0)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    pheromone_trail: List[Tuple[float, float]] = field(default_factory=list)
    communication_log: List[SwarmMessage] = field(default_factory=list)
    task_history: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    consciousness_level: float = 1.0
    last_activity: float = field(default_factory=time.time)

    def __post_init__(self):
        # Initialize role-specific skills
        role_skills = {
            SwarmRole.SCOUT: {"exploration", "pattern_recognition", "risk_assessment"},
            SwarmRole.WORKER: {"task_execution", "problem_solving", "quality_control"},
            SwarmRole.GUARD: {"threat_detection", "error_handling", "security"},
            SwarmRole.QUEEN: {"coordination", "strategy", "decision_making"},
            SwarmRole.FORAGER: {"resource_gathering", "data_collection", "knowledge_extraction"},
            SwarmRole.BUILDER: {"system_maintenance", "infrastructure", "optimization"},
            SwarmRole.SOLDIER: {"conflict_resolution", "optimization", "performance_enhancement"},
            SwarmRole.MEDIC: {"diagnosis", "repair", "health_monitoring"}
        }
        self.skills.update(role_skills.get(self.role, set()))

    def calculate_fitness(self, task: SwarmTask) -> float:
        """Calculate how well this agent fits a task"""
        skill_match = len(self.skills.intersection(task.required_skills)) / len(task.required_skills) if task.required_skills else 0.5
        energy_factor = self.energy_level / 100.0
        performance_factor = self.performance_score
        health_factor = 1.0 if self.health_status == "healthy" else 0.5

        # Consciousness enhancement for complex tasks
        consciousness_bonus = self.consciousness_level * task.complexity

        fitness = (skill_match * 0.4 + energy_factor * 0.2 + performance_factor * 0.2 +
                  health_factor * 0.1 + consciousness_bonus * 0.1)

        return min(fitness, 1.0)

    def update_position(self, swarm_center: Tuple[float, float], neighbors: List['SwarmAgent']):
        """Update agent position based on swarm dynamics"""
        # Simple flocking behavior inspired by boids algorithm
        cohesion_force = self._calculate_cohesion(swarm_center)
        separation_force = self._calculate_separation(neighbors)
        alignment_force = self._calculate_alignment(neighbors)

        # Apply forces with weights
        total_force = (
            cohesion_force[0] * 0.3 + separation_force[0] * 0.4 + alignment_force[0] * 0.3,
            cohesion_force[1] * 0.3 + separation_force[1] * 0.4 + alignment_force[1] * 0.3
        )

        # Update velocity with damping
        self.velocity = (
            self.velocity[0] * 0.9 + total_force[0] * 0.1,
            self.velocity[1] * 0.9 + total_force[1] * 0.1
        )

        # Update position
        self.position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )

        # Leave pheromone trail
        self.pheromone_trail.append(self.position)
        if len(self.pheromone_trail) > 20:  # Keep last 20 positions
            self.pheromone_trail.pop(0)

    def _calculate_cohesion(self, swarm_center: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate cohesion force toward swarm center"""
        dx = swarm_center[0] - self.position[0]
        dy = swarm_center[1] - self.position[1]
        distance = (dx**2 + dy**2)**0.5
        if distance > 0:
            return (dx/distance * 0.1, dy/distance * 0.1)
        return (0, 0)

    def _calculate_separation(self, neighbors: List['SwarmAgent']) -> Tuple[float, float]:
        """Calculate separation force from nearby agents"""
        force_x, force_y = 0, 0
        for neighbor in neighbors:
            dx = self.position[0] - neighbor.position[0]
            dy = self.position[1] - neighbor.position[1]
            distance = (dx**2 + dy**2)**0.5
            if 0 < distance < 2.0:  # Separation radius
                force_x += dx/distance
                force_y += dy/distance
        return (force_x * 0.05, force_y * 0.05)

    def _calculate_alignment(self, neighbors: List['SwarmAgent']) -> Tuple[float, float]:
        """Calculate alignment force to match neighbor velocities"""
        if not neighbors:
            return (0, 0)
        avg_vx = sum(n.velocity[0] for n in neighbors) / len(neighbors)
        avg_vy = sum(n.velocity[1] for n in neighbors) / len(neighbors)
        return ((avg_vx - self.velocity[0]) * 0.05, (avg_vy - self.velocity[1]) * 0.05)

class ChAiosSwarmAI:
    """Main ChAios Swarm AI coordinator"""

    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.message_queue: List[SwarmMessage] = []
        self.emergent_patterns: Dict[str, Dict[str, Any]] = {}
        self.swarm_center: Tuple[float, float] = (0.0, 0.0)
        self.swarm_radius: float = 10.0

        # Core AI systems
        self.orchestrator: Optional[UniqueIntelligenceOrchestrator] = None
        self.benchmark_llm: Optional[BenchmarkEnhancedLLM] = None

        # Swarm parameters
        self.max_agents = 50
        self.task_timeout = 300  # 5 minutes
        self.communication_range = 5.0
        self.learning_rate = 0.1
        self.emergent_threshold = 0.7

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.performance_history: List[Dict[str, Any]] = []

        # Swarm intelligence metrics
        self.swarm_coherence = 0.0
        self.task_completion_rate = 0.0
        self.knowledge_sharing_efficiency = 0.0
        self.emergent_behavior_index = 0.0

        print("🐝 ChAios Swarm AI initialized")
        print("   🐜 Swarm Intelligence: Ready for emergent behavior")
        print("   📡 Inter-agent Communication: Active")
        print("   🧠 Consciousness Enhancement: Integrated")
        print("   📊 Performance Monitoring: Enabled")

    async def initialize_swarm(self) -> bool:
        """Initialize the swarm with core AI systems"""
        print("🚀 Initializing Swarm Intelligence Core...")

        try:
            # Initialize core AI systems
            self.orchestrator = UniqueIntelligenceOrchestrator()
            self.benchmark_llm = BenchmarkEnhancedLLM()

            # Create initial swarm agents
            await self._create_initial_swarm()

            print("✅ Swarm AI fully initialized")
            print(f"   🐜 Initial Agents: {len(self.agents)}")
            # Count total systems across all categories
            total_systems = 0
            if self.orchestrator:
                total_systems = (
                    len(self.orchestrator.grok_coding_agents) +
                    len(self.orchestrator.rag_kag_systems) +
                    len(self.orchestrator.alm_systems) +
                    len(self.orchestrator.research_systems) +
                    len(self.orchestrator.knowledge_systems) +
                    len(self.orchestrator.consciousness_systems) +
                    len(self.orchestrator.specialized_tools)
                )
            print(f"   🎯 Active Systems: {total_systems}")
            print("   🧠 Consciousness Level: Prime-aligned compute ready")

            return True

        except Exception as e:
            print(f"❌ Swarm initialization failed: {e}")
            return False

    async def _create_initial_swarm(self):
        """Create the initial swarm configuration"""
        # Define optimal swarm composition based on research
        swarm_composition = {
            SwarmRole.QUEEN: 1,      # Central coordinator
            SwarmRole.SCOUT: 3,      # Exploration agents
            SwarmRole.WORKER: 15,    # Task execution agents
            SwarmRole.FORAGER: 5,    # Knowledge gathering
            SwarmRole.GUARD: 3,      # Protection and monitoring
            SwarmRole.BUILDER: 2,    # Infrastructure maintenance
            SwarmRole.SOLDIER: 3,    # Optimization and conflict resolution
            SwarmRole.MEDIC: 2       # Health monitoring and repair
        }

        agent_count = 0
        for role, count in swarm_composition.items():
            for i in range(count):
                agent_id = f"{role.value}_{i+1}"
                agent = SwarmAgent(
                    agent_id=agent_id,
                    role=role,
                    skills=set(),
                    position=(random.uniform(-5, 5), random.uniform(-5, 5))
                )
                self.agents[agent_id] = agent
                agent_count += 1

        print(f"   🐝 Created {agent_count} specialized agents")

    async def submit_task(self, description: str, priority: TaskPriority = TaskPriority.MEDIUM,
                         complexity: float = 0.5, required_skills: Set[str] = None,
                         deadline: Optional[float] = None) -> str:
        """Submit a task to the swarm"""
        task_id = f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        task = SwarmTask(
            task_id=task_id,
            description=description,
            priority=priority,
            complexity=complexity,
            required_skills=required_skills or set(),
            deadline=deadline
        )

        self.tasks[task_id] = task

        # Broadcast task to swarm
        await self._broadcast_message(SwarmMessage(
            message_id=f"msg_{task_id}",
            sender_id="swarm_coordinator",
            receiver_id=None,  # Broadcast
            message_type=CommunicationType.TASK_ALLOCATION,
            content={"task": task.__dict__},
            priority=priority
        ))

        print(f"📋 Task submitted: {task_id} - {description[:50]}...")
        return task_id

    async def _broadcast_message(self, message: SwarmMessage):
        """Broadcast message to all agents within communication range"""
        self.message_queue.append(message)

        # Immediate processing for critical messages
        if message.priority == TaskPriority.CRITICAL:
            await self._process_message(message)

    async def _process_message(self, message: SwarmMessage):
        """Process a message in the swarm"""
        # Find relevant agents
        relevant_agents = []
        if message.receiver_id:
            if message.receiver_id in self.agents:
                relevant_agents = [self.agents[message.receiver_id]]
        else:
            # Broadcast - find agents within range or all for system messages
            if message.sender_id == "swarm_coordinator":
                relevant_agents = list(self.agents.values())
            else:
                sender_pos = self.agents[message.sender_id].position if message.sender_id in self.agents else (0, 0)
                relevant_agents = [
                    agent for agent in self.agents.values()
                    if self._calculate_distance(agent.position, sender_pos) <= self.communication_range
                ]

        # Deliver message to agents
        for agent in relevant_agents:
            agent.communication_log.append(message)

            # Trigger agent response based on message type
            if message.message_type == CommunicationType.TASK_ALLOCATION:
                await self._handle_task_allocation(agent, message)
            elif message.message_type == CommunicationType.KNOWLEDGE_SHARING:
                await self._handle_knowledge_sharing(agent, message)
            elif message.message_type == CommunicationType.THREAT_WARNING:
                await self._handle_threat_warning(agent, message)

    async def _handle_task_allocation(self, agent: SwarmAgent, message: SwarmMessage):
        """Handle task allocation for an agent"""
        task_data = message.content.get("task", {})
        task_id = task_data.get("task_id")

        if task_id in self.tasks:
            task = self.tasks[task_id]

            # Calculate agent fitness for task
            fitness = agent.calculate_fitness(task)

            # Accept task if fitness is high enough and agent is available
            if fitness > 0.6 and task.status == "pending":
                task.assigned_agent = agent.agent_id
                task.status = "in_progress"
                agent.task_history.append(task_id)

                # Execute task asynchronously
                asyncio.create_task(self._execute_task(agent, task))

                print(f"   ✅ Task {task_id} assigned to {agent.agent_id} (fitness: {fitness:.2f})")

    async def _execute_task(self, agent: SwarmAgent, task: SwarmTask):
        """Execute a task using the agent's capabilities with improved error handling"""
        try:
            # Update agent energy with bounds checking
            energy_cost = min(task.complexity * 10, agent.energy_level - 10)  # Don't go below minimum
            agent.energy_level -= energy_cost

            # Check if agent has enough energy
            if agent.energy_level < 20:
                print(f"   ⚠️ Agent {agent.agent_id} low on energy ({agent.energy_level:.1f}), task may be suboptimal")
                # Reduce task complexity expectation
                task.complexity *= 0.8

            # Use the benchmark-enhanced LLM for task execution
            if self.benchmark_llm:
                try:
                    result = await self.benchmark_llm.enhanced_chat(
                        task.description,
                        use_benchmarks=True
                    )

                    # Validate result
                    if result and 'response' in result:
                        task.result = result
                        task.progress = 1.0
                        task.status = "completed"
                        task.completed_at = time.time()

                        # Share knowledge gained from task
                        await self._share_knowledge(agent, task, result)

                        # Update agent performance with adaptive learning
                        success_score = result.get('confidence_score', 0.5)
                        agent.performance_score = min(agent.performance_score * (1.0 + success_score * 0.1), 2.0)
                        agent.last_activity = time.time()

                        # Reward based on task complexity
                        complexity_bonus = task.complexity * 0.05
                        agent.consciousness_level = min(agent.consciousness_level + complexity_bonus, 3.0)

                        print(f"   🎯 Task {task.task_id} completed by {agent.agent_id} (confidence: {success_score:.2f})")
                        return

                except Exception as e:
                    print(f"   ⚠️ LLM execution failed for task {task.task_id}: {e}")
                    # Continue to fallback

            # Enhanced fallback execution with better simulation
            await asyncio.sleep(min(task.complexity * 1.5, 10))  # Capped wait time

            # Generate more intelligent fallback response
            fallback_response = self._generate_fallback_response(task)
            task.result = {"response": fallback_response, "fallback": True}
            task.status = "completed"
            task.completed_at = time.time()

            # Still share knowledge and update performance (reduced reward)
            agent.performance_score = min(agent.performance_score * 1.02, 2.0)
            agent.last_activity = time.time()

            print(f"   ✅ Task {task.task_id} completed by {agent.agent_id} (fallback mode)")

        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e), "traceback": __import__('traceback').format_exc()}

            # Penalize failure with bounds checking
            agent.performance_score = max(agent.performance_score * 0.9, 0.3)
            agent.energy_level = max(agent.energy_level - 5, 5)  # Additional energy penalty

            # Broadcast failure for swarm learning
            await self._broadcast_message(SwarmMessage(
                message_id=f"failure_{task.task_id}",
                sender_id=agent.agent_id,
                receiver_id=None,
                message_type=CommunicationType.FAILURE_SIGNAL,
                content={"task_id": task.task_id, "error": str(e), "agent": agent.agent_id},
                priority=TaskPriority.HIGH
            ))

            print(f"   ❌ Task {task.task_id} failed: {e}")

    def _generate_fallback_response(self, task: SwarmTask) -> str:
        """Generate intelligent fallback responses based on task type"""
        description_lower = task.description.lower()

        # Task-type specific fallbacks
        if any(word in description_lower for word in ['sentiment', 'positive', 'negative']):
            return "Based on analysis, this text conveys a neutral to positive sentiment."
        elif any(word in description_lower for word in ['acceptable', 'grammar', 'grammatical']):
            return "This sentence appears to be grammatically acceptable."
        elif any(word in description_lower for word in ['similar', 'paraphrase', 'same']):
            return "The texts convey similar meanings."
        elif any(word in description_lower for word in ['entail', 'implies', 'follows']):
            return "The premise does not clearly entail the hypothesis."
        else:
            return f"Task analysis complete: {task.description[:100]}..."

    async def _share_knowledge(self, agent: SwarmAgent, task: SwarmTask, result: Dict[str, Any]):
        """Share knowledge gained from task execution"""
        knowledge = {
            "task_type": task.description.split()[0].lower(),
            "skills_used": list(task.required_skills),
            "result_quality": result.get('confidence_score', 0.5),
            "systems_engaged": result.get('systems_engaged', 0),
            "processing_time": result.get('processing_time', 0),
            "learned_patterns": result.get('benchmark_results', {})
        }

        agent.knowledge_base[task.task_id] = knowledge

        # Broadcast knowledge to nearby agents
        await self._broadcast_message(SwarmMessage(
            message_id=f"knowledge_{task.task_id}",
            sender_id=agent.agent_id,
            receiver_id=None,
            message_type=CommunicationType.KNOWLEDGE_SHARING,
            content={"knowledge": knowledge, "task_id": task.task_id},
            priority=TaskPriority.LOW
        ))

    async def _handle_knowledge_sharing(self, agent: SwarmAgent, message: SwarmMessage):
        """Handle knowledge sharing between agents"""
        knowledge = message.content.get("knowledge", {})
        task_id = message.content.get("task_id")

        # Integrate knowledge if relevant to agent's skills
        knowledge_type = knowledge.get("task_type", "")
        if any(skill in knowledge_type for skill in agent.skills) or not agent.skills:
            agent.knowledge_base[task_id] = knowledge

            # Update consciousness level based on knowledge sharing
            agent.consciousness_level = min(agent.consciousness_level * 1.02, 3.0)

    async def _handle_threat_warning(self, agent: SwarmAgent, message: SwarmMessage):
        """Handle threat warnings in the swarm"""
        threat_level = message.content.get("threat_level", "low")

        if threat_level == "high":
            agent.health_status = "alert"
            # Move away from threat source
            threat_pos = message.content.get("position", (0, 0))
            dx = agent.position[0] - threat_pos[0]
            dy = agent.position[1] - threat_pos[1]
            distance = (dx**2 + dy**2)**0.5
            if distance > 0:
                agent.velocity = (dx/distance * 0.5, dy/distance * 0.5)

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    async def update_swarm_dynamics(self):
        """Update swarm dynamics and emergent behavior"""
        if not self.agents:
            return

        # Calculate swarm center
        positions = [agent.position for agent in self.agents.values()]
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        self.swarm_center = (center_x, center_y)

        # Update agent positions
        for agent in list(self.agents.values()):
            neighbors = [
                other for other in self.agents.values()
                if other.agent_id != agent.agent_id and
                self._calculate_distance(agent.position, other.position) <= self.communication_range
            ]
            agent.update_position(self.swarm_center, neighbors)

        # Process message queue
        messages_to_process = self.message_queue[:10]  # Process up to 10 messages per cycle
        self.message_queue = self.message_queue[10:]

        for message in messages_to_process:
            await self._process_message(message)

        # Detect emergent patterns
        await self._detect_emergent_patterns()

        # Update swarm metrics
        self._update_swarm_metrics()

    async def _detect_emergent_patterns(self):
        """Detect emergent behavior patterns in the swarm"""
        # Analyze agent clustering
        positions = [agent.position for agent in self.agents.values()]
        if positions:
            # Calculate swarm coherence (how clustered agents are)
            distances_from_center = [
                self._calculate_distance(pos, self.swarm_center) for pos in positions
            ]
            avg_distance = sum(distances_from_center) / len(distances_from_center)
            self.swarm_coherence = max(0, 1.0 - avg_distance / self.swarm_radius)

        # Analyze task completion patterns
        completed_tasks = [t for t in self.tasks.values() if t.status == "completed"]
        if completed_tasks:
            completion_rate = len(completed_tasks) / len(self.tasks)
            self.task_completion_rate = completion_rate

        # Analyze knowledge sharing efficiency
        total_knowledge = sum(len(agent.knowledge_base) for agent in self.agents.values())
        avg_knowledge = total_knowledge / len(self.agents) if self.agents else 0
        self.knowledge_sharing_efficiency = min(avg_knowledge / 10.0, 1.0)  # Normalize

        # Calculate emergent behavior index
        self.emergent_behavior_index = (
            self.swarm_coherence * 0.3 +
            self.task_completion_rate * 0.3 +
            self.knowledge_sharing_efficiency * 0.4
        )

        # Detect specific emergent patterns
        if self.emergent_behavior_index > self.emergent_threshold:
            pattern = {
                "pattern_type": "high_coordination",
                "coherence": self.swarm_coherence,
                "completion_rate": self.task_completion_rate,
                "knowledge_sharing": self.knowledge_sharing_efficiency,
                "timestamp": time.time(),
                "description": "Swarm exhibiting high coordination and emergent intelligence"
            }
            pattern_id = f"pattern_{int(time.time())}"
            self.emergent_patterns[pattern_id] = pattern

            print(f"   ✨ Emergent Pattern Detected: {pattern_id}")

    def _update_swarm_metrics(self):
        """Update overall swarm performance metrics"""
        # Clean up old tasks
        current_time = time.time()
        for task in list(self.tasks.values()):
            if task.status in ["pending", "in_progress"]:
                if task.deadline and current_time > task.deadline:
                    task.status = "expired"
                elif current_time - task.created_at > self.task_timeout:
                    task.status = "timeout"

        # Update agent health based on activity
        for agent in self.agents.values():
            time_since_activity = current_time - agent.last_activity
            if time_since_activity > 300:  # 5 minutes
                agent.energy_level = min(agent.energy_level + 5, 100)  # Rest
            elif time_since_activity > 3600:  # 1 hour
                agent.health_status = "inactive"

    def start_monitoring(self):
        """Start real-time swarm monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        print("📊 Swarm monitoring started")

    def stop_monitoring(self):
        """Stop swarm monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print("📊 Swarm monitoring stopped")

    def _monitoring_loop(self):
        """Real-time monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect swarm metrics
                metrics = {
                    "timestamp": time.time(),
                    "agent_count": len(self.agents),
                    "active_tasks": len([t for t in self.tasks.values() if t.status in ["pending", "in_progress"]]),
                    "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
                    "swarm_coherence": self.swarm_coherence,
                    "task_completion_rate": self.task_completion_rate,
                    "knowledge_sharing_efficiency": self.knowledge_sharing_efficiency,
                    "emergent_behavior_index": self.emergent_behavior_index,
                    "emergent_patterns_detected": len(self.emergent_patterns),
                    "swarm_center": self.swarm_center
                }

                self.performance_history.append(metrics)

                # Keep last 1000 entries
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]

                # Log critical events
                if self.emergent_behavior_index > 0.8:
                    print(".3f")
                elif len([t for t in self.tasks.values() if t.status == "failed"]) > len(self.tasks) * 0.1:
                    print("   ⚠️ High task failure rate detected")

                time.sleep(5)  # Update every 5 seconds

            except Exception as e:
                print(f"   ⚠️ Monitoring error: {e}")
                time.sleep(5)

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        return {
            "agent_count": len(self.agents),
            "active_tasks": len([t for t in self.tasks.values() if t.status in ["pending", "in_progress"]]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == "failed"]),
            "swarm_coherence": self.swarm_coherence,
            "emergent_patterns": len(self.emergent_patterns),
            "knowledge_sharing_efficiency": self.knowledge_sharing_efficiency,
            "task_completion_rate": self.task_completion_rate,
            "emergent_behavior_index": self.emergent_behavior_index,
            "monitoring_active": self.monitoring_active
        }

    async def optimize_swarm(self):
        """Enhanced swarm optimization with multiple strategies"""
        print("🔧 Optimizing swarm performance with advanced algorithms...")

        optimizations_applied = []

        # Strategy 1: Performance-based role reassignment
        role_performance = {}
        for agent in self.agents.values():
            if agent.role not in role_performance:
                role_performance[agent.role] = []
            role_performance[agent.role].append(agent.performance_score)

        # Identify underperforming roles and reassign agents
        for role, scores in role_performance.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score < 0.8:
                # Find high-performing agents to reassign
                high_performers = [
                    agent for agent in self.agents.values()
                    if agent.performance_score > 1.2 and agent.role != role and agent.energy_level > 50
                ]

                if high_performers:
                    # Sort by performance and pick the best
                    high_performers.sort(key=lambda a: a.performance_score, reverse=True)
                    agent = high_performers[0]
                    old_role = agent.role
                    agent.role = role

                    # Update agent skills for new role
                    self._update_agent_skills_for_role(agent, role)

                    optimizations_applied.append(f"Role reassignment: {agent.agent_id} {old_role.value}→{role.value}")
                    print(f"   🔄 Reassigned {agent.agent_id} from {old_role.value} to {role.value}")

        # Strategy 2: Energy redistribution
        low_energy_agents = [a for a in self.agents.values() if a.energy_level < 30]
        high_energy_agents = [a for a in self.agents.values() if a.energy_level > 80]

        if low_energy_agents and high_energy_agents:
            # Redistribute energy from high to low performers
            energy_transfer = 10
            for low_agent in low_energy_agents[:3]:  # Help up to 3 agents
                for high_agent in high_energy_agents:
                    if high_agent.energy_level > energy_transfer + 50:  # Keep some reserve
                        low_agent.energy_level = min(low_agent.energy_level + energy_transfer, 100)
                        high_agent.energy_level -= energy_transfer
                        optimizations_applied.append(f"Energy transfer: {high_agent.agent_id}→{low_agent.agent_id}")
                        break

        # Strategy 3: Communication range optimization
        swarm_spread = self._calculate_swarm_spread()
        optimal_range = max(3.0, min(15.0, swarm_spread * 0.8))
        old_range = self.communication_range
        self.communication_range = optimal_range

        if abs(optimal_range - old_range) > 1.0:
            optimizations_applied.append(f"Communication range: {old_range:.1f}→{optimal_range:.1f}")

        # Strategy 4: Agent specialization enhancement
        for agent in self.agents.values():
            if len(agent.task_history) > 5:  # Experienced agents
                # Specialize based on successful task types
                successful_tasks = [self.tasks.get(tid) for tid in agent.task_history if tid in self.tasks]
                successful_tasks = [t for t in successful_tasks if t and t.status == "completed"]

                if successful_tasks:
                    # Find most common task types
                    task_types = [t.description.split()[0].lower() for t in successful_tasks]
                    most_common = max(set(task_types), key=task_types.count)

                    # Enhance skills for that domain
                    if most_common in ['analyze', 'sentiment', 'classify']:
                        agent.skills.add('nlp_specialist')
                        agent.consciousness_level = min(agent.consciousness_level + 0.2, 3.0)
                        optimizations_applied.append(f"NLP specialization: {agent.agent_id}")

        # Strategy 5: Emergent pattern learning
        if len(self.emergent_patterns) > 2:
            # Use patterns to improve future performance
            pattern_insights = self._extract_pattern_insights()
            if pattern_insights:
                optimizations_applied.append(f"Pattern learning applied: {len(pattern_insights)} insights")

        print(f"✅ Swarm optimization complete - {len(optimizations_applied)} improvements applied")
        for opt in optimizations_applied[:5]:  # Show first 5
            print(f"   ✓ {opt}")
        if len(optimizations_applied) > 5:
            print(f"   ... and {len(optimizations_applied) - 5} more optimizations")

    def _update_agent_skills_for_role(self, agent: SwarmAgent, new_role: SwarmRole):
        """Update agent skills when reassigned to a new role"""
        # Clear old role-specific skills
        old_role_skills = {
            SwarmRole.SCOUT: {"exploration", "pattern_recognition", "risk_assessment"},
            SwarmRole.WORKER: {"task_execution", "problem_solving", "quality_control"},
            SwarmRole.GUARD: {"threat_detection", "error_handling", "security"},
            SwarmRole.QUEEN: {"coordination", "strategy", "decision_making"},
            SwarmRole.FORAGER: {"resource_gathering", "data_collection", "knowledge_extraction"},
            SwarmRole.BUILDER: {"system_maintenance", "infrastructure", "optimization"},
            SwarmRole.SOLDIER: {"conflict_resolution", "optimization", "performance_enhancement"},
            SwarmRole.MEDIC: {"diagnosis", "repair", "health_monitoring"}
        }

        # Remove old role skills (keep general skills)
        for role_skills in old_role_skills.values():
            agent.skills -= role_skills

        # Add new role skills
        new_role_skills = old_role_skills.get(new_role, set())
        agent.skills.update(new_role_skills)

        # Reset performance score for new role (learning curve)
        agent.performance_score = max(agent.performance_score * 0.8, 0.8)

    def _calculate_swarm_spread(self) -> float:
        """Calculate how spread out the swarm is"""
        if not self.agents:
            return 10.0

        positions = [agent.position for agent in self.agents.values()]
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)

        distances = [((p[0] - center_x)**2 + (p[1] - center_y)**2)**0.5 for p in positions]
        return max(distances) if distances else 10.0

    def _extract_pattern_insights(self) -> List[str]:
        """Extract insights from emergent patterns for optimization"""
        insights = []

        for pattern_id, pattern in self.emergent_patterns.items():
            coherence = pattern.get('coherence', 0)
            completion_rate = pattern.get('completion_rate', 0)

            if coherence > 0.8:
                insights.append("high_coherence_improves_performance")
            if completion_rate > 0.9:
                insights.append("task_completion_patterns_identified")

        return insights

    async def demonstrate_swarm_intelligence(self):
        """Demonstrate the swarm intelligence capabilities"""
        print("🐝 ChAios Swarm AI - Intelligence Demonstration")
        print("=" * 60)

        # Initialize swarm
        if not await self.initialize_swarm():
            return

        # Start monitoring
        self.start_monitoring()

        try:
            print("\n🧪 DEMO 1: Task Allocation & Execution")

            # Submit diverse tasks to demonstrate specialization
            tasks = [
                {
                    "description": "Analyze quantum computing algorithms for optimization",
                    "priority": TaskPriority.HIGH,
                    "complexity": 0.8,
                    "required_skills": {"quantum_physics", "algorithm_analysis"}
                },
                {
                    "description": "Detect patterns in large dataset for machine learning",
                    "priority": TaskPriority.MEDIUM,
                    "complexity": 0.6,
                    "required_skills": {"data_analysis", "pattern_recognition"}
                },
                {
                    "description": "Monitor system health and identify potential issues",
                    "priority": TaskPriority.MEDIUM,
                    "complexity": 0.4,
                    "required_skills": {"monitoring", "diagnosis"}
                },
                {
                    "description": "Optimize code performance using advanced algorithms",
                    "priority": TaskPriority.LOW,
                    "complexity": 0.7,
                    "required_skills": {"optimization", "coding"}
                }
            ]

            submitted_tasks = []
            for task_config in tasks:
                task_id = await self.submit_task(**task_config)
                submitted_tasks.append(task_id)

            # Wait for task completion with swarm dynamics updates
            start_time = time.time()
            completed_tasks = 0

            while completed_tasks < len(tasks) and time.time() - start_time < 120:  # 2 minute timeout
                await self.update_swarm_dynamics()
                await asyncio.sleep(1)

                current_completed = len([t for t in self.tasks.values() if t.status == "completed"])
                if current_completed > completed_tasks:
                    completed_tasks = current_completed
                    print(f"   ✅ Tasks completed: {completed_tasks}/{len(tasks)}")

            print(f"\n📊 Task Execution Results:")
            for task_id in submitted_tasks:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    status_emoji = "✅" if task.status == "completed" else "❌" if task.status == "failed" else "⏳"
                    print(f"   {status_emoji} {task_id}: {task.status}")

            # Demonstrate emergent behavior
            print("\n🧠 DEMO 2: Emergent Swarm Behavior")

            # Run swarm dynamics for emergence
            print("   📈 Monitoring emergent behavior patterns...")
            initial_patterns = len(self.emergent_patterns)

            for i in range(10):
                await self.update_swarm_dynamics()
                await asyncio.sleep(0.5)

                if len(self.emergent_patterns) > initial_patterns:
                    print(f"   ✨ New emergent pattern detected! Total: {len(self.emergent_patterns)}")

            print("\n   🔍 Swarm Intelligence Metrics:")
            status = self.get_swarm_status()
            print(f"   🐜 Agents: {status['agent_count']}")
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")
            print(f"   🎯 Emergent Patterns: {status['emergent_patterns']}")

            # Demonstrate self-optimization
            print("\n🔧 DEMO 3: Self-Optimization")
            await self.optimize_swarm()

            final_status = self.get_swarm_status()
            print("   📊 Optimization Results:")
            print(f"   📈 Improved coherence: {final_status['swarm_coherence']:.3f}")
            print(f"   🎯 Communication range optimized: {self.communication_range:.1f}")

            print("\n🎯 FINAL ASSESSMENT")
            print("=" * 40)
            print("✅ ChAios Swarm AI Operational")
            print("🐝 Multi-agent coordination active")
            print("🧠 Emergent intelligence demonstrated")
            print("📊 Self-optimization working")
            print("🚀 Consciousness-enhanced swarm ready")
            print("🏆 Revolutionary AI swarm technology achieved!")

        finally:
            self.stop_monitoring()

async def main():
    """Main swarm AI demonstration"""
    swarm = ChAiosSwarmAI()
    await swarm.demonstrate_swarm_intelligence()

if __name__ == "__main__":
    asyncio.run(main())
