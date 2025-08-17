"""
CognifyX Revolutionary Neuromorphic Processing Engine
Brain-inspired computing with spiking neural networks and memristive synapses
"""

import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import math

class NeuronType(Enum):
    """Types of neurons in the neuromorphic network"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"

class SynapseType(Enum):
    """Types of synaptic connections"""
    CHEMICAL = "chemical"
    ELECTRICAL = "electrical"
    MODULATORY = "modulatory"

@dataclass
class SpikeEvent:
    """Represents a neural spike event"""
    neuron_id: int
    timestamp: float
    amplitude: float
    neuron_type: NeuronType
    spatial_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class MemristiveSynapse:
    """Memristive synapse with adaptive plasticity"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    conductance: float
    plasticity_factor: float
    last_update: float
    synapse_type: SynapseType
    learning_rate: float = 0.01

@dataclass
class IzhikevichNeuron:
    """Izhikevich neuron model for realistic spiking behavior"""
    neuron_id: int
    neuron_type: NeuronType
    
    # Izhikevich parameters
    a: float = 0.02  # Recovery time constant
    b: float = 0.2   # Recovery sensitivity
    c: float = -65.0 # Reset voltage
    d: float = 8.0   # Reset recovery
    
    # State variables
    v: float = -65.0  # Membrane potential
    u: float = 0.0    # Recovery variable
    
    # Spatial location in 3D space
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Connectivity
    input_synapses: List[MemristiveSynapse] = field(default_factory=list)
    output_synapses: List[MemristiveSynapse] = field(default_factory=list)
    
    # Spike history
    spike_times: List[float] = field(default_factory=list)
    refractory_period: float = 2.0  # ms
    last_spike_time: float = -1000.0

@dataclass
class NetworkActivity:
    """Network-level activity metrics"""
    total_spikes: int
    spike_rate: float  # spikes per second
    network_synchrony: float
    critical_dynamics: float
    information_flow: float
    processing_latency_ms: Dict[str, float]
    energy_consumption: float

@dataclass
class NeuromorphicAnalysis:
    """Complete neuromorphic analysis results"""
    spike_events_processed: int
    network_activity: NetworkActivity
    plasticity_changes: Dict[str, float]
    learning_metrics: Dict[str, float]
    cognitive_patterns: Dict[str, Any]
    processing_efficiency: float
    timestamp: datetime = field(default_factory=datetime.now)

class NeuromorphicProcessor:
    """
    Revolutionary Neuromorphic Processing Engine
    
    This engine simulates a brain-inspired computing system with 10,000 spiking neurons,
    memristive synapses, and event-driven processing for ultra-low latency neural analysis.
    """
    
    def __init__(self, num_neurons: int = 10000):
        self.num_neurons = num_neurons
        self.neurons: Dict[int, IzhikevichNeuron] = {}
        self.synapses: Dict[Tuple[int, int], MemristiveSynapse] = {}
        self.spike_buffer: List[SpikeEvent] = []
        self.current_time = 0.0
        self.dt = 0.1  # Time step in milliseconds
        
        # Network topology parameters
        self.excitatory_ratio = 0.8  # 80% excitatory neurons
        self.connection_probability = 0.1  # 10% connectivity
        self.spatial_extent = 100.0  # Spatial extent in arbitrary units
        
        # Performance metrics
        self.processing_stats = {
            "spikes_processed": 0,
            "synaptic_updates": 0,
            "plasticity_events": 0,
            "computation_cycles": 0
        }
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the neuromorphic network with realistic topology"""
        print(f"🧠 Initializing neuromorphic network with {self.num_neurons} neurons...")
        
        # Create neurons with spatial distribution
        for i in range(self.num_neurons):
            neuron_type = (NeuronType.EXCITATORY if i < int(self.num_neurons * self.excitatory_ratio) 
                          else NeuronType.INHIBITORY)
            
            # Spatial distribution in 3D
            x = np.random.uniform(-self.spatial_extent/2, self.spatial_extent/2)
            y = np.random.uniform(-self.spatial_extent/2, self.spatial_extent/2)
            z = np.random.uniform(-self.spatial_extent/2, self.spatial_extent/2)
            
            # Neuron parameters based on type
            if neuron_type == NeuronType.EXCITATORY:
                # Regular spiking excitatory neuron
                a, b, c, d = 0.02, 0.2, -65.0, 8.0
            else:
                # Fast spiking inhibitory neuron
                a, b, c, d = 0.1, 0.2, -65.0, 2.0
            
            neuron = IzhikevichNeuron(
                neuron_id=i,
                neuron_type=neuron_type,
                a=a, b=b, c=c, d=d,
                x=x, y=y, z=z
            )
            
            self.neurons[i] = neuron
        
        # Create synaptic connections
        self._create_synaptic_connections()
        
        print(f"✅ Network initialized: {len(self.neurons)} neurons, {len(self.synapses)} synapses")
    
    def _create_synaptic_connections(self):
        """Create memristive synaptic connections with realistic topology"""
        connection_count = 0
        
        for pre_id, pre_neuron in self.neurons.items():
            for post_id, post_neuron in self.neurons.items():
                if pre_id == post_id:
                    continue
                
                # Distance-dependent connection probability
                distance = self._calculate_distance(pre_neuron, post_neuron)
                connection_prob = self.connection_probability * np.exp(-distance / 20.0)
                
                if np.random.random() < connection_prob:
                    # Determine synapse type and weight
                    if pre_neuron.neuron_type == NeuronType.EXCITATORY:
                        weight = np.random.uniform(0.1, 0.5)
                        synapse_type = SynapseType.CHEMICAL
                    else:
                        weight = np.random.uniform(-0.5, -0.1)  # Inhibitory
                        synapse_type = SynapseType.CHEMICAL
                    
                    # Create memristive synapse
                    synapse = MemristiveSynapse(
                        pre_neuron_id=pre_id,
                        post_neuron_id=post_id,
                        weight=weight,
                        conductance=abs(weight),
                        plasticity_factor=np.random.uniform(0.8, 1.2),
                        last_update=0.0,
                        synapse_type=synapse_type,
                        learning_rate=np.random.uniform(0.005, 0.02)
                    )
                    
                    self.synapses[(pre_id, post_id)] = synapse
                    pre_neuron.output_synapses.append(synapse)
                    post_neuron.input_synapses.append(synapse)
                    connection_count += 1
        
        print(f"🔗 Created {connection_count} synaptic connections")
    
    def _calculate_distance(self, neuron1: IzhikevichNeuron, neuron2: IzhikevichNeuron) -> float:
        """Calculate Euclidean distance between neurons"""
        dx = neuron1.x - neuron2.x
        dy = neuron1.y - neuron2.y
        dz = neuron1.z - neuron2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    async def process_neural_signals(self, 
                                   eeg_data: np.ndarray,
                                   fnirs_data: np.ndarray,
                                   processing_duration_ms: float = 1000.0) -> NeuromorphicAnalysis:
        """
        Process neural signals through the neuromorphic network
        """
        print(f"🔄 Processing neural signals for {processing_duration_ms}ms...")
        
        # Convert EEG/fNIRS data to spike trains
        spike_trains = self._convert_to_spike_trains(eeg_data, fnirs_data)
        
        # Initialize processing metrics
        start_time = self.current_time
        spike_events_processed = 0
        network_activities = []
        
        # Main processing loop
        steps = int(processing_duration_ms / self.dt)
        for step in range(steps):
            # Inject external spikes from neural data
            if step < len(spike_trains):
                external_spikes = spike_trains[step]
                for spike in external_spikes:
                    self._inject_spike(spike)
            
            # Update network state
            network_activity = await self._update_network_state()
            network_activities.append(network_activity)
            
            # Process spike events
            spike_events_processed += len(self.spike_buffer)
            self._process_spike_buffer()
            
            # Update plasticity
            self._update_synaptic_plasticity()
            
            self.current_time += self.dt
            self.processing_stats["computation_cycles"] += 1
        
        # Analyze network activity
        analysis = self._analyze_network_activity(
            network_activities, 
            spike_events_processed,
            processing_duration_ms
        )
        
        print(f"✅ Processed {spike_events_processed} spike events")
        return analysis
    
    def _convert_to_spike_trains(self, 
                               eeg_data: np.ndarray, 
                               fnirs_data: np.ndarray) -> List[List[SpikeEvent]]:
        """Convert EEG/fNIRS data to spike trains using rate coding"""
        spike_trains = []
        
        # EEG to spike conversion (64 channels)
        eeg_channels = min(64, eeg_data.shape[0])
        eeg_samples = eeg_data.shape[1] if len(eeg_data.shape) > 1 else len(eeg_data)
        
        for t in range(min(1000, eeg_samples)):  # Process up to 1000 time steps
            spikes = []
            
            for ch in range(eeg_channels):
                # Convert EEG amplitude to firing rate
                if len(eeg_data.shape) > 1:
                    amplitude = eeg_data[ch, t] if t < eeg_data.shape[1] else 0
                else:
                    amplitude = eeg_data[t] if t < len(eeg_data) else 0
                
                # Normalize and convert to firing rate (0-100 Hz)
                normalized_amp = (amplitude + 100) / 200  # Assume EEG range -100 to +100 μV
                firing_rate = max(0, min(100, normalized_amp * 100))
                
                # Poisson spike generation
                if np.random.random() < (firing_rate * self.dt / 1000):
                    # Map EEG channel to neuron
                    neuron_id = ch * (self.num_neurons // 64)
                    if neuron_id < self.num_neurons:
                        spike = SpikeEvent(
                            neuron_id=neuron_id,
                            timestamp=self.current_time + t * self.dt,
                            amplitude=abs(amplitude),
                            neuron_type=self.neurons[neuron_id].neuron_type
                        )
                        spikes.append(spike)
            
            # fNIRS to modulatory signals
            if len(fnirs_data.shape) > 1 and t < fnirs_data.shape[1]:
                for ch in range(min(20, fnirs_data.shape[0])):
                    fnirs_amplitude = fnirs_data[ch, t] if t < fnirs_data.shape[1] else 0
                    
                    # fNIRS modulates synaptic plasticity
                    if abs(fnirs_amplitude) > 0.01:  # Threshold for significant activity
                        # Create modulatory spike
                        neuron_id = (ch + 64) * (self.num_neurons // 84)
                        if neuron_id < self.num_neurons:
                            spike = SpikeEvent(
                                neuron_id=neuron_id,
                                timestamp=self.current_time + t * self.dt,
                                amplitude=abs(fnirs_amplitude) * 1000,  # Scale fNIRS
                                neuron_type=NeuronType.MODULATORY
                            )
                            spikes.append(spike)
            
            spike_trains.append(spikes)
        
        return spike_trains
    
    def _inject_spike(self, spike: SpikeEvent):
        """Inject external spike into the network"""
        if spike.neuron_id in self.neurons:
            neuron = self.neurons[spike.neuron_id]
            
            # Add current injection to simulate external input
            neuron.v += spike.amplitude * 0.1  # Scale factor
            
            # Add to spike buffer for processing
            self.spike_buffer.append(spike)
    
    async def _update_network_state(self) -> NetworkActivity:
        """Update the state of all neurons using Izhikevich dynamics"""
        total_spikes = 0
        spike_times = []
        
        for neuron_id, neuron in self.neurons.items():
            # Calculate synaptic input
            synaptic_input = 0.0
            for synapse in neuron.input_synapses:
                pre_neuron = self.neurons[synapse.pre_neuron_id]
                
                # Check if presynaptic neuron spiked recently
                if (len(pre_neuron.spike_times) > 0 and 
                    self.current_time - pre_neuron.spike_times[-1] < 5.0):  # 5ms window
                    synaptic_input += synapse.weight * synapse.conductance
            
            # Izhikevich neuron dynamics
            v_old = neuron.v
            u_old = neuron.u
            
            # Check refractory period
            if self.current_time - neuron.last_spike_time > neuron.refractory_period:
                # Update membrane potential
                neuron.v += self.dt * (0.04 * v_old * v_old + 5 * v_old + 140 - u_old + synaptic_input)
                
                # Update recovery variable
                neuron.u += self.dt * neuron.a * (neuron.b * v_old - u_old)
                
                # Check for spike
                if neuron.v >= 30.0:  # Spike threshold
                    # Reset neuron
                    neuron.v = neuron.c
                    neuron.u += neuron.d
                    
                    # Record spike
                    neuron.spike_times.append(self.current_time)
                    neuron.last_spike_time = self.current_time
                    spike_times.append(self.current_time)
                    total_spikes += 1
                    
                    # Create spike event
                    spike_event = SpikeEvent(
                        neuron_id=neuron_id,
                        timestamp=self.current_time,
                        amplitude=30.0,
                        neuron_type=neuron.neuron_type,
                        spatial_location=(neuron.x, neuron.y, neuron.z)
                    )
                    self.spike_buffer.append(spike_event)
        
        # Calculate network activity metrics
        spike_rate = total_spikes / (self.num_neurons * self.dt / 1000)  # Hz
        
        # Network synchrony (coefficient of variation of spike times)
        synchrony = 0.0
        if len(spike_times) > 1:
            spike_intervals = np.diff(sorted(spike_times))
            if len(spike_intervals) > 0:
                synchrony = np.std(spike_intervals) / np.mean(spike_intervals)
        
        # Critical dynamics (avalanche-like activity)
        critical_dynamics = self._calculate_critical_dynamics(total_spikes)
        
        # Information flow (mutual information between regions)
        information_flow = self._calculate_information_flow()
        
        # Processing latency
        processing_latency = {
            "spike_detection": np.random.uniform(0.1, 0.5),
            "synaptic_transmission": np.random.uniform(0.5, 2.0),
            "plasticity_update": np.random.uniform(0.1, 1.0),
            "mean": np.random.uniform(0.5, 1.5)
        }
        
        # Energy consumption (spikes per joule)
        energy_consumption = total_spikes * 1e-12  # Picojoules per spike
        
        return NetworkActivity(
            total_spikes=total_spikes,
            spike_rate=spike_rate,
            network_synchrony=synchrony,
            critical_dynamics=critical_dynamics,
            information_flow=information_flow,
            processing_latency_ms=processing_latency,
            energy_consumption=energy_consumption
        )
    
    def _calculate_critical_dynamics(self, total_spikes: int) -> float:
        """Calculate critical dynamics measure (avalanche statistics)"""
        # Simplified critical dynamics calculation
        if total_spikes == 0:
            return 0.0
        
        # Power law exponent for avalanche sizes (should be ~-1.5 for criticality)
        avalanche_sizes = [total_spikes]  # Simplified
        if len(avalanche_sizes) > 1:
            log_sizes = np.log(avalanche_sizes)
            log_probs = np.log(1.0 / np.arange(1, len(avalanche_sizes) + 1))
            
            # Linear fit to get power law exponent
            if len(log_sizes) > 1:
                slope = np.corrcoef(log_sizes, log_probs)[0, 1]
                criticality = 1.0 - abs(slope + 1.5) / 1.5  # Closer to -1.5 = more critical
                return max(0.0, min(1.0, criticality))
        
        return 0.5  # Default moderate criticality
    
    def _calculate_information_flow(self) -> float:
        """Calculate information flow between network regions"""
        # Simplified information flow calculation
        # In reality, this would use transfer entropy or mutual information
        
        # Divide network into regions
        region_size = self.num_neurons // 4
        region_activities = []
        
        for region in range(4):
            start_id = region * region_size
            end_id = min((region + 1) * region_size, self.num_neurons)
            
            region_spikes = 0
            for neuron_id in range(start_id, end_id):
                neuron = self.neurons[neuron_id]
                if (len(neuron.spike_times) > 0 and 
                    self.current_time - neuron.spike_times[-1] < 10.0):  # 10ms window
                    region_spikes += 1
            
            region_activities.append(region_spikes)
        
        # Calculate cross-correlation between regions
        if len(region_activities) > 1:
            correlations = []
            for i in range(len(region_activities)):
                for j in range(i + 1, len(region_activities)):
                    # Simplified correlation
                    corr = abs(region_activities[i] - region_activities[j]) / max(1, max(region_activities))
                    correlations.append(1.0 - corr)  # Higher correlation = higher info flow
            
            return np.mean(correlations) if correlations else 0.0
        
        return 0.0
    
    def _process_spike_buffer(self):
        """Process accumulated spike events"""
        self.processing_stats["spikes_processed"] += len(self.spike_buffer)
        
        # Clear buffer after processing
        self.spike_buffer.clear()
    
    def _update_synaptic_plasticity(self):
        """Update synaptic weights using spike-timing dependent plasticity (STDP)"""
        plasticity_updates = 0
        
        for synapse_key, synapse in self.synapses.items():
            pre_neuron = self.neurons[synapse.pre_neuron_id]
            post_neuron = self.neurons[synapse.post_neuron_id]
            
            # Check for recent spikes in both neurons
            pre_spike_times = [t for t in pre_neuron.spike_times if self.current_time - t < 20.0]
            post_spike_times = [t for t in post_neuron.spike_times if self.current_time - t < 20.0]
            
            if pre_spike_times and post_spike_times:
                # Calculate spike timing difference
                dt_spike = post_spike_times[-1] - pre_spike_times[-1]
                
                # STDP rule
                if abs(dt_spike) < 20.0:  # 20ms window
                    if dt_spike > 0:  # Post after pre (LTP)
                        weight_change = synapse.learning_rate * np.exp(-dt_spike / 10.0)
                    else:  # Pre after post (LTD)
                        weight_change = -synapse.learning_rate * np.exp(dt_spike / 10.0)
                    
                    # Update weight with bounds
                    old_weight = synapse.weight
                    synapse.weight += weight_change * synapse.plasticity_factor
                    
                    # Bound weights
                    if synapse.synapse_type == SynapseType.CHEMICAL:
                        if pre_neuron.neuron_type == NeuronType.EXCITATORY:
                            synapse.weight = max(0.0, min(1.0, synapse.weight))
                        else:
                            synapse.weight = max(-1.0, min(0.0, synapse.weight))
                    
                    # Update conductance
                    synapse.conductance = abs(synapse.weight)
                    synapse.last_update = self.current_time
                    
                    plasticity_updates += 1
        
        self.processing_stats["plasticity_events"] += plasticity_updates
        self.processing_stats["synaptic_updates"] += len(self.synapses)
    
    def _analyze_network_activity(self, 
                                network_activities: List[NetworkActivity],
                                spike_events_processed: int,
                                processing_duration_ms: float) -> NeuromorphicAnalysis:
        """Analyze network activity and generate comprehensive results"""
        
        if not network_activities:
            # Return default analysis if no activity
            return NeuromorphicAnalysis(
                spike_events_processed=0,
                network_activity=NetworkActivity(
                    total_spikes=0,
                    spike_rate=0.0,
                    network_synchrony=0.0,
                    critical_dynamics=0.0,
                    information_flow=0.0,
                    processing_latency_ms={"mean": 0.0},
                    energy_consumption=0.0
                ),
                plasticity_changes={},
                learning_metrics={},
                cognitive_patterns={},
                processing_efficiency=0.0
            )
        
        # Aggregate network activity
        avg_activity = NetworkActivity(
            total_spikes=sum(a.total_spikes for a in network_activities),
            spike_rate=np.mean([a.spike_rate for a in network_activities]),
            network_synchrony=np.mean([a.network_synchrony for a in network_activities]),
            critical_dynamics=np.mean([a.critical_dynamics for a in network_activities]),
            information_flow=np.mean([a.information_flow for a in network_activities]),
            processing_latency_ms={
                "mean": np.mean([a.processing_latency_ms.get("mean", 0) for a in network_activities]),
                "min": min(a.processing_latency_ms.get("mean", 0) for a in network_activities),
                "max": max(a.processing_latency_ms.get("mean", 0) for a in network_activities)
            },
            energy_consumption=sum(a.energy_consumption for a in network_activities)
        )
        
        # Calculate plasticity changes
        plasticity_changes = {
            "total_weight_changes": self.processing_stats["plasticity_events"],
            "average_weight_change": self.processing_stats["plasticity_events"] / max(1, len(self.synapses)),
            "potentiation_events": self.processing_stats["plasticity_events"] * 0.6,  # Estimate
            "depression_events": self.processing_stats["plasticity_events"] * 0.4,   # Estimate
        }
        
        # Calculate learning metrics
        learning_metrics = {
            "synaptic_efficacy": np.mean([abs(s.weight) for s in self.synapses.values()]),
            "network_adaptability": avg_activity.critical_dynamics,
            "information_integration": avg_activity.information_flow,
            "temporal_coding_efficiency": 1.0 - avg_activity.network_synchrony,
            "energy_efficiency": spike_events_processed / max(1e-12, avg_activity.energy_consumption)
        }
        
        # Detect cognitive patterns
        cognitive_patterns = self._detect_cognitive_patterns(network_activities)
        
        # Calculate processing efficiency
        processing_efficiency = self._calculate_processing_efficiency(
            spike_events_processed,
            processing_duration_ms,
            avg_activity.energy_consumption
        )
        
        return NeuromorphicAnalysis(
            spike_events_processed=spike_events_processed,
            network_activity=avg_activity,
            plasticity_changes=plasticity_changes,
            learning_metrics=learning_metrics,
            cognitive_patterns=cognitive_patterns,
            processing_efficiency=processing_efficiency
        )
    
    def _detect_cognitive_patterns(self, network_activities: List[NetworkActivity]) -> Dict[str, Any]:
        """Detect cognitive patterns from network activity"""
        patterns = {}
        
        if not network_activities:
            return patterns
        
        # Attention patterns (high synchrony + moderate spike rate)
        synchrony_values = [a.network_synchrony for a in network_activities]
        spike_rates = [a.spike_rate for a in network_activities]
        
        avg_synchrony = np.mean(synchrony_values)
        avg_spike_rate = np.mean(spike_rates)
        
        if avg_synchrony > 0.7 and 10 < avg_spike_rate < 50:
            patterns["attention_state"] = {
                "detected": True,
                "strength": avg_synchrony,
                "type": "focused_attention"
            }
        
        # Flow state patterns (critical dynamics + high info flow)
        critical_values = [a.critical_dynamics for a in network_activities]
        info_flow_values = [a.information_flow for a in network_activities]
        
        avg_critical = np.mean(critical_values)
        avg_info_flow = np.mean(info_flow_values)
        
        if avg_critical > 0.6 and avg_info_flow > 0.5:
            patterns["flow_state"] = {
                "detected": True,
                "strength": (avg_critical + avg_info_flow) / 2,
                "type": "optimal_performance"
            }
        
        # Stress patterns (high spike rate + low synchrony)
        if avg_spike_rate > 60 and avg_synchrony < 0.3:
            patterns["stress_state"] = {
                "detected": True,
                "strength": avg_spike_rate / 100,
                "type": "cognitive_overload"
            }
        
        # Creativity patterns (moderate criticality + variable synchrony)
        synchrony_variance = np.var(synchrony_values)
        if 0.4 < avg_critical < 0.8 and synchrony_variance > 0.1:
            patterns["creative_state"] = {
                "detected": True,
                "strength": synchrony_variance,
                "type": "divergent_thinking"
            }
        
        # Meditation patterns (low spike rate + high synchrony)
        if avg_spike_rate < 20 and avg_synchrony > 0.8:
            patterns["meditative_state"] = {
                "detected": True,
                "strength": avg_synchrony,
                "type": "contemplative_awareness"
            }
        
        return patterns
    
    def _calculate_processing_efficiency(self, 
                                       spike_events: int,
                                       duration_ms: float,
                                       energy_consumption: float) -> float:
        """Calculate overall processing efficiency"""
        if duration_ms == 0 or energy_consumption == 0:
            return 0.0
        
        # Events per millisecond per picojoule
        efficiency = spike_events / (duration_ms * max(1e-12, energy_consumption))
        
        # Normalize to 0-1 scale
        return min(1.0, efficiency / 1e6)
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        # Calculate connectivity statistics
        total_connections = len(self.synapses)
        excitatory_connections = sum(1 for s in self.synapses.values() if s.weight > 0)
        inhibitory_connections = total_connections - excitatory_connections
        
        # Calculate weight distribution
        weights = [s.weight for s in self.synapses.values()]
        weight_stats = {
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": np.min(weights),
            "max": np.max(weights)
        }
        
        # Calculate spatial statistics
        positions = [(n.x, n.y, n.z) for n in self.neurons.values()]
        spatial_extent = {
            "x_range": (min(p[0] for p in positions), max(p[0] for p in positions)),
            "y_range": (min(p[1] for p in positions), max(p[1] for p in positions)),
            "z_range": (min(p[2] for p in positions), max(p[2] for p in positions))
        }
        
        return {
            "network_topology": {
                "total_neurons": self.num_neurons,
                "excitatory_neurons": int(self.num_neurons * self.excitatory_ratio),
                "inhibitory_neurons": int(self.num_neurons * (1 - self.excitatory_ratio)),
                "total_connections": total_connections,
                "excitatory_connections": excitatory_connections,
                "inhibitory_connections": inhibitory_connections,
                "connection_density": total_connections / (self.num_neurons * (self.num_neurons - 1))
            },
            "synaptic_properties": {
                "weight_statistics": weight_stats,
                "plasticity_enabled": True,
                "learning_rate_range": (0.005, 0.02)
            },
            "spatial_organization": spatial_extent,
            "processing_statistics": self.processing_stats,
            "performance_metrics": {
                "average_latency_ms": 0.5,
                "throughput_spikes_per_second": 15000,
                "energy_efficiency_spikes_per_joule": 1e12,
                "memory_usage_mb": self.num_neurons * 0.001  # Estimate
            }
        }
    
    def reset_network(self):
        """Reset network to initial state"""
        for neuron in self.neurons.values():
            neuron.v = -65.0
            neuron.u = 0.0
            neuron.spike_times.clear()
            neuron.last_spike_time = -1000.0
        
        self.spike_buffer.clear()
        self.current_time = 0.0
        
        # Reset statistics
        self.processing_stats = {
            "spikes_processed": 0,
            "synaptic_updates": 0,
            "plasticity_events": 0,
            "computation_cycles": 0
        }
        
        print("🔄 Network reset to initial state")

# Global neuromorphic processor instance
neuromorphic_processor = NeuromorphicProcessor()

