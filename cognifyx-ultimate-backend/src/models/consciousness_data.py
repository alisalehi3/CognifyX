"""
CognifyX Ultimate - Advanced Consciousness Data Model
Revolutionary database schema for consciousness and cognitive data management
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Text, Index, CheckConstraint
import uuid

db = SQLAlchemy()

class ConsciousnessSession(db.Model):
    """
    Main consciousness monitoring session model
    Tracks complete user sessions with comprehensive metadata
    """
    __tablename__ = 'consciousness_sessions'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    
    # Session metadata
    start_time = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    end_time = db.Column(db.DateTime(timezone=True), nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    session_type = db.Column(db.Enum('meditation', 'focus', 'creativity', 'learning', 'performance', name='session_types'), nullable=False)
    
    # Consciousness state tracking
    initial_consciousness_level = db.Column(db.String(50), nullable=True)
    final_consciousness_level = db.Column(db.String(50), nullable=True)
    peak_consciousness_depth = db.Column(db.Float, nullable=True)
    average_consciousness_depth = db.Column(db.Float, nullable=True)
    
    # Session quality metrics
    data_quality_score = db.Column(db.Float, nullable=True)
    signal_artifacts_count = db.Column(db.Integer, default=0)
    session_completion_rate = db.Column(db.Float, nullable=True)
    
    # Environmental context
    environment_data = db.Column(JSONB, nullable=True)  # Light, sound, temperature, etc.
    device_configuration = db.Column(JSONB, nullable=True)  # EEG/fNIRS device settings
    
    # Relationships
    neural_data = db.relationship('NeuralData', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    cognitive_assessments = db.relationship('CognitiveAssessment', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    consciousness_states = db.relationship('ConsciousnessState', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_user_time', 'user_id', 'start_time'),
        Index('idx_session_type_time', 'session_type', 'start_time'),
        CheckConstraint('peak_consciousness_depth >= 0 AND peak_consciousness_depth <= 1', name='check_peak_depth_range'),
        CheckConstraint('average_consciousness_depth >= 0 AND average_consciousness_depth <= 1', name='check_avg_depth_range'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'session_token': self.session_token,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'session_type': self.session_type,
            'initial_consciousness_level': self.initial_consciousness_level,
            'final_consciousness_level': self.final_consciousness_level,
            'peak_consciousness_depth': self.peak_consciousness_depth,
            'average_consciousness_depth': self.average_consciousness_depth,
            'data_quality_score': self.data_quality_score,
            'environment_data': self.environment_data,
            'device_configuration': self.device_configuration
        }

class NeuralData(db.Model):
    """
    High-frequency neural signal data storage
    Optimized for real-time EEG and fNIRS data ingestion
    """
    __tablename__ = 'neural_data'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = db.Column(UUID(as_uuid=True), db.ForeignKey('consciousness_sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    
    # Data type and source
    data_type = db.Column(db.Enum('eeg', 'fnirs', 'ecg', 'gsr', 'eye_tracking', name='neural_data_types'), nullable=False)
    device_id = db.Column(db.String(100), nullable=False)
    sampling_rate = db.Column(db.Integer, nullable=False)
    
    # EEG-specific data
    eeg_channels = db.Column(JSONB, nullable=True)  # Channel names and positions
    eeg_raw_data = db.Column(JSONB, nullable=True)  # Raw voltage values
    eeg_processed_data = db.Column(JSONB, nullable=True)  # Filtered and processed signals
    eeg_power_spectrum = db.Column(JSONB, nullable=True)  # Frequency domain analysis
    eeg_connectivity = db.Column(JSONB, nullable=True)  # Inter-channel connectivity
    
    # fNIRS-specific data
    fnirs_channels = db.Column(JSONB, nullable=True)  # Optode positions
    fnirs_oxy_hb = db.Column(JSONB, nullable=True)  # Oxygenated hemoglobin
    fnirs_deoxy_hb = db.Column(JSONB, nullable=True)  # Deoxygenated hemoglobin
    fnirs_total_hb = db.Column(JSONB, nullable=True)  # Total hemoglobin
    fnirs_signal_quality = db.Column(JSONB, nullable=True)  # Signal quality metrics
    
    # Signal quality metrics
    signal_to_noise_ratio = db.Column(db.Float, nullable=True)
    artifact_probability = db.Column(db.Float, nullable=True)
    data_completeness = db.Column(db.Float, nullable=True)
    
    # Processing metadata
    processing_pipeline = db.Column(JSONB, nullable=True)  # Applied filters and algorithms
    processing_timestamp = db.Column(db.DateTime(timezone=True), nullable=True)
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_neural_session_time', 'session_id', 'timestamp'),
        Index('idx_neural_type_time', 'data_type', 'timestamp'),
        Index('idx_neural_device_time', 'device_id', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type,
            'device_id': self.device_id,
            'sampling_rate': self.sampling_rate,
            'eeg_power_spectrum': self.eeg_power_spectrum,
            'fnirs_oxy_hb': self.fnirs_oxy_hb,
            'signal_to_noise_ratio': self.signal_to_noise_ratio,
            'data_completeness': self.data_completeness
        }

class ConsciousnessState(db.Model):
    """
    Consciousness state analysis and classification
    Tracks moment-to-moment consciousness levels and characteristics
    """
    __tablename__ = 'consciousness_states'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = db.Column(UUID(as_uuid=True), db.ForeignKey('consciousness_sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    
    # Consciousness dimensions
    consciousness_level = db.Column(db.String(50), nullable=False)  # AWAKENING, AWARE, TRANSCENDENT, etc.
    consciousness_depth = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    consciousness_clarity = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    consciousness_stability = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    
    # Awareness components
    self_awareness = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    environmental_awareness = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    temporal_awareness = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    metacognitive_awareness = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Transcendence indicators
    unity_experience = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    transcendence_level = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    ego_dissolution = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    
    # Flow state components
    flow_state_intensity = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    challenge_skill_balance = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    action_awareness_merge = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    clear_goals = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    immediate_feedback = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    time_transformation = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    autotelic_experience = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Confidence and reliability
    classification_confidence = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    model_version = db.Column(db.String(50), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_consciousness_session_time', 'session_id', 'timestamp'),
        Index('idx_consciousness_level_time', 'consciousness_level', 'timestamp'),
        CheckConstraint('consciousness_depth >= 0 AND consciousness_depth <= 1', name='check_depth_range'),
        CheckConstraint('consciousness_clarity >= 0 AND consciousness_clarity <= 1', name='check_clarity_range'),
        CheckConstraint('flow_state_intensity >= 0 AND flow_state_intensity <= 1', name='check_flow_range'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'timestamp': self.timestamp.isoformat(),
            'consciousness_level': self.consciousness_level,
            'consciousness_depth': self.consciousness_depth,
            'consciousness_clarity': self.consciousness_clarity,
            'consciousness_stability': self.consciousness_stability,
            'self_awareness': self.self_awareness,
            'unity_experience': self.unity_experience,
            'transcendence_level': self.transcendence_level,
            'flow_state_intensity': self.flow_state_intensity,
            'challenge_skill_balance': self.challenge_skill_balance,
            'classification_confidence': self.classification_confidence
        }

class CognitiveAssessment(db.Model):
    """
    Cognitive performance and capability assessment
    Tracks various cognitive functions and their performance metrics
    """
    __tablename__ = 'cognitive_assessments'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = db.Column(UUID(as_uuid=True), db.ForeignKey('consciousness_sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    
    # Assessment metadata
    assessment_type = db.Column(db.String(100), nullable=False)  # attention, memory, processing, etc.
    assessment_duration = db.Column(db.Integer, nullable=True)  # Duration in seconds
    
    # Attention systems
    sustained_attention = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    selective_attention = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    divided_attention = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    executive_attention = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    attention_stability = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Memory systems
    working_memory_capacity = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    working_memory_efficiency = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    episodic_memory = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    semantic_memory = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    procedural_memory = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Processing capabilities
    processing_speed = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    processing_accuracy = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    cognitive_flexibility = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    inhibitory_control = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    task_switching = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Executive functions
    planning_ability = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    decision_making = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    problem_solving = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    abstract_reasoning = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Creative intelligence
    divergent_thinking = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    convergent_thinking = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    originality = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    fluency = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    flexibility = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    elaboration = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Emotional intelligence
    emotional_awareness = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    emotional_regulation = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    empathy = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    social_cognition = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Assessment reliability
    confidence_score = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    assessment_version = db.Column(db.String(50), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_cognitive_session_time', 'session_id', 'timestamp'),
        Index('idx_cognitive_type_time', 'assessment_type', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'timestamp': self.timestamp.isoformat(),
            'assessment_type': self.assessment_type,
            'sustained_attention': self.sustained_attention,
            'working_memory_capacity': self.working_memory_capacity,
            'processing_speed': self.processing_speed,
            'divergent_thinking': self.divergent_thinking,
            'emotional_regulation': self.emotional_regulation,
            'confidence_score': self.confidence_score
        }

class QuantumProcessingMetrics(db.Model):
    """
    Quantum cognitive processing metrics and performance data
    Tracks quantum-enhanced cognitive computations and their effectiveness
    """
    __tablename__ = 'quantum_processing_metrics'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = db.Column(UUID(as_uuid=True), db.ForeignKey('consciousness_sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    
    # Quantum system metrics
    active_qubits = db.Column(db.Integer, nullable=False)
    coherence_time_microseconds = db.Column(db.Float, nullable=False)
    entanglement_fidelity = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    gate_error_rate = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    decoherence_rate = db.Column(db.Float, nullable=False)
    
    # Quantum advantage metrics
    quantum_advantage_factor = db.Column(db.Float, nullable=True)  # Speedup over classical
    quantum_operations_completed = db.Column(db.BigInteger, nullable=False)
    quantum_circuit_depth = db.Column(db.Integer, nullable=True)
    quantum_volume = db.Column(db.Integer, nullable=True)
    
    # Quantum cognitive processing
    superposition_states_utilized = db.Column(db.Integer, nullable=True)
    entanglement_operations = db.Column(db.Integer, nullable=True)
    quantum_interference_patterns = db.Column(JSONB, nullable=True)
    
    # Performance metrics
    classical_processing_time_ms = db.Column(db.Float, nullable=True)
    quantum_processing_time_ms = db.Column(db.Float, nullable=True)
    hybrid_processing_efficiency = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Error correction and reliability
    error_correction_overhead = db.Column(db.Float, nullable=True)
    logical_error_rate = db.Column(db.Float, nullable=True)
    quantum_state_fidelity = db.Column(db.Float, nullable=True)  # 0.0 to 1.0
    
    # Indexes
    __table_args__ = (
        Index('idx_quantum_session_time', 'session_id', 'timestamp'),
        CheckConstraint('entanglement_fidelity >= 0 AND entanglement_fidelity <= 1', name='check_entanglement_range'),
        CheckConstraint('gate_error_rate >= 0 AND gate_error_rate <= 1', name='check_gate_error_range'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'timestamp': self.timestamp.isoformat(),
            'active_qubits': self.active_qubits,
            'coherence_time_microseconds': self.coherence_time_microseconds,
            'entanglement_fidelity': self.entanglement_fidelity,
            'gate_error_rate': self.gate_error_rate,
            'quantum_advantage_factor': self.quantum_advantage_factor,
            'quantum_operations_completed': self.quantum_operations_completed,
            'quantum_processing_time_ms': self.quantum_processing_time_ms,
            'hybrid_processing_efficiency': self.hybrid_processing_efficiency
        }

class User(db.Model):
    """
    Enhanced user model with comprehensive profile and preferences
    """
    __tablename__ = 'users'
    
    # Primary identifiers
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    # Authentication
    password_hash = db.Column(db.String(255), nullable=False)
    salt = db.Column(db.String(255), nullable=False)
    
    # Profile information
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    timezone = db.Column(db.String(50), default='UTC')
    
    # Account metadata
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    
    # Consciousness profile
    baseline_consciousness_level = db.Column(db.String(50), nullable=True)
    consciousness_goals = db.Column(JSONB, nullable=True)  # User's consciousness development goals
    meditation_experience_years = db.Column(db.Float, nullable=True)
    
    # Preferences and settings
    preferred_session_types = db.Column(JSONB, nullable=True)  # Array of preferred session types
    notification_preferences = db.Column(JSONB, nullable=True)
    privacy_settings = db.Column(JSONB, nullable=True)
    
    # Relationships
    sessions = db.relationship('ConsciousnessSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_created', 'created_at'),
    )
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'timezone': self.timezone,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'baseline_consciousness_level': self.baseline_consciousness_level,
            'consciousness_goals': self.consciousness_goals,
            'preferred_session_types': self.preferred_session_types
        }

