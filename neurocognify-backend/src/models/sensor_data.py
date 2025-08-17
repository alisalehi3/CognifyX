"""
NeuroCognify Sensor Data Models
Comprehensive data models for EEG, fNIRS, and biometric sensor data
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from typing import Dict, List, Optional

db = SQLAlchemy()

class Session(db.Model):
    """User session model for tracking neurofeedback sessions"""
    __tablename__ = 'sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    session_type = db.Column(db.String(50), default='general')  # general, meditation, focus, etc.
    
    # Session quality metrics
    data_quality_score = db.Column(db.Float, default=0.0)
    artifact_percentage = db.Column(db.Float, default=0.0)
    signal_to_noise_ratio = db.Column(db.Float, default=0.0)
    
    # Session outcomes
    presence_score = db.Column(db.Float, default=0.0)
    agency_sensitivity_avg = db.Column(db.Float, default=0.0)
    cognitive_load_avg = db.Column(db.Float, default=0.0)
    
    # Relationships
    eeg_data = db.relationship('EEGData', backref='session', lazy=True, cascade='all, delete-orphan')
    fnirs_data = db.relationship('fNIRSData', backref='session', lazy=True, cascade='all, delete-orphan')
    cognitive_states = db.relationship('CognitiveState', backref='session', lazy=True, cascade='all, delete-orphan')
    interventions = db.relationship('Intervention', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'session_type': self.session_type,
            'data_quality_score': self.data_quality_score,
            'presence_score': self.presence_score,
            'agency_sensitivity_avg': self.agency_sensitivity_avg,
            'cognitive_load_avg': self.cognitive_load_avg
        }

class EEGData(db.Model):
    """EEG sensor data model for brainwave recordings"""
    __tablename__ = 'eeg_data'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Raw EEG channels (Muse S has 4 electrodes: TP9, AF7, AF8, TP10)
    tp9 = db.Column(db.Float)  # Left ear
    af7 = db.Column(db.Float)  # Left forehead
    af8 = db.Column(db.Float)  # Right forehead
    tp10 = db.Column(db.Float) # Right ear
    
    # Processed band powers (μV²)
    delta_power = db.Column(db.Float)    # 0.5-4 Hz
    theta_power = db.Column(db.Float)    # 4-8 Hz
    alpha_power = db.Column(db.Float)    # 8-13 Hz
    beta_power = db.Column(db.Float)     # 13-30 Hz
    gamma_power = db.Column(db.Float)    # 30-100 Hz
    
    # Derived metrics
    alpha_theta_ratio = db.Column(db.Float)
    beta_alpha_ratio = db.Column(db.Float)
    engagement_index = db.Column(db.Float)
    meditation_index = db.Column(db.Float)
    
    # Data quality indicators
    is_artifact = db.Column(db.Boolean, default=False)
    signal_quality = db.Column(db.Float, default=1.0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'channels': {
                'tp9': self.tp9,
                'af7': self.af7,
                'af8': self.af8,
                'tp10': self.tp10
            },
            'band_powers': {
                'delta': self.delta_power,
                'theta': self.theta_power,
                'alpha': self.alpha_power,
                'beta': self.beta_power,
                'gamma': self.gamma_power
            },
            'derived_metrics': {
                'alpha_theta_ratio': self.alpha_theta_ratio,
                'beta_alpha_ratio': self.beta_alpha_ratio,
                'engagement_index': self.engagement_index,
                'meditation_index': self.meditation_index
            },
            'quality': {
                'is_artifact': self.is_artifact,
                'signal_quality': self.signal_quality
            }
        }

class fNIRSData(db.Model):
    """fNIRS sensor data model for hemodynamic responses"""
    __tablename__ = 'fnirs_data'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Hemoglobin concentrations (μM)
    hbo2_left = db.Column(db.Float)      # Oxygenated Hb - Left PFC
    hbo2_right = db.Column(db.Float)     # Oxygenated Hb - Right PFC
    hb_left = db.Column(db.Float)        # Deoxygenated Hb - Left PFC
    hb_right = db.Column(db.Float)       # Deoxygenated Hb - Right PFC
    
    # Total hemoglobin
    hbt_left = db.Column(db.Float)       # Total Hb - Left PFC
    hbt_right = db.Column(db.Float)      # Total Hb - Right PFC
    
    # Derived metrics
    oxygenation_index_left = db.Column(db.Float)   # HbO2/(HbO2+Hb) - Left
    oxygenation_index_right = db.Column(db.Float)  # HbO2/(HbO2+Hb) - Right
    laterality_index = db.Column(db.Float)         # (Right-Left)/(Right+Left)
    
    # Cognitive load indicators
    cognitive_effort = db.Column(db.Float)         # Based on PFC activation
    mental_fatigue = db.Column(db.Float)           # Based on sustained activation
    
    # Data quality
    signal_quality = db.Column(db.Float, default=1.0)
    motion_artifact = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'hemoglobin': {
                'hbo2_left': self.hbo2_left,
                'hbo2_right': self.hbo2_right,
                'hb_left': self.hb_left,
                'hb_right': self.hb_right,
                'hbt_left': self.hbt_left,
                'hbt_right': self.hbt_right
            },
            'derived_metrics': {
                'oxygenation_index_left': self.oxygenation_index_left,
                'oxygenation_index_right': self.oxygenation_index_right,
                'laterality_index': self.laterality_index,
                'cognitive_effort': self.cognitive_effort,
                'mental_fatigue': self.mental_fatigue
            },
            'quality': {
                'signal_quality': self.signal_quality,
                'motion_artifact': self.motion_artifact
            }
        }

class CognitiveState(db.Model):
    """AI-predicted cognitive states and agency sensitivity"""
    __tablename__ = 'cognitive_states'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Core cognitive states (0-1 scale)
    focus_level = db.Column(db.Float, default=0.0)
    relaxation_level = db.Column(db.Float, default=0.0)
    presence_score = db.Column(db.Float, default=0.0)
    cognitive_load = db.Column(db.Float, default=0.0)
    
    # Agency detection system
    agency_sensitivity = db.Column(db.Float, default=0.5)    # 0=under-sensitive, 1=over-sensitive
    agency_threshold = db.Column(db.Float, default=0.5)      # Dynamic threshold
    pattern_detection_bias = db.Column(db.Float, default=0.0)
    
    # Emotional states
    emotional_valence = db.Column(db.Float, default=0.0)     # -1=negative, +1=positive
    emotional_arousal = db.Column(db.Float, default=0.0)     # 0=calm, 1=excited
    stress_level = db.Column(db.Float, default=0.0)
    
    # Flow state indicators
    flow_state_probability = db.Column(db.Float, default=0.0)
    challenge_skill_balance = db.Column(db.Float, default=0.0)
    time_distortion = db.Column(db.Float, default=0.0)
    
    # Resilience metrics
    mental_resilience = db.Column(db.Float, default=0.5)
    adaptability_score = db.Column(db.Float, default=0.5)
    recovery_rate = db.Column(db.Float, default=0.5)
    
    # AI model confidence
    prediction_confidence = db.Column(db.Float, default=0.0)
    model_version = db.Column(db.String(20), default='v1.0')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'cognitive_states': {
                'focus_level': self.focus_level,
                'relaxation_level': self.relaxation_level,
                'presence_score': self.presence_score,
                'cognitive_load': self.cognitive_load
            },
            'agency_system': {
                'agency_sensitivity': self.agency_sensitivity,
                'agency_threshold': self.agency_threshold,
                'pattern_detection_bias': self.pattern_detection_bias
            },
            'emotional_states': {
                'valence': self.emotional_valence,
                'arousal': self.emotional_arousal,
                'stress_level': self.stress_level
            },
            'flow_state': {
                'probability': self.flow_state_probability,
                'challenge_skill_balance': self.challenge_skill_balance,
                'time_distortion': self.time_distortion
            },
            'resilience': {
                'mental_resilience': self.mental_resilience,
                'adaptability_score': self.adaptability_score,
                'recovery_rate': self.recovery_rate
            },
            'meta': {
                'prediction_confidence': self.prediction_confidence,
                'model_version': self.model_version
            }
        }

class BiometricData(db.Model):
    """Additional biometric sensor data (HRV, GSR, etc.)"""
    __tablename__ = 'biometric_data'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Heart rate variability
    heart_rate = db.Column(db.Float)
    rr_interval = db.Column(db.Float)
    hrv_rmssd = db.Column(db.Float)      # Root mean square of successive differences
    hrv_pnn50 = db.Column(db.Float)      # Percentage of successive RR intervals > 50ms
    
    # Galvanic skin response
    gsr_conductance = db.Column(db.Float)
    gsr_resistance = db.Column(db.Float)
    
    # Respiratory data
    breathing_rate = db.Column(db.Float)
    breathing_depth = db.Column(db.Float)
    breathing_variability = db.Column(db.Float)
    
    # Environmental context
    ambient_temperature = db.Column(db.Float)
    ambient_light = db.Column(db.Float)
    noise_level = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'heart_rate_variability': {
                'heart_rate': self.heart_rate,
                'rr_interval': self.rr_interval,
                'hrv_rmssd': self.hrv_rmssd,
                'hrv_pnn50': self.hrv_pnn50
            },
            'galvanic_skin_response': {
                'conductance': self.gsr_conductance,
                'resistance': self.gsr_resistance
            },
            'respiratory': {
                'breathing_rate': self.breathing_rate,
                'breathing_depth': self.breathing_depth,
                'breathing_variability': self.breathing_variability
            },
            'environment': {
                'temperature': self.ambient_temperature,
                'light': self.ambient_light,
                'noise_level': self.noise_level
            }
        }

class Intervention(db.Model):
    """AI-recommended interventions and user responses"""
    __tablename__ = 'interventions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Intervention details
    intervention_type = db.Column(db.String(50), nullable=False)  # breathing, story, meditation, etc.
    intervention_id = db.Column(db.String(100))  # Reference to specific content
    trigger_reason = db.Column(db.String(200))   # Why this intervention was recommended
    
    # Intervention parameters
    duration_seconds = db.Column(db.Integer)
    intensity_level = db.Column(db.Float, default=0.5)  # 0=gentle, 1=intense
    personalization_score = db.Column(db.Float, default=0.5)
    
    # User response
    user_accepted = db.Column(db.Boolean, default=False)
    user_completed = db.Column(db.Boolean, default=False)
    completion_percentage = db.Column(db.Float, default=0.0)
    user_rating = db.Column(db.Integer)  # 1-5 stars
    user_feedback = db.Column(db.Text)
    
    # Effectiveness metrics
    pre_intervention_state = db.Column(db.Text)  # JSON of cognitive state before
    post_intervention_state = db.Column(db.Text) # JSON of cognitive state after
    effectiveness_score = db.Column(db.Float)    # Calculated improvement
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'intervention': {
                'type': self.intervention_type,
                'id': self.intervention_id,
                'trigger_reason': self.trigger_reason,
                'duration_seconds': self.duration_seconds,
                'intensity_level': self.intensity_level,
                'personalization_score': self.personalization_score
            },
            'user_response': {
                'accepted': self.user_accepted,
                'completed': self.user_completed,
                'completion_percentage': self.completion_percentage,
                'rating': self.user_rating,
                'feedback': self.user_feedback
            },
            'effectiveness': {
                'pre_state': json.loads(self.pre_intervention_state) if self.pre_intervention_state else None,
                'post_state': json.loads(self.post_intervention_state) if self.post_intervention_state else None,
                'effectiveness_score': self.effectiveness_score
            }
        }

class DeviceCalibration(db.Model):
    """Device calibration and baseline data"""
    __tablename__ = 'device_calibrations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    device_id = db.Column(db.String(100), nullable=False)  # Muse device identifier
    calibration_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # EEG baselines
    baseline_alpha = db.Column(db.Float)
    baseline_beta = db.Column(db.Float)
    baseline_theta = db.Column(db.Float)
    baseline_delta = db.Column(db.Float)
    baseline_gamma = db.Column(db.Float)
    
    # fNIRS baselines
    baseline_hbo2 = db.Column(db.Float)
    baseline_hb = db.Column(db.Float)
    
    # Individual thresholds
    focus_threshold = db.Column(db.Float, default=0.5)
    relaxation_threshold = db.Column(db.Float, default=0.5)
    agency_sensitivity_baseline = db.Column(db.Float, default=0.5)
    
    # Calibration quality
    calibration_quality = db.Column(db.Float, default=0.0)
    calibration_notes = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'device_id': self.device_id,
            'calibration_date': self.calibration_date.isoformat(),
            'eeg_baselines': {
                'alpha': self.baseline_alpha,
                'beta': self.baseline_beta,
                'theta': self.baseline_theta,
                'delta': self.baseline_delta,
                'gamma': self.baseline_gamma
            },
            'fnirs_baselines': {
                'hbo2': self.baseline_hbo2,
                'hb': self.baseline_hb
            },
            'thresholds': {
                'focus': self.focus_threshold,
                'relaxation': self.relaxation_threshold,
                'agency_sensitivity': self.agency_sensitivity_baseline
            },
            'quality': {
                'calibration_quality': self.calibration_quality,
                'notes': self.calibration_notes
            }
        }

