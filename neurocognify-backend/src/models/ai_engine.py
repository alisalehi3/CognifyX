"""
NeuroCognify AI Engine Models
Advanced machine learning models for cognitive state prediction and agency detection
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from flask_sqlalchemy import SQLAlchemy
from src.models.sensor_data import db

logger = logging.getLogger(__name__)

@dataclass
class CognitiveStateVector:
    """Data structure for cognitive state representation"""
    focus_level: float
    relaxation_level: float
    presence_score: float
    cognitive_load: float
    agency_sensitivity: float
    emotional_valence: float
    emotional_arousal: float
    stress_level: float
    flow_state_probability: float
    mental_resilience: float
    timestamp: datetime

@dataclass
class MultimodalFeatures:
    """Multimodal feature vector for AI model input"""
    eeg_features: Dict[str, float]
    fnirs_features: Dict[str, float]
    biometric_features: Dict[str, float]
    contextual_features: Dict[str, float]
    temporal_features: Dict[str, float]

class AgencySensitivityCalculator:
    """Advanced agency sensitivity index calculation"""
    
    def __init__(self):
        self.baseline_patterns = {}
        self.context_weights = {
            'stress_level': 0.3,
            'cognitive_load': 0.2,
            'environmental_uncertainty': 0.2,
            'social_context': 0.15,
            'recent_events': 0.15
        }
        self.mismatch_negativity_detector = MismatchNegativityDetector()
        self.anticipatory_activity_analyzer = AnticipatoryActivityAnalyzer()
    
    def calculate_agency_sensitivity_index(self, 
                                         eeg_data: Dict, 
                                         fnirs_data: Dict, 
                                         context: Dict,
                                         user_baseline: Optional[Dict] = None) -> float:
        """
        Calculate the Agency Sensitivity Index (ASI) based on neurophysiological data
        
        ASI ranges from 0 (under-sensitive) to 1 (over-sensitive)
        Optimal range is typically 0.4-0.6
        """
        try:
            # Detect mismatch negativity (MMN) - indicates prediction error processing
            mmn_amplitude = self.mismatch_negativity_detector.detect_mmn(eeg_data)
            
            # Measure anticipatory brain activity
            anticipatory_activity = self.anticipatory_activity_analyzer.analyze_anticipation(eeg_data)
            
            # Analyze prefrontal cortex activation patterns
            pfc_activation = self._analyze_pfc_activation(fnirs_data)
            
            # Calculate base ASI from neurophysiological markers
            base_asi = self._calculate_base_asi(mmn_amplitude, anticipatory_activity, pfc_activation)
            
            # Apply contextual adjustments
            context_adjusted_asi = self._apply_contextual_adjustments(base_asi, context)
            
            # Personalize based on user baseline
            if user_baseline:
                personalized_asi = self._personalize_asi(context_adjusted_asi, user_baseline)
            else:
                personalized_asi = context_adjusted_asi
            
            # Ensure ASI is within valid range
            final_asi = np.clip(personalized_asi, 0.0, 1.0)
            
            logger.debug(f"Calculated ASI: {final_asi:.3f} (base: {base_asi:.3f}, context: {context_adjusted_asi:.3f})")
            
            return final_asi
            
        except Exception as e:
            logger.error(f"Error calculating agency sensitivity index: {str(e)}")
            return 0.5  # Return neutral value on error
    
    def _analyze_pfc_activation(self, fnirs_data: Dict) -> Dict[str, float]:
        """Analyze prefrontal cortex activation patterns"""
        pfc_metrics = {}
        
        # Calculate bilateral PFC activation
        left_activation = fnirs_data.get('hbo2_left', 0) - fnirs_data.get('hb_left', 0)
        right_activation = fnirs_data.get('hbo2_right', 0) - fnirs_data.get('hb_right', 0)
        
        pfc_metrics['left_activation'] = left_activation
        pfc_metrics['right_activation'] = right_activation
        pfc_metrics['bilateral_activation'] = (left_activation + right_activation) / 2
        pfc_metrics['laterality'] = (right_activation - left_activation) / (abs(right_activation) + abs(left_activation) + 1e-10)
        
        return pfc_metrics
    
    def _calculate_base_asi(self, mmn_amplitude: float, anticipatory_activity: float, pfc_activation: Dict) -> float:
        """Calculate base ASI from neurophysiological markers"""
        # Higher MMN amplitude indicates stronger prediction error processing
        mmn_component = np.clip(mmn_amplitude / 10.0, 0, 1)  # Normalize to 0-1
        
        # Higher anticipatory activity indicates hypervigilance
        anticipatory_component = np.clip(anticipatory_activity / 5.0, 0, 1)
        
        # PFC activation patterns related to cognitive control
        pfc_component = np.clip(pfc_activation['bilateral_activation'] / 20.0, 0, 1)
        
        # Weighted combination
        base_asi = (0.4 * mmn_component + 0.3 * anticipatory_component + 0.3 * pfc_component)
        
        return base_asi
    
    def _apply_contextual_adjustments(self, base_asi: float, context: Dict) -> float:
        """Apply contextual adjustments to ASI"""
        adjustment_factor = 0.0
        
        # Stress increases agency sensitivity
        stress_level = context.get('stress_level', 0.0)
        adjustment_factor += self.context_weights['stress_level'] * stress_level
        
        # High cognitive load can increase false pattern detection
        cognitive_load = context.get('cognitive_load', 0.0)
        adjustment_factor += self.context_weights['cognitive_load'] * cognitive_load
        
        # Environmental uncertainty increases vigilance
        uncertainty = context.get('environmental_uncertainty', 0.0)
        adjustment_factor += self.context_weights['environmental_uncertainty'] * uncertainty
        
        # Social context can modulate agency detection
        social_stress = context.get('social_stress', 0.0)
        adjustment_factor += self.context_weights['social_context'] * social_stress
        
        # Recent negative events increase sensitivity
        recent_events_impact = context.get('recent_events_impact', 0.0)
        adjustment_factor += self.context_weights['recent_events'] * recent_events_impact
        
        # Apply adjustment (can increase or decrease ASI)
        adjusted_asi = base_asi + (adjustment_factor - 0.5) * 0.3  # Scale adjustment
        
        return adjusted_asi
    
    def _personalize_asi(self, asi: float, user_baseline: Dict) -> float:
        """Personalize ASI based on user's historical patterns"""
        baseline_asi = user_baseline.get('baseline_agency_sensitivity', 0.5)
        sensitivity_factor = user_baseline.get('sensitivity_factor', 1.0)
        
        # Adjust based on personal baseline
        deviation = asi - baseline_asi
        personalized_asi = baseline_asi + (deviation * sensitivity_factor)
        
        return personalized_asi

class MismatchNegativityDetector:
    """Detects mismatch negativity (MMN) in EEG signals"""
    
    def __init__(self):
        self.mmn_window = (100, 250)  # MMN typically occurs 100-250ms after stimulus
        self.baseline_window = (-100, 0)  # Pre-stimulus baseline
    
    def detect_mmn(self, eeg_data: Dict) -> float:
        """Detect mismatch negativity amplitude"""
        try:
            # Focus on frontal electrodes (AF7, AF8) for MMN detection
            frontal_channels = ['af7', 'af8']
            mmn_amplitudes = []
            
            for channel in frontal_channels:
                if channel in eeg_data and isinstance(eeg_data[channel], list):
                    signal = np.array(eeg_data[channel])
                    
                    # Simple MMN detection based on negative deflection
                    # In real implementation, this would use event-related potentials
                    if len(signal) > 100:
                        # Look for negative peaks in the MMN time window
                        mmn_region = signal[-100:]  # Last 100 samples as proxy
                        baseline = np.mean(signal[:-100]) if len(signal) > 200 else 0
                        
                        negative_deflection = baseline - np.min(mmn_region)
                        mmn_amplitudes.append(max(0, negative_deflection))
            
            return np.mean(mmn_amplitudes) if mmn_amplitudes else 0.0
            
        except Exception as e:
            logger.error(f"Error detecting MMN: {str(e)}")
            return 0.0

class AnticipatoryActivityAnalyzer:
    """Analyzes anticipatory brain activity patterns"""
    
    def __init__(self):
        self.anticipatory_bands = {
            'theta': (4, 8),    # Anticipatory theta
            'alpha': (8, 13),   # Alpha suppression
            'beta': (13, 30)    # Beta enhancement
        }
    
    def analyze_anticipation(self, eeg_data: Dict) -> float:
        """Analyze anticipatory activity patterns"""
        try:
            anticipatory_score = 0.0
            
            # Calculate band powers for anticipatory analysis
            for channel, signal in eeg_data.items():
                if isinstance(signal, list) and len(signal) > 250:
                    signal_array = np.array(signal)
                    
                    # Calculate power spectral density
                    fft = np.fft.fft(signal_array)
                    freqs = np.fft.fftfreq(len(signal_array), 1/250)  # 250Hz sampling
                    power_spectrum = np.abs(fft) ** 2
                    
                    # Calculate anticipatory markers
                    theta_power = self._calculate_band_power(power_spectrum, freqs, 4, 8)
                    alpha_power = self._calculate_band_power(power_spectrum, freqs, 8, 13)
                    beta_power = self._calculate_band_power(power_spectrum, freqs, 13, 30)
                    
                    # Anticipatory pattern: increased theta, decreased alpha, increased beta
                    anticipatory_pattern = (theta_power + beta_power) / (alpha_power + 1e-10)
                    anticipatory_score += anticipatory_pattern
            
            # Average across channels
            num_channels = len([ch for ch, sig in eeg_data.items() if isinstance(sig, list) and len(sig) > 250])
            if num_channels > 0:
                anticipatory_score /= num_channels
            
            return anticipatory_score
            
        except Exception as e:
            logger.error(f"Error analyzing anticipatory activity: {str(e)}")
            return 0.0
    
    def _calculate_band_power(self, power_spectrum: np.ndarray, freqs: np.ndarray, 
                            low_freq: float, high_freq: float) -> float:
        """Calculate power in a specific frequency band"""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.mean(power_spectrum[band_mask])

class CognitiveStatePredictor:
    """Advanced cognitive state prediction using multimodal data"""
    
    def __init__(self):
        self.feature_extractor = MultimodalFeatureExtractor()
        self.state_classifier = NeuralStateClassifier()
        self.flow_state_detector = FlowStateDetector()
        self.resilience_calculator = ResilienceCalculator()
    
    def predict_cognitive_state(self, 
                              eeg_data: Dict, 
                              fnirs_data: Dict, 
                              biometric_data: Dict,
                              context: Dict,
                              user_profile: Optional[Dict] = None) -> CognitiveStateVector:
        """Predict comprehensive cognitive state from multimodal data"""
        try:
            # Extract multimodal features
            features = self.feature_extractor.extract_features(
                eeg_data, fnirs_data, biometric_data, context
            )
            
            # Predict core cognitive states
            focus_level = self._predict_focus_level(features)
            relaxation_level = self._predict_relaxation_level(features)
            presence_score = self._predict_presence_score(features)
            cognitive_load = self._predict_cognitive_load(features)
            
            # Calculate agency sensitivity
            agency_calculator = AgencySensitivityCalculator()
            agency_sensitivity = agency_calculator.calculate_agency_sensitivity_index(
                eeg_data, fnirs_data, context, user_profile
            )
            
            # Predict emotional states
            emotional_valence = self._predict_emotional_valence(features)
            emotional_arousal = self._predict_emotional_arousal(features)
            stress_level = self._predict_stress_level(features)
            
            # Detect flow state
            flow_state_probability = self.flow_state_detector.detect_flow_state(features)
            
            # Calculate resilience
            mental_resilience = self.resilience_calculator.calculate_resilience(features, context)
            
            # Create cognitive state vector
            cognitive_state = CognitiveStateVector(
                focus_level=focus_level,
                relaxation_level=relaxation_level,
                presence_score=presence_score,
                cognitive_load=cognitive_load,
                agency_sensitivity=agency_sensitivity,
                emotional_valence=emotional_valence,
                emotional_arousal=emotional_arousal,
                stress_level=stress_level,
                flow_state_probability=flow_state_probability,
                mental_resilience=mental_resilience,
                timestamp=datetime.utcnow()
            )
            
            return cognitive_state
            
        except Exception as e:
            logger.error(f"Error predicting cognitive state: {str(e)}")
            # Return neutral state on error
            return CognitiveStateVector(
                focus_level=0.5, relaxation_level=0.5, presence_score=0.5,
                cognitive_load=0.5, agency_sensitivity=0.5, emotional_valence=0.0,
                emotional_arousal=0.5, stress_level=0.5, flow_state_probability=0.0,
                mental_resilience=0.5, timestamp=datetime.utcnow()
            )
    
    def _predict_focus_level(self, features: MultimodalFeatures) -> float:
        """Predict focus level from multimodal features"""
        # Focus correlates with beta/alpha ratio and PFC activation
        beta_alpha_ratio = features.eeg_features.get('beta_alpha_ratio', 1.0)
        pfc_activation = features.fnirs_features.get('bilateral_activation', 0.0)
        
        # Normalize and combine
        focus_eeg = np.clip(beta_alpha_ratio / 3.0, 0, 1)
        focus_fnirs = np.clip((pfc_activation + 10) / 20, 0, 1)
        
        focus_level = 0.7 * focus_eeg + 0.3 * focus_fnirs
        return np.clip(focus_level, 0.0, 1.0)
    
    def _predict_relaxation_level(self, features: MultimodalFeatures) -> float:
        """Predict relaxation level from multimodal features"""
        # Relaxation correlates with alpha power and low stress markers
        alpha_power = features.eeg_features.get('alpha_power', 0.0)
        hrv = features.biometric_features.get('hrv_rmssd', 30.0)
        
        # Normalize and combine
        relaxation_eeg = np.clip(alpha_power / 50.0, 0, 1)
        relaxation_hrv = np.clip(hrv / 100.0, 0, 1)
        
        relaxation_level = 0.6 * relaxation_eeg + 0.4 * relaxation_hrv
        return np.clip(relaxation_level, 0.0, 1.0)
    
    def _predict_presence_score(self, features: MultimodalFeatures) -> float:
        """Predict mindful presence score"""
        # Presence correlates with theta/alpha balance and sustained attention
        theta_power = features.eeg_features.get('theta_power', 0.0)
        alpha_power = features.eeg_features.get('alpha_power', 0.0)
        attention_stability = features.temporal_features.get('attention_stability', 0.5)
        
        theta_alpha_balance = theta_power / (alpha_power + 1e-10)
        presence_neural = np.clip(theta_alpha_balance / 2.0, 0, 1)
        
        presence_score = 0.5 * presence_neural + 0.5 * attention_stability
        return np.clip(presence_score, 0.0, 1.0)
    
    def _predict_cognitive_load(self, features: MultimodalFeatures) -> float:
        """Predict cognitive load level"""
        # Cognitive load correlates with PFC activation and theta power
        pfc_activation = features.fnirs_features.get('bilateral_activation', 0.0)
        theta_power = features.eeg_features.get('theta_power', 0.0)
        
        load_fnirs = np.clip((pfc_activation + 5) / 15, 0, 1)
        load_eeg = np.clip(theta_power / 30.0, 0, 1)
        
        cognitive_load = 0.6 * load_fnirs + 0.4 * load_eeg
        return np.clip(cognitive_load, 0.0, 1.0)
    
    def _predict_emotional_valence(self, features: MultimodalFeatures) -> float:
        """Predict emotional valence (-1 negative, +1 positive)"""
        # Valence correlates with frontal alpha asymmetry
        left_alpha = features.eeg_features.get('left_alpha', 0.0)
        right_alpha = features.eeg_features.get('right_alpha', 0.0)
        
        # Asymmetry score (positive = more positive emotion)
        asymmetry = (right_alpha - left_alpha) / (right_alpha + left_alpha + 1e-10)
        
        # Convert to -1 to +1 scale
        valence = np.clip(asymmetry * 2, -1.0, 1.0)
        return valence
    
    def _predict_emotional_arousal(self, features: MultimodalFeatures) -> float:
        """Predict emotional arousal level"""
        # Arousal correlates with beta power and heart rate
        beta_power = features.eeg_features.get('beta_power', 0.0)
        heart_rate = features.biometric_features.get('heart_rate', 70.0)
        
        arousal_eeg = np.clip(beta_power / 40.0, 0, 1)
        arousal_hr = np.clip((heart_rate - 60) / 40, 0, 1)
        
        arousal = 0.6 * arousal_eeg + 0.4 * arousal_hr
        return np.clip(arousal, 0.0, 1.0)
    
    def _predict_stress_level(self, features: MultimodalFeatures) -> float:
        """Predict stress level"""
        # Stress correlates with high beta, low HRV, high GSR
        beta_power = features.eeg_features.get('beta_power', 0.0)
        hrv = features.biometric_features.get('hrv_rmssd', 50.0)
        gsr = features.biometric_features.get('gsr_conductance', 5.0)
        
        stress_eeg = np.clip(beta_power / 50.0, 0, 1)
        stress_hrv = np.clip((100 - hrv) / 100, 0, 1)  # Lower HRV = higher stress
        stress_gsr = np.clip(gsr / 20.0, 0, 1)
        
        stress_level = 0.4 * stress_eeg + 0.3 * stress_hrv + 0.3 * stress_gsr
        return np.clip(stress_level, 0.0, 1.0)

class MultimodalFeatureExtractor:
    """Extracts features from multimodal sensor data"""
    
    def extract_features(self, eeg_data: Dict, fnirs_data: Dict, 
                        biometric_data: Dict, context: Dict) -> MultimodalFeatures:
        """Extract comprehensive feature set from all data modalities"""
        
        # Extract EEG features
        eeg_features = self._extract_eeg_features(eeg_data)
        
        # Extract fNIRS features
        fnirs_features = self._extract_fnirs_features(fnirs_data)
        
        # Extract biometric features
        biometric_features = self._extract_biometric_features(biometric_data)
        
        # Extract contextual features
        contextual_features = self._extract_contextual_features(context)
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(eeg_data, fnirs_data)
        
        return MultimodalFeatures(
            eeg_features=eeg_features,
            fnirs_features=fnirs_features,
            biometric_features=biometric_features,
            contextual_features=contextual_features,
            temporal_features=temporal_features
        )
    
    def _extract_eeg_features(self, eeg_data: Dict) -> Dict[str, float]:
        """Extract EEG-specific features"""
        features = {}
        
        # Band powers (already calculated in data ingestion)
        features['delta_power'] = eeg_data.get('delta_power', 0.0)
        features['theta_power'] = eeg_data.get('theta_power', 0.0)
        features['alpha_power'] = eeg_data.get('alpha_power', 0.0)
        features['beta_power'] = eeg_data.get('beta_power', 0.0)
        features['gamma_power'] = eeg_data.get('gamma_power', 0.0)
        
        # Derived ratios
        features['alpha_theta_ratio'] = eeg_data.get('alpha_theta_ratio', 1.0)
        features['beta_alpha_ratio'] = eeg_data.get('beta_alpha_ratio', 1.0)
        features['engagement_index'] = eeg_data.get('engagement_index', 0.5)
        features['meditation_index'] = eeg_data.get('meditation_index', 0.5)
        
        # Asymmetry features (if available)
        if 'af7' in eeg_data and 'af8' in eeg_data:
            features['left_alpha'] = eeg_data.get('af7', {}).get('alpha', 0.0)
            features['right_alpha'] = eeg_data.get('af8', {}).get('alpha', 0.0)
        
        return features
    
    def _extract_fnirs_features(self, fnirs_data: Dict) -> Dict[str, float]:
        """Extract fNIRS-specific features"""
        features = {}
        
        # Hemoglobin concentrations
        features['hbo2_left'] = fnirs_data.get('hbo2_left', 0.0)
        features['hbo2_right'] = fnirs_data.get('hbo2_right', 0.0)
        features['hb_left'] = fnirs_data.get('hb_left', 0.0)
        features['hb_right'] = fnirs_data.get('hb_right', 0.0)
        
        # Derived metrics
        features['bilateral_activation'] = (
            (fnirs_data.get('hbo2_left', 0) + fnirs_data.get('hbo2_right', 0)) / 2
        )
        features['oxygenation_index_left'] = fnirs_data.get('oxygenation_index_left', 0.5)
        features['oxygenation_index_right'] = fnirs_data.get('oxygenation_index_right', 0.5)
        features['laterality_index'] = fnirs_data.get('laterality_index', 0.0)
        
        return features
    
    def _extract_biometric_features(self, biometric_data: Dict) -> Dict[str, float]:
        """Extract biometric features"""
        features = {}
        
        # Heart rate variability
        features['heart_rate'] = biometric_data.get('heart_rate', 70.0)
        features['hrv_rmssd'] = biometric_data.get('hrv_rmssd', 50.0)
        features['hrv_pnn50'] = biometric_data.get('hrv_pnn50', 20.0)
        
        # Galvanic skin response
        features['gsr_conductance'] = biometric_data.get('gsr_conductance', 5.0)
        features['gsr_resistance'] = biometric_data.get('gsr_resistance', 200.0)
        
        # Respiratory
        features['breathing_rate'] = biometric_data.get('breathing_rate', 15.0)
        features['breathing_depth'] = biometric_data.get('breathing_depth', 0.5)
        
        return features
    
    def _extract_contextual_features(self, context: Dict) -> Dict[str, float]:
        """Extract contextual features"""
        features = {}
        
        # Environmental context
        features['time_of_day'] = context.get('time_of_day', 12.0) / 24.0  # Normalize to 0-1
        features['ambient_light'] = context.get('ambient_light', 500.0) / 1000.0
        features['noise_level'] = context.get('noise_level', 40.0) / 100.0
        features['temperature'] = context.get('temperature', 22.0) / 40.0
        
        # User context
        features['stress_level'] = context.get('stress_level', 0.5)
        features['fatigue_level'] = context.get('fatigue_level', 0.5)
        features['motivation_level'] = context.get('motivation_level', 0.5)
        
        return features
    
    def _extract_temporal_features(self, eeg_data: Dict, fnirs_data: Dict) -> Dict[str, float]:
        """Extract temporal dynamics features"""
        features = {}
        
        # Placeholder for temporal features
        # In a real implementation, these would be calculated from time series
        features['signal_stability'] = 0.8  # Measure of signal consistency
        features['attention_stability'] = 0.7  # Measure of sustained attention
        features['variability_index'] = 0.3  # Measure of signal variability
        
        return features

class FlowStateDetector:
    """Detects flow state from multimodal features"""
    
    def detect_flow_state(self, features: MultimodalFeatures) -> float:
        """Detect probability of being in flow state"""
        # Flow state characteristics:
        # - High focus, low self-consciousness
        # - Balanced challenge-skill ratio
        # - Time distortion
        # - Intrinsic motivation
        
        focus_score = features.eeg_features.get('engagement_index', 0.5)
        relaxation_score = features.eeg_features.get('meditation_index', 0.5)
        attention_stability = features.temporal_features.get('attention_stability', 0.5)
        
        # Flow requires high focus with relaxed alertness
        flow_neural = focus_score * relaxation_score
        
        # Combine with attention stability
        flow_probability = 0.6 * flow_neural + 0.4 * attention_stability
        
        return np.clip(flow_probability, 0.0, 1.0)

class ResilienceCalculator:
    """Calculates mental resilience metrics"""
    
    def calculate_resilience(self, features: MultimodalFeatures, context: Dict) -> float:
        """Calculate mental resilience score"""
        # Resilience factors:
        # - HRV (autonomic flexibility)
        # - Stress recovery rate
        # - Cognitive flexibility
        # - Emotional regulation
        
        hrv_score = np.clip(features.biometric_features.get('hrv_rmssd', 50) / 100, 0, 1)
        stress_level = features.contextual_features.get('stress_level', 0.5)
        emotional_regulation = 1.0 - abs(features.eeg_features.get('beta_alpha_ratio', 1.5) - 1.5) / 1.5
        
        # Higher HRV, lower stress, better emotional regulation = higher resilience
        resilience = 0.4 * hrv_score + 0.3 * (1 - stress_level) + 0.3 * emotional_regulation
        
        return np.clip(resilience, 0.0, 1.0)

class NeuralStateClassifier:
    """Neural network-based state classification (simplified version)"""
    
    def __init__(self):
        # In a real implementation, this would load pre-trained models
        self.model_weights = self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize simple linear model weights"""
        return {
            'focus': np.random.normal(0, 0.1, 10),
            'relaxation': np.random.normal(0, 0.1, 10),
            'stress': np.random.normal(0, 0.1, 10)
        }
    
    def classify_state(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Classify cognitive state from feature vector"""
        # Simple linear classification for demonstration
        states = {}
        
        for state_name, weights in self.model_weights.items():
            if len(feature_vector) >= len(weights):
                score = np.dot(feature_vector[:len(weights)], weights)
                states[state_name] = 1 / (1 + np.exp(-score))  # Sigmoid activation
            else:
                states[state_name] = 0.5
        
        return states

