"""
NeuroCognify Data Ingestion API
Real-time sensor data ingestion and processing endpoints
"""

from flask import Blueprint, request, jsonify, current_app
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from src.models.sensor_data import (
    db, Session, EEGData, fNIRSData, CognitiveState, 
    BiometricData, Intervention, DeviceCalibration
)
from src.models.user import User

# Create blueprint
data_ingestion_bp = Blueprint('data_ingestion', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

class DataIngestionService:
    """Core data ingestion and preprocessing service"""
    
    def __init__(self):
        self.active_sessions = {}
        self.data_buffers = {}
        self.noise_filters = {}
        self.artifact_detectors = {}
    
    def initialize_session_buffer(self, session_id: int):
        """Initialize data buffers for a new session"""
        self.data_buffers[session_id] = {
            'eeg': [],
            'fnirs': [],
            'biometric': [],
            'timestamps': []
        }
        self.noise_filters[session_id] = AdaptiveNoiseFilter()
        self.artifact_detectors[session_id] = ArtifactDetector()
    
    def process_eeg_data(self, session_id: int, raw_eeg: Dict) -> Dict:
        """Process raw EEG data with filtering and feature extraction"""
        try:
            # Apply noise filtering
            filtered_eeg = self.noise_filters[session_id].filter_eeg(raw_eeg)
            
            # Detect artifacts
            is_artifact = self.artifact_detectors[session_id].detect_eeg_artifacts(filtered_eeg)
            
            # Calculate band powers
            band_powers = self.calculate_band_powers(filtered_eeg)
            
            # Calculate derived metrics
            derived_metrics = self.calculate_eeg_derived_metrics(band_powers)
            
            return {
                'filtered_channels': filtered_eeg,
                'band_powers': band_powers,
                'derived_metrics': derived_metrics,
                'is_artifact': is_artifact,
                'signal_quality': self.calculate_signal_quality(filtered_eeg)
            }
        except Exception as e:
            logger.error(f"Error processing EEG data for session {session_id}: {str(e)}")
            return None
    
    def process_fnirs_data(self, session_id: int, raw_fnirs: Dict) -> Dict:
        """Process raw fNIRS data with hemodynamic response calculation"""
        try:
            # Apply motion artifact correction
            corrected_fnirs = self.artifact_detectors[session_id].correct_fnirs_motion(raw_fnirs)
            
            # Calculate hemodynamic metrics
            hemodynamic_metrics = self.calculate_hemodynamic_metrics(corrected_fnirs)
            
            # Calculate cognitive load indicators
            cognitive_indicators = self.calculate_cognitive_load_indicators(hemodynamic_metrics)
            
            return {
                'hemoglobin_concentrations': corrected_fnirs,
                'hemodynamic_metrics': hemodynamic_metrics,
                'cognitive_indicators': cognitive_indicators,
                'motion_artifact': self.detect_motion_artifacts(raw_fnirs),
                'signal_quality': self.calculate_fnirs_signal_quality(corrected_fnirs)
            }
        except Exception as e:
            logger.error(f"Error processing fNIRS data for session {session_id}: {str(e)}")
            return None
    
    def calculate_band_powers(self, eeg_data: Dict) -> Dict:
        """Calculate EEG band powers using FFT"""
        band_powers = {}
        
        for channel, signal in eeg_data.items():
            if len(signal) < 250:  # Need at least 1 second of data at 250Hz
                continue
                
            # Convert to numpy array
            signal_array = np.array(signal)
            
            # Apply window function
            windowed_signal = signal_array * np.hanning(len(signal_array))
            
            # Calculate FFT
            fft = np.fft.fft(windowed_signal)
            freqs = np.fft.fftfreq(len(windowed_signal), 1/250)  # 250Hz sampling rate
            power_spectrum = np.abs(fft) ** 2
            
            # Calculate band powers
            band_powers[channel] = {
                'delta': self.calculate_band_power(power_spectrum, freqs, 0.5, 4),
                'theta': self.calculate_band_power(power_spectrum, freqs, 4, 8),
                'alpha': self.calculate_band_power(power_spectrum, freqs, 8, 13),
                'beta': self.calculate_band_power(power_spectrum, freqs, 13, 30),
                'gamma': self.calculate_band_power(power_spectrum, freqs, 30, 100)
            }
        
        return band_powers
    
    def calculate_band_power(self, power_spectrum: np.ndarray, freqs: np.ndarray, 
                           low_freq: float, high_freq: float) -> float:
        """Calculate power in a specific frequency band"""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.mean(power_spectrum[band_mask])
    
    def calculate_eeg_derived_metrics(self, band_powers: Dict) -> Dict:
        """Calculate derived EEG metrics"""
        derived = {}
        
        for channel, powers in band_powers.items():
            derived[channel] = {
                'alpha_theta_ratio': powers['alpha'] / (powers['theta'] + 1e-10),
                'beta_alpha_ratio': powers['beta'] / (powers['alpha'] + 1e-10),
                'engagement_index': powers['beta'] / (powers['alpha'] + powers['theta'] + 1e-10),
                'meditation_index': powers['alpha'] / (powers['beta'] + 1e-10)
            }
        
        return derived
    
    def calculate_hemodynamic_metrics(self, fnirs_data: Dict) -> Dict:
        """Calculate hemodynamic response metrics"""
        metrics = {}
        
        for region in ['left', 'right']:
            hbo2_key = f'hbo2_{region}'
            hb_key = f'hb_{region}'
            
            if hbo2_key in fnirs_data and hb_key in fnirs_data:
                hbo2 = fnirs_data[hbo2_key]
                hb = fnirs_data[hb_key]
                
                metrics[region] = {
                    'total_hb': hbo2 + hb,
                    'oxygenation_index': hbo2 / (hbo2 + hb + 1e-10),
                    'hb_diff': hbo2 - hb
                }
        
        # Calculate laterality index
        if 'left' in metrics and 'right' in metrics:
            left_oxy = metrics['left']['oxygenation_index']
            right_oxy = metrics['right']['oxygenation_index']
            metrics['laterality_index'] = (right_oxy - left_oxy) / (right_oxy + left_oxy + 1e-10)
        
        return metrics
    
    def calculate_cognitive_load_indicators(self, hemodynamic_metrics: Dict) -> Dict:
        """Calculate cognitive load from hemodynamic data"""
        indicators = {}
        
        # Average oxygenation across regions
        if 'left' in hemodynamic_metrics and 'right' in hemodynamic_metrics:
            avg_oxygenation = (
                hemodynamic_metrics['left']['oxygenation_index'] + 
                hemodynamic_metrics['right']['oxygenation_index']
            ) / 2
            
            indicators['cognitive_effort'] = min(1.0, max(0.0, avg_oxygenation))
            indicators['mental_fatigue'] = max(0.0, 1.0 - avg_oxygenation)
        
        return indicators
    
    def calculate_signal_quality(self, signal_data: Dict) -> float:
        """Calculate overall signal quality score"""
        quality_scores = []
        
        for channel, signal in signal_data.items():
            if len(signal) > 0:
                signal_array = np.array(signal)
                # Simple quality metric based on signal variance and outliers
                signal_std = np.std(signal_array)
                outlier_ratio = np.sum(np.abs(signal_array) > 3 * signal_std) / len(signal_array)
                quality = max(0.0, 1.0 - outlier_ratio)
                quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def calculate_fnirs_signal_quality(self, fnirs_data: Dict) -> float:
        """Calculate fNIRS signal quality"""
        quality_scores = []
        
        for channel, values in fnirs_data.items():
            if isinstance(values, (int, float)):
                # Simple range check for physiological plausibility
                if -50 <= values <= 50:  # Typical range for hemoglobin changes
                    quality_scores.append(1.0)
                else:
                    quality_scores.append(0.0)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def detect_motion_artifacts(self, raw_fnirs: Dict) -> bool:
        """Detect motion artifacts in fNIRS data"""
        # Simple threshold-based detection
        for channel, value in raw_fnirs.items():
            if isinstance(value, (int, float)) and abs(value) > 100:
                return True
        return False

class AdaptiveNoiseFilter:
    """Adaptive noise filtering for EEG signals"""
    
    def __init__(self):
        self.filter_coefficients = {}
        self.signal_history = {}
    
    def filter_eeg(self, eeg_data: Dict) -> Dict:
        """Apply adaptive filtering to EEG channels"""
        filtered_data = {}
        
        for channel, signal in eeg_data.items():
            if isinstance(signal, list) and len(signal) > 0:
                # Simple moving average filter for now
                filtered_signal = self.moving_average_filter(signal, window_size=5)
                filtered_data[channel] = filtered_signal
            else:
                filtered_data[channel] = signal
        
        return filtered_data
    
    def moving_average_filter(self, signal: List[float], window_size: int) -> List[float]:
        """Apply moving average filter"""
        if len(signal) < window_size:
            return signal
        
        filtered = []
        for i in range(len(signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2 + 1)
            filtered.append(np.mean(signal[start_idx:end_idx]))
        
        return filtered

class ArtifactDetector:
    """Artifact detection for EEG and fNIRS signals"""
    
    def __init__(self):
        self.eeg_thresholds = {
            'amplitude': 100,  # μV
            'gradient': 50     # μV/sample
        }
        self.fnirs_thresholds = {
            'amplitude': 100,  # μM
            'gradient': 20     # μM/sample
        }
    
    def detect_eeg_artifacts(self, eeg_data: Dict) -> bool:
        """Detect artifacts in EEG data"""
        for channel, signal in eeg_data.items():
            if isinstance(signal, list) and len(signal) > 1:
                signal_array = np.array(signal)
                
                # Check amplitude threshold
                if np.any(np.abs(signal_array) > self.eeg_thresholds['amplitude']):
                    return True
                
                # Check gradient threshold
                gradient = np.diff(signal_array)
                if np.any(np.abs(gradient) > self.eeg_thresholds['gradient']):
                    return True
        
        return False
    
    def correct_fnirs_motion(self, fnirs_data: Dict) -> Dict:
        """Apply motion artifact correction to fNIRS data"""
        corrected_data = {}
        
        for channel, value in fnirs_data.items():
            if isinstance(value, (int, float)):
                # Simple clipping for extreme values
                corrected_value = np.clip(value, -50, 50)
                corrected_data[channel] = corrected_value
            else:
                corrected_data[channel] = value
        
        return corrected_data

# Initialize service
data_service = DataIngestionService()

@data_ingestion_bp.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new neurofeedback session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        session_type = data.get('session_type', 'general')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Create new session
        session = Session(
            user_id=user_id,
            session_type=session_type,
            start_time=datetime.utcnow()
        )
        
        db.session.add(session)
        db.session.commit()
        
        # Initialize data buffers
        data_service.initialize_session_buffer(session.id)
        data_service.active_sessions[session.id] = {
            'start_time': datetime.utcnow(),
            'user_id': user_id,
            'session_type': session_type
        }
        
        logger.info(f"Started session {session.id} for user {user_id}")
        
        return jsonify({
            'session_id': session.id,
            'status': 'started',
            'start_time': session.start_time.isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        return jsonify({'error': 'Failed to start session'}), 500

@data_ingestion_bp.route('/api/session/<int:session_id>/stop', methods=['POST'])
def stop_session(session_id):
    """Stop an active neurofeedback session"""
    try:
        session = Session.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Update session end time
        session.end_time = datetime.utcnow()
        session.duration_seconds = int((session.end_time - session.start_time).total_seconds())
        
        # Calculate session metrics
        session_metrics = calculate_session_metrics(session_id)
        session.presence_score = session_metrics.get('avg_presence', 0.0)
        session.agency_sensitivity_avg = session_metrics.get('avg_agency_sensitivity', 0.0)
        session.cognitive_load_avg = session_metrics.get('avg_cognitive_load', 0.0)
        session.data_quality_score = session_metrics.get('data_quality', 0.0)
        
        db.session.commit()
        
        # Clean up buffers
        if session_id in data_service.active_sessions:
            del data_service.active_sessions[session_id]
        if session_id in data_service.data_buffers:
            del data_service.data_buffers[session_id]
        
        logger.info(f"Stopped session {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'status': 'stopped',
            'duration_seconds': session.duration_seconds,
            'metrics': session_metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error stopping session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to stop session'}), 500

@data_ingestion_bp.route('/api/data/eeg', methods=['POST'])
def ingest_eeg_data():
    """Ingest real-time EEG data"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        eeg_channels = data.get('eeg_channels', {})
        
        if not session_id or session_id not in data_service.active_sessions:
            return jsonify({'error': 'Invalid or inactive session'}), 400
        
        # Process EEG data
        processed_data = data_service.process_eeg_data(session_id, eeg_channels)
        
        if processed_data is None:
            return jsonify({'error': 'Failed to process EEG data'}), 500
        
        # Store in database
        eeg_record = EEGData(
            session_id=session_id,
            timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
            tp9=eeg_channels.get('tp9'),
            af7=eeg_channels.get('af7'),
            af8=eeg_channels.get('af8'),
            tp10=eeg_channels.get('tp10'),
            delta_power=processed_data['band_powers'].get('af7', {}).get('delta'),
            theta_power=processed_data['band_powers'].get('af7', {}).get('theta'),
            alpha_power=processed_data['band_powers'].get('af7', {}).get('alpha'),
            beta_power=processed_data['band_powers'].get('af7', {}).get('beta'),
            gamma_power=processed_data['band_powers'].get('af7', {}).get('gamma'),
            alpha_theta_ratio=processed_data['derived_metrics'].get('af7', {}).get('alpha_theta_ratio'),
            beta_alpha_ratio=processed_data['derived_metrics'].get('af7', {}).get('beta_alpha_ratio'),
            engagement_index=processed_data['derived_metrics'].get('af7', {}).get('engagement_index'),
            meditation_index=processed_data['derived_metrics'].get('af7', {}).get('meditation_index'),
            is_artifact=processed_data['is_artifact'],
            signal_quality=processed_data['signal_quality']
        )
        
        db.session.add(eeg_record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'processed_data': {
                'band_powers': processed_data['band_powers'],
                'derived_metrics': processed_data['derived_metrics'],
                'signal_quality': processed_data['signal_quality'],
                'is_artifact': processed_data['is_artifact']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error ingesting EEG data: {str(e)}")
        return jsonify({'error': 'Failed to ingest EEG data'}), 500

@data_ingestion_bp.route('/api/data/fnirs', methods=['POST'])
def ingest_fnirs_data():
    """Ingest real-time fNIRS data"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        fnirs_data = data.get('fnirs_data', {})
        
        if not session_id or session_id not in data_service.active_sessions:
            return jsonify({'error': 'Invalid or inactive session'}), 400
        
        # Process fNIRS data
        processed_data = data_service.process_fnirs_data(session_id, fnirs_data)
        
        if processed_data is None:
            return jsonify({'error': 'Failed to process fNIRS data'}), 500
        
        # Store in database
        fnirs_record = fNIRSData(
            session_id=session_id,
            timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
            hbo2_left=fnirs_data.get('hbo2_left'),
            hbo2_right=fnirs_data.get('hbo2_right'),
            hb_left=fnirs_data.get('hb_left'),
            hb_right=fnirs_data.get('hb_right'),
            hbt_left=processed_data['hemodynamic_metrics'].get('left', {}).get('total_hb'),
            hbt_right=processed_data['hemodynamic_metrics'].get('right', {}).get('total_hb'),
            oxygenation_index_left=processed_data['hemodynamic_metrics'].get('left', {}).get('oxygenation_index'),
            oxygenation_index_right=processed_data['hemodynamic_metrics'].get('right', {}).get('oxygenation_index'),
            laterality_index=processed_data['hemodynamic_metrics'].get('laterality_index'),
            cognitive_effort=processed_data['cognitive_indicators'].get('cognitive_effort'),
            mental_fatigue=processed_data['cognitive_indicators'].get('mental_fatigue'),
            signal_quality=processed_data['signal_quality'],
            motion_artifact=processed_data['motion_artifact']
        )
        
        db.session.add(fnirs_record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'processed_data': {
                'hemodynamic_metrics': processed_data['hemodynamic_metrics'],
                'cognitive_indicators': processed_data['cognitive_indicators'],
                'signal_quality': processed_data['signal_quality'],
                'motion_artifact': processed_data['motion_artifact']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error ingesting fNIRS data: {str(e)}")
        return jsonify({'error': 'Failed to ingest fNIRS data'}), 500

@data_ingestion_bp.route('/api/data/biometric', methods=['POST'])
def ingest_biometric_data():
    """Ingest additional biometric sensor data"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())
        biometric_data = data.get('biometric_data', {})
        
        if not session_id or session_id not in data_service.active_sessions:
            return jsonify({'error': 'Invalid or inactive session'}), 400
        
        # Store biometric data
        biometric_record = BiometricData(
            session_id=session_id,
            timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
            heart_rate=biometric_data.get('heart_rate'),
            rr_interval=biometric_data.get('rr_interval'),
            hrv_rmssd=biometric_data.get('hrv_rmssd'),
            hrv_pnn50=biometric_data.get('hrv_pnn50'),
            gsr_conductance=biometric_data.get('gsr_conductance'),
            gsr_resistance=biometric_data.get('gsr_resistance'),
            breathing_rate=biometric_data.get('breathing_rate'),
            breathing_depth=biometric_data.get('breathing_depth'),
            breathing_variability=biometric_data.get('breathing_variability'),
            ambient_temperature=biometric_data.get('ambient_temperature'),
            ambient_light=biometric_data.get('ambient_light'),
            noise_level=biometric_data.get('noise_level')
        )
        
        db.session.add(biometric_record)
        db.session.commit()
        
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        logger.error(f"Error ingesting biometric data: {str(e)}")
        return jsonify({'error': 'Failed to ingest biometric data'}), 500

@data_ingestion_bp.route('/api/session/<int:session_id>/status', methods=['GET'])
def get_session_status(session_id):
    """Get current session status and real-time metrics"""
    try:
        if session_id not in data_service.active_sessions:
            return jsonify({'error': 'Session not active'}), 404
        
        session_info = data_service.active_sessions[session_id]
        
        # Get latest data points
        latest_eeg = EEGData.query.filter_by(session_id=session_id).order_by(EEGData.timestamp.desc()).first()
        latest_fnirs = fNIRSData.query.filter_by(session_id=session_id).order_by(fNIRSData.timestamp.desc()).first()
        latest_cognitive = CognitiveState.query.filter_by(session_id=session_id).order_by(CognitiveState.timestamp.desc()).first()
        
        # Calculate session duration
        duration = (datetime.utcnow() - session_info['start_time']).total_seconds()
        
        return jsonify({
            'session_id': session_id,
            'status': 'active',
            'duration_seconds': int(duration),
            'latest_data': {
                'eeg': latest_eeg.to_dict() if latest_eeg else None,
                'fnirs': latest_fnirs.to_dict() if latest_fnirs else None,
                'cognitive_state': latest_cognitive.to_dict() if latest_cognitive else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        return jsonify({'error': 'Failed to get session status'}), 500

def calculate_session_metrics(session_id: int) -> Dict:
    """Calculate aggregate metrics for a completed session"""
    try:
        # Get all cognitive states for the session
        cognitive_states = CognitiveState.query.filter_by(session_id=session_id).all()
        
        if not cognitive_states:
            return {
                'avg_presence': 0.0,
                'avg_agency_sensitivity': 0.5,
                'avg_cognitive_load': 0.0,
                'data_quality': 0.0
            }
        
        # Calculate averages
        presence_scores = [cs.presence_score for cs in cognitive_states if cs.presence_score is not None]
        agency_scores = [cs.agency_sensitivity for cs in cognitive_states if cs.agency_sensitivity is not None]
        cognitive_loads = [cs.cognitive_load for cs in cognitive_states if cs.cognitive_load is not None]
        
        # Get data quality from EEG records
        eeg_records = EEGData.query.filter_by(session_id=session_id).all()
        quality_scores = [eeg.signal_quality for eeg in eeg_records if eeg.signal_quality is not None]
        
        return {
            'avg_presence': np.mean(presence_scores) if presence_scores else 0.0,
            'avg_agency_sensitivity': np.mean(agency_scores) if agency_scores else 0.5,
            'avg_cognitive_load': np.mean(cognitive_loads) if cognitive_loads else 0.0,
            'data_quality': np.mean(quality_scores) if quality_scores else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error calculating session metrics: {str(e)}")
        return {
            'avg_presence': 0.0,
            'avg_agency_sensitivity': 0.5,
            'avg_cognitive_load': 0.0,
            'data_quality': 0.0
        }

