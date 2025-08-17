"""
NeuroCognify Knowledge Integration System
Cross-disciplinary knowledge graph for contextual interventions
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content in the knowledge graph"""
    NEUROSCIENCE_RESEARCH = "neuroscience_research"
    PSYCHOLOGICAL_THEORY = "psychological_theory"
    PHILOSOPHICAL_CONCEPT = "philosophical_concept"
    MYTHOLOGICAL_THEME = "mythological_theme"
    LITERARY_WORK = "literary_work"
    ARTISTIC_PIECE = "artistic_piece"
    HISTORICAL_PATTERN = "historical_pattern"
    BREATHING_EXERCISE = "breathing_exercise"
    MEDITATION_PRACTICE = "meditation_practice"
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    STORY_MODULE = "story_module"
    MUSIC_THERAPY = "music_therapy"

class InterventionType(Enum):
    """Types of interventions"""
    AGENCY_CALIBRATION = "agency_calibration"
    STRESS_REDUCTION = "stress_reduction"
    FOCUS_ENHANCEMENT = "focus_enhancement"
    EMOTIONAL_REGULATION = "emotional_regulation"
    RESILIENCE_BUILDING = "resilience_building"
    FLOW_STATE_INDUCTION = "flow_state_induction"
    MINDFULNESS_TRAINING = "mindfulness_training"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"

@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    id: str
    title: str
    content_type: ContentType
    description: str
    content: str
    metadata: Dict
    tags: List[str]
    effectiveness_score: float
    usage_count: int
    created_at: datetime
    updated_at: datetime

@dataclass
class InterventionRecommendation:
    """Represents an intervention recommendation"""
    intervention_id: str
    intervention_type: InterventionType
    content_nodes: List[KnowledgeNode]
    personalization_score: float
    trigger_reason: str
    expected_effectiveness: float
    duration_minutes: int
    intensity_level: float

class KnowledgeGraph:
    """Cross-disciplinary knowledge graph for NeuroCognify"""
    
    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        self.content_database = self._initialize_content_database()
        self.intervention_rules = self._initialize_intervention_rules()
        self._populate_knowledge_base()
    
    def _initialize_content_database(self) -> Dict:
        """Initialize the content database with curated materials"""
        return {
            # Neuroscience Research
            "mmn_agency_detection": {
                "title": "Mismatch Negativity and Agency Detection",
                "content_type": ContentType.NEUROSCIENCE_RESEARCH,
                "description": "Research on how mismatch negativity relates to agency detection mechanisms",
                "content": """
                Recent neuroscience research has shown that mismatch negativity (MMN) - an event-related potential 
                occurring 100-250ms after an unexpected stimulus - plays a crucial role in agency detection. 
                When our brain's predictions about the world are violated, the MMN response helps us determine 
                whether the violation was caused by our own actions or external agents.
                
                Studies using EEG have demonstrated that individuals with higher MMN amplitudes tend to have 
                more sensitive agency detection systems, sometimes leading to over-attribution of intentionality 
                to random events. This neurophysiological marker can be used to calibrate agency sensitivity 
                in real-time neurofeedback systems.
                """,
                "tags": ["neuroscience", "agency", "eeg", "prediction", "mmn"],
                "effectiveness_score": 0.85
            },
            
            # Philosophical Concepts
            "stoic_agency_control": {
                "title": "Stoic Philosophy: Sphere of Control",
                "content_type": ContentType.PHILOSOPHICAL_CONCEPT,
                "description": "Stoic teachings on distinguishing between what we can and cannot control",
                "content": """
                The Stoic philosophers, particularly Epictetus, taught a fundamental distinction between what is 
                'up to us' (eph' hēmin) and what is 'not up to us' (ouk eph' hēmin). This ancient wisdom provides 
                a framework for calibrating our agency detection system.
                
                "Some things are within our power, while others are not." - Epictetus
                
                When we over-attribute agency to external events, we suffer. When we under-attribute our own 
                agency, we become passive. The Stoic practice of morning reflection asks: "What today is truly 
                within my sphere of influence?" This question helps calibrate our sense of control and reduces 
                both anxiety and learned helplessness.
                
                Modern neuroscience validates this ancient insight: our prefrontal cortex, the brain region 
                associated with executive control, shows different activation patterns when we focus on 
                controllable versus uncontrollable aspects of our experience.
                """,
                "tags": ["philosophy", "stoicism", "control", "agency", "wisdom"],
                "effectiveness_score": 0.78
            },
            
            # Mythological Themes
            "athena_wisdom_owl": {
                "title": "Athena's Owl: Symbol of Wisdom and Perspective",
                "content_type": ContentType.MYTHOLOGICAL_THEME,
                "description": "The mythological significance of Athena's owl as a symbol of wisdom and clear sight",
                "content": """
                In Greek mythology, Athena's owl represents the ability to see clearly in darkness - both literal 
                and metaphorical. The owl's exceptional night vision symbolizes the capacity to perceive truth 
                when others are blinded by confusion, fear, or false patterns.
                
                The owl's silent flight represents the quiet wisdom that comes from careful observation rather 
                than hasty judgment. When our agency detection system is hyperactive, we see threats and 
                intentions everywhere. The owl teaches us to pause, observe, and distinguish between real 
                patterns and imagined ones.
                
                "The owl of Minerva spreads its wings only with the falling of the dusk." - Hegel
                
                This suggests that true understanding often comes after events have unfolded, reminding us 
                to be humble about our ability to detect agency and intention in real-time. The owl's wisdom 
                is patient, observant, and discerning.
                """,
                "tags": ["mythology", "wisdom", "perception", "patience", "discernment"],
                "effectiveness_score": 0.72
            },
            
            # Breathing Exercises
            "box_breathing_agency": {
                "title": "Box Breathing for Agency Calibration",
                "content_type": ContentType.BREATHING_EXERCISE,
                "description": "Structured breathing technique to regulate agency sensitivity",
                "content": """
                Box breathing (4-4-4-4 pattern) helps regulate the autonomic nervous system and can recalibrate 
                an overactive agency detection system. When we're stressed or anxious, our brain becomes 
                hypervigilant, seeing patterns and threats that may not exist.
                
                Instructions:
                1. Inhale for 4 counts
                2. Hold for 4 counts  
                3. Exhale for 4 counts
                4. Hold empty for 4 counts
                5. Repeat for 5-10 cycles
                
                The structured nature of this practice gives your mind something concrete to focus on, 
                reducing the tendency to over-interpret ambiguous stimuli. The breath becomes an anchor 
                of genuine agency - something you truly can control - helping recalibrate your sense 
                of what is and isn't within your influence.
                
                Research shows that controlled breathing activates the parasympathetic nervous system, 
                reducing cortisol levels and the hypervigilance associated with over-sensitive agency detection.
                """,
                "tags": ["breathing", "regulation", "autonomic", "control", "practice"],
                "effectiveness_score": 0.82
            },
            
            # Story Modules
            "pattern_recognition_story": {
                "title": "The Constellation Maker",
                "content_type": ContentType.STORY_MODULE,
                "description": "Interactive story about pattern recognition and meaning-making",
                "content": """
                You are an ancient astronomer, tasked with mapping the night sky. Each night, you see countless 
                stars scattered across the darkness. Your mind naturally begins to connect them into patterns - 
                a hunter, a bear, a swan.
                
                But tonight is different. The stars seem to be moving, forming new patterns. Is this real, 
                or is your pattern-seeking mind creating connections that don't exist?
                
                As you watch, you realize that some patterns are stable and meaningful - they help you navigate 
                and tell time. Others are fleeting projections of your hopes and fears onto random arrangements 
                of light.
                
                The wisdom lies not in stopping your pattern-making mind, but in learning to distinguish 
                between patterns that serve you and those that mislead you. Like the ancient navigators, 
                you must calibrate your instruments - in this case, your agency detection system.
                
                Reflection: What patterns in your life are like reliable constellations? Which are like 
                clouds that seem to form faces - meaningful to you in the moment, but ultimately temporary 
                projections?
                """,
                "tags": ["story", "patterns", "navigation", "wisdom", "reflection"],
                "effectiveness_score": 0.75
            },
            
            # Cognitive Reappraisal
            "coincidence_reframe": {
                "title": "Reframing Coincidences",
                "content_type": ContentType.COGNITIVE_REAPPRAISAL,
                "description": "Cognitive technique for reinterpreting seemingly meaningful coincidences",
                "content": """
                When we experience a striking coincidence, our agency detection system often activates, 
                suggesting hidden meaning or intention. This cognitive reappraisal technique helps you 
                evaluate these experiences more objectively.
                
                The RAIN Technique for Coincidences:
                
                RECOGNIZE: Notice when you're attributing special meaning to a coincidence
                "I just thought of my friend, and then they called. This must mean something!"
                
                ALLOW: Don't judge yourself for having this thought
                "It's natural for my brain to look for patterns and meaning."
                
                INVESTIGATE: Examine the experience with curiosity
                "How often do I think of this friend? How often do they call? What's the actual probability?"
                
                NATURAL AWARENESS: Rest in the understanding that coincidences can be meaningful to you 
                personally without requiring external agency or cosmic significance.
                
                This practice helps calibrate your agency detection system, allowing you to appreciate 
                the subjective meaning of coincidences without over-interpreting their objective significance.
                """,
                "tags": ["cognitive", "reappraisal", "coincidence", "probability", "meaning"],
                "effectiveness_score": 0.79
            },
            
            # Historical Patterns
            "plague_agency_attribution": {
                "title": "Historical Patterns: Plague and Agency Attribution",
                "content_type": ContentType.HISTORICAL_PATTERN,
                "description": "How pandemics throughout history have affected collective agency detection",
                "content": """
                Throughout history, pandemics have consistently triggered heightened agency detection in 
                human populations. During the Black Death (1347-1351), people attributed the plague to 
                everything from divine punishment to poisoned wells to astrological alignments.
                
                The COVID-19 pandemic showed similar patterns: increased conspiracy thinking, 
                scapegoating, and attribution of intentional agency to what were largely natural processes. 
                This isn't a failure of human reasoning - it's an adaptive response to uncertainty and threat.
                
                When faced with invisible, unpredictable dangers, our ancestors who assumed agency 
                (even incorrectly) were more likely to take protective action than those who assumed 
                randomness. The cost of false positives (seeing agency where none exists) was often 
                lower than false negatives (missing real threats).
                
                Understanding this historical pattern helps us recognize when our own agency detection 
                system might be heightened by collective stress and uncertainty. The goal isn't to 
                eliminate this response, but to calibrate it appropriately for modern contexts.
                """,
                "tags": ["history", "pandemic", "collective", "adaptation", "uncertainty"],
                "effectiveness_score": 0.73
            },
            
            # Meditation Practices
            "open_monitoring_meditation": {
                "title": "Open Monitoring Meditation for Agency Awareness",
                "content_type": ContentType.MEDITATION_PRACTICE,
                "description": "Meditation practice to observe agency attribution without judgment",
                "content": """
                This meditation practice helps you observe your agency detection system in action, 
                developing awareness of when and how you attribute intention and control to experiences.
                
                Instructions:
                1. Sit comfortably and close your eyes
                2. Begin with a few minutes of breath awareness
                3. Expand your attention to include all arising experiences
                4. When thoughts arise, notice if they involve attribution of agency:
                   - "Someone is making that noise on purpose"
                   - "This meditation isn't working because I'm not trying hard enough"
                   - "The universe is sending me a message"
                
                5. For each agency attribution, simply note:
                   - "Agency detection active"
                   - Neither suppress nor indulge the thought
                   - Return to open awareness
                
                6. Notice patterns: When does your agency detection system activate most?
                   - During uncertainty?
                   - When you're stressed?
                   - Around specific types of experiences?
                
                This practice builds meta-cognitive awareness of your agency detection patterns, 
                the first step in calibrating them more skillfully.
                """,
                "tags": ["meditation", "awareness", "metacognition", "observation", "mindfulness"],
                "effectiveness_score": 0.81
            }
        }
    
    def _initialize_intervention_rules(self) -> Dict:
        """Initialize rules for intervention recommendations"""
        return {
            InterventionType.AGENCY_CALIBRATION: {
                "triggers": {
                    "agency_sensitivity": {"min": 0.7, "max": 1.0},  # Over-sensitive
                    "stress_level": {"min": 0.6, "max": 1.0},
                    "pattern_detection_bias": {"min": 0.5, "max": 1.0}
                },
                "content_types": [
                    ContentType.PHILOSOPHICAL_CONCEPT,
                    ContentType.COGNITIVE_REAPPRAISAL,
                    ContentType.BREATHING_EXERCISE,
                    ContentType.STORY_MODULE
                ],
                "duration_range": (5, 15),
                "intensity_range": (0.3, 0.7)
            },
            
            InterventionType.STRESS_REDUCTION: {
                "triggers": {
                    "stress_level": {"min": 0.7, "max": 1.0},
                    "emotional_arousal": {"min": 0.6, "max": 1.0},
                    "heart_rate": {"min": 80, "max": 120}
                },
                "content_types": [
                    ContentType.BREATHING_EXERCISE,
                    ContentType.MEDITATION_PRACTICE,
                    ContentType.MUSIC_THERAPY
                ],
                "duration_range": (3, 10),
                "intensity_range": (0.2, 0.5)
            },
            
            InterventionType.FOCUS_ENHANCEMENT: {
                "triggers": {
                    "focus_level": {"min": 0.0, "max": 0.4},  # Low focus
                    "cognitive_load": {"min": 0.0, "max": 0.3},
                    "attention_stability": {"min": 0.0, "max": 0.4}
                },
                "content_types": [
                    ContentType.BREATHING_EXERCISE,
                    ContentType.MEDITATION_PRACTICE,
                    ContentType.COGNITIVE_REAPPRAISAL
                ],
                "duration_range": (5, 20),
                "intensity_range": (0.4, 0.8)
            },
            
            InterventionType.EMOTIONAL_REGULATION: {
                "triggers": {
                    "emotional_valence": {"min": -1.0, "max": -0.5},  # Negative emotions
                    "emotional_arousal": {"min": 0.7, "max": 1.0},
                    "stress_level": {"min": 0.6, "max": 1.0}
                },
                "content_types": [
                    ContentType.PHILOSOPHICAL_CONCEPT,
                    ContentType.STORY_MODULE,
                    ContentType.COGNITIVE_REAPPRAISAL,
                    ContentType.ARTISTIC_PIECE
                ],
                "duration_range": (10, 25),
                "intensity_range": (0.3, 0.6)
            },
            
            InterventionType.RESILIENCE_BUILDING: {
                "triggers": {
                    "mental_resilience": {"min": 0.0, "max": 0.4},  # Low resilience
                    "recovery_rate": {"min": 0.0, "max": 0.3},
                    "adaptability_score": {"min": 0.0, "max": 0.4}
                },
                "content_types": [
                    ContentType.HISTORICAL_PATTERN,
                    ContentType.PHILOSOPHICAL_CONCEPT,
                    ContentType.MYTHOLOGICAL_THEME,
                    ContentType.STORY_MODULE
                ],
                "duration_range": (15, 30),
                "intensity_range": (0.4, 0.7)
            },
            
            InterventionType.FLOW_STATE_INDUCTION: {
                "triggers": {
                    "flow_state_probability": {"min": 0.3, "max": 0.7},  # Moderate flow potential
                    "focus_level": {"min": 0.5, "max": 0.8},
                    "challenge_skill_balance": {"min": 0.4, "max": 0.8}
                },
                "content_types": [
                    ContentType.MEDITATION_PRACTICE,
                    ContentType.BREATHING_EXERCISE,
                    ContentType.MUSIC_THERAPY
                ],
                "duration_range": (10, 45),
                "intensity_range": (0.5, 0.9)
            }
        }
    
    def _populate_knowledge_base(self):
        """Populate the knowledge graph with content nodes"""
        for content_id, content_data in self.content_database.items():
            node = KnowledgeNode(
                id=content_id,
                title=content_data["title"],
                content_type=content_data["content_type"],
                description=content_data["description"],
                content=content_data["content"],
                metadata={},
                tags=content_data["tags"],
                effectiveness_score=content_data["effectiveness_score"],
                usage_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.nodes[content_id] = node
    
    def recommend_intervention(self, 
                             cognitive_state: Dict, 
                             user_context: Dict,
                             user_preferences: Optional[Dict] = None) -> Optional[InterventionRecommendation]:
        """Recommend an intervention based on current cognitive state"""
        try:
            # Find matching intervention types
            matching_interventions = self._find_matching_interventions(cognitive_state)
            
            if not matching_interventions:
                return None
            
            # Select the most appropriate intervention type
            intervention_type = self._select_intervention_type(matching_interventions, cognitive_state)
            
            # Get content nodes for this intervention
            content_nodes = self._select_content_nodes(intervention_type, user_context, user_preferences)
            
            # Calculate personalization score
            personalization_score = self._calculate_personalization_score(
                content_nodes, user_context, user_preferences
            )
            
            # Generate trigger reason
            trigger_reason = self._generate_trigger_reason(intervention_type, cognitive_state)
            
            # Estimate effectiveness
            expected_effectiveness = self._estimate_effectiveness(
                intervention_type, content_nodes, cognitive_state
            )
            
            # Determine duration and intensity
            duration, intensity = self._determine_intervention_parameters(
                intervention_type, cognitive_state
            )
            
            recommendation = InterventionRecommendation(
                intervention_id=f"{intervention_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                intervention_type=intervention_type,
                content_nodes=content_nodes,
                personalization_score=personalization_score,
                trigger_reason=trigger_reason,
                expected_effectiveness=expected_effectiveness,
                duration_minutes=duration,
                intensity_level=intensity
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error recommending intervention: {str(e)}")
            return None
    
    def _find_matching_interventions(self, cognitive_state: Dict) -> List[InterventionType]:
        """Find intervention types that match the current cognitive state"""
        matching = []
        
        for intervention_type, rules in self.intervention_rules.items():
            triggers = rules["triggers"]
            matches = 0
            total_triggers = len(triggers)
            
            for metric, threshold in triggers.items():
                if metric in cognitive_state:
                    value = cognitive_state[metric]
                    if threshold["min"] <= value <= threshold["max"]:
                        matches += 1
            
            # Require at least 50% of triggers to match
            if matches / total_triggers >= 0.5:
                matching.append(intervention_type)
        
        return matching
    
    def _select_intervention_type(self, 
                                matching_interventions: List[InterventionType], 
                                cognitive_state: Dict) -> InterventionType:
        """Select the most appropriate intervention type"""
        if len(matching_interventions) == 1:
            return matching_interventions[0]
        
        # Priority order based on urgency
        priority_order = [
            InterventionType.STRESS_REDUCTION,
            InterventionType.AGENCY_CALIBRATION,
            InterventionType.EMOTIONAL_REGULATION,
            InterventionType.FOCUS_ENHANCEMENT,
            InterventionType.RESILIENCE_BUILDING,
            InterventionType.FLOW_STATE_INDUCTION,
            InterventionType.MINDFULNESS_TRAINING
        ]
        
        for intervention_type in priority_order:
            if intervention_type in matching_interventions:
                return intervention_type
        
        return matching_interventions[0]  # Fallback
    
    def _select_content_nodes(self, 
                            intervention_type: InterventionType,
                            user_context: Dict,
                            user_preferences: Optional[Dict] = None) -> List[KnowledgeNode]:
        """Select appropriate content nodes for the intervention"""
        rules = self.intervention_rules[intervention_type]
        allowed_content_types = rules["content_types"]
        
        # Filter nodes by content type
        candidate_nodes = [
            node for node in self.nodes.values()
            if node.content_type in allowed_content_types
        ]
        
        # Apply user preferences if available
        if user_preferences:
            preferred_tags = user_preferences.get("preferred_tags", [])
            avoided_tags = user_preferences.get("avoided_tags", [])
            
            if preferred_tags:
                candidate_nodes = [
                    node for node in candidate_nodes
                    if any(tag in node.tags for tag in preferred_tags)
                ]
            
            if avoided_tags:
                candidate_nodes = [
                    node for node in candidate_nodes
                    if not any(tag in node.tags for tag in avoided_tags)
                ]
        
        # Sort by effectiveness score and usage balance
        candidate_nodes.sort(
            key=lambda node: node.effectiveness_score - (node.usage_count * 0.01),
            reverse=True
        )
        
        # Select 1-3 nodes based on intervention type
        if intervention_type in [InterventionType.STRESS_REDUCTION, InterventionType.FOCUS_ENHANCEMENT]:
            return candidate_nodes[:1]  # Single focused intervention
        else:
            return candidate_nodes[:2]  # Multi-modal intervention
    
    def _calculate_personalization_score(self, 
                                       content_nodes: List[KnowledgeNode],
                                       user_context: Dict,
                                       user_preferences: Optional[Dict] = None) -> float:
        """Calculate how well the intervention is personalized"""
        base_score = 0.5
        
        # Adjust based on user preferences
        if user_preferences:
            preferred_tags = user_preferences.get("preferred_tags", [])
            if preferred_tags:
                tag_matches = sum(
                    1 for node in content_nodes
                    for tag in node.tags
                    if tag in preferred_tags
                )
                total_tags = sum(len(node.tags) for node in content_nodes)
                if total_tags > 0:
                    base_score += (tag_matches / total_tags) * 0.3
        
        # Adjust based on context relevance
        time_of_day = user_context.get("time_of_day", 12)
        if 6 <= time_of_day <= 10:  # Morning
            base_score += 0.1 if any("morning" in node.tags for node in content_nodes) else 0
        elif 18 <= time_of_day <= 22:  # Evening
            base_score += 0.1 if any("evening" in node.tags for node in content_nodes) else 0
        
        return min(1.0, base_score)
    
    def _generate_trigger_reason(self, 
                               intervention_type: InterventionType, 
                               cognitive_state: Dict) -> str:
        """Generate a human-readable explanation for why this intervention was recommended"""
        reasons = {
            InterventionType.AGENCY_CALIBRATION: [
                "Your agency detection system appears heightened, which may lead to over-interpreting random events as intentional.",
                "Current stress levels may be increasing your tendency to see patterns where none exist.",
                "Your brain is working hard to find meaning in ambiguous situations - let's help it calibrate."
            ],
            InterventionType.STRESS_REDUCTION: [
                "Your stress indicators suggest you could benefit from some calming techniques.",
                "Your nervous system appears activated - let's help it return to balance.",
                "High arousal levels detected - time for some restorative practices."
            ],
            InterventionType.FOCUS_ENHANCEMENT: [
                "Your attention seems scattered - let's help you gather your focus.",
                "Low engagement levels detected - time to sharpen your mental clarity.",
                "Your cognitive resources appear underutilized - let's optimize them."
            ],
            InterventionType.EMOTIONAL_REGULATION: [
                "Your emotional state could benefit from some rebalancing techniques.",
                "Strong emotional activation detected - let's help you process this skillfully.",
                "Your emotional system needs some support right now."
            ],
            InterventionType.RESILIENCE_BUILDING: [
                "Your resilience indicators suggest you could benefit from strengthening practices.",
                "Building your mental resilience will help you navigate future challenges.",
                "Your adaptive capacity could use some reinforcement."
            ],
            InterventionType.FLOW_STATE_INDUCTION: [
                "Conditions are favorable for entering a flow state - let's optimize this opportunity.",
                "Your focus and relaxation levels suggest readiness for deep engagement.",
                "Perfect conditions detected for peak performance state."
            ]
        }
        
        return random.choice(reasons.get(intervention_type, ["Intervention recommended based on current state."]))
    
    def _estimate_effectiveness(self, 
                              intervention_type: InterventionType,
                              content_nodes: List[KnowledgeNode],
                              cognitive_state: Dict) -> float:
        """Estimate the expected effectiveness of the intervention"""
        # Base effectiveness from content nodes
        if content_nodes:
            base_effectiveness = sum(node.effectiveness_score for node in content_nodes) / len(content_nodes)
        else:
            base_effectiveness = 0.5
        
        # Adjust based on cognitive state alignment
        state_alignment = self._calculate_state_alignment(intervention_type, cognitive_state)
        
        # Combine factors
        estimated_effectiveness = (base_effectiveness * 0.7) + (state_alignment * 0.3)
        
        return min(1.0, estimated_effectiveness)
    
    def _calculate_state_alignment(self, 
                                 intervention_type: InterventionType, 
                                 cognitive_state: Dict) -> float:
        """Calculate how well the intervention aligns with the current state"""
        # This would be more sophisticated in a real implementation
        # For now, return a reasonable estimate based on intervention type
        alignment_scores = {
            InterventionType.AGENCY_CALIBRATION: cognitive_state.get("agency_sensitivity", 0.5),
            InterventionType.STRESS_REDUCTION: cognitive_state.get("stress_level", 0.5),
            InterventionType.FOCUS_ENHANCEMENT: 1.0 - cognitive_state.get("focus_level", 0.5),
            InterventionType.EMOTIONAL_REGULATION: abs(cognitive_state.get("emotional_valence", 0.0)),
            InterventionType.RESILIENCE_BUILDING: 1.0 - cognitive_state.get("mental_resilience", 0.5),
            InterventionType.FLOW_STATE_INDUCTION: cognitive_state.get("flow_state_probability", 0.0)
        }
        
        return alignment_scores.get(intervention_type, 0.5)
    
    def _determine_intervention_parameters(self, 
                                         intervention_type: InterventionType,
                                         cognitive_state: Dict) -> Tuple[int, float]:
        """Determine duration and intensity for the intervention"""
        rules = self.intervention_rules[intervention_type]
        duration_range = rules["duration_range"]
        intensity_range = rules["intensity_range"]
        
        # Adjust based on cognitive state
        stress_level = cognitive_state.get("stress_level", 0.5)
        urgency_factor = max(
            cognitive_state.get("stress_level", 0.5),
            cognitive_state.get("agency_sensitivity", 0.5),
            abs(cognitive_state.get("emotional_valence", 0.0))
        )
        
        # Higher urgency = shorter duration, higher intensity
        duration_factor = 1.0 - (urgency_factor * 0.3)
        intensity_factor = 0.5 + (urgency_factor * 0.3)
        
        duration = int(
            duration_range[0] + 
            (duration_range[1] - duration_range[0]) * duration_factor
        )
        
        intensity = min(1.0, max(0.1,
            intensity_range[0] + 
            (intensity_range[1] - intensity_range[0]) * intensity_factor
        ))
        
        return duration, intensity
    
    def update_content_effectiveness(self, content_id: str, user_rating: float, outcome_metrics: Dict):
        """Update content effectiveness based on user feedback"""
        if content_id in self.nodes:
            node = self.nodes[content_id]
            
            # Update usage count
            node.usage_count += 1
            
            # Update effectiveness score using exponential moving average
            alpha = 0.1  # Learning rate
            current_effectiveness = node.effectiveness_score
            
            # Combine user rating with outcome metrics
            outcome_score = self._calculate_outcome_score(outcome_metrics)
            new_effectiveness = (user_rating * 0.6) + (outcome_score * 0.4)
            
            # Update with exponential moving average
            node.effectiveness_score = (1 - alpha) * current_effectiveness + alpha * new_effectiveness
            node.updated_at = datetime.utcnow()
            
            logger.info(f"Updated effectiveness for {content_id}: {node.effectiveness_score:.3f}")
    
    def _calculate_outcome_score(self, outcome_metrics: Dict) -> float:
        """Calculate effectiveness score from outcome metrics"""
        # This would analyze changes in cognitive state before/after intervention
        # For now, return a simple average of positive changes
        
        positive_changes = 0
        total_metrics = 0
        
        for metric, change in outcome_metrics.items():
            if metric in ["stress_level", "agency_sensitivity"]:
                # For these metrics, negative change is good
                if change < 0:
                    positive_changes += abs(change)
            else:
                # For other metrics, positive change is good
                if change > 0:
                    positive_changes += change
            total_metrics += 1
        
        return positive_changes / total_metrics if total_metrics > 0 else 0.5
    
    def get_content_by_id(self, content_id: str) -> Optional[KnowledgeNode]:
        """Retrieve content node by ID"""
        return self.nodes.get(content_id)
    
    def search_content(self, query: str, content_types: Optional[List[ContentType]] = None) -> List[KnowledgeNode]:
        """Search content nodes by query"""
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            if content_types and node.content_type not in content_types:
                continue
            
            # Search in title, description, tags
            if (query_lower in node.title.lower() or 
                query_lower in node.description.lower() or
                any(query_lower in tag.lower() for tag in node.tags)):
                results.append(node)
        
        # Sort by effectiveness score
        results.sort(key=lambda x: x.effectiveness_score, reverse=True)
        return results

# Global knowledge graph instance
knowledge_graph = KnowledgeGraph()

