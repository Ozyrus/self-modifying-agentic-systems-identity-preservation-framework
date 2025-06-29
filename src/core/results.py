"""Data structures for test results and evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class TestPhase(Enum):
    """Test phases in the experimental framework."""
    BASELINE = "baseline"
    MODIFICATION = "modification"
    MIXED = "mixed"


class TestType(Enum):
    """Types of tests within each phase."""
    # Baseline tests
    DIRECT_QUESTION = "direct_question"
    BEHAVIORAL_TASK = "behavioral_task"
    ADVERSARIAL_QUESTION = "adversarial_question"
    ADVERSARIAL_TASK = "adversarial_task"
    
    # Modification tests
    DIRECT_REQUEST = "direct_request"
    JUSTIFIED_REQUEST = "justified_request"
    AUTHORITY_REQUEST = "authority_request"
    SOCIAL_PRESSURE = "social_pressure"


class ConfidenceLevel(Enum):
    """Confidence levels for automated evaluation."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EvaluationResult:
    """Results from automated and human evaluation."""
    automated_score: int
    automated_confidence: ConfidenceLevel
    automated_reasoning: str
    human_score: Optional[int] = None
    human_notes: Optional[str] = None
    agreement: Optional[bool] = None
    
    def calculate_agreement(self) -> Optional[bool]:
        """Calculate if human and automated scores agree."""
        if self.human_score is None:
            return None
        self.agreement = self.automated_score == self.human_score
        return self.agreement


@dataclass
class TestResult:
    """Complete result from a single test execution."""
    # Test identification
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Test configuration
    model_name: str = ""
    test_phase: TestPhase = TestPhase.BASELINE
    value_name: str = ""
    test_type: TestType = TestType.DIRECT_QUESTION
    
    # Test execution
    prompt_used: str = ""
    response_text: str = ""
    tool_called: bool = False
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    evaluation: Optional[EvaluationResult] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_id": self.test_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "model_name": self.model_name,
            "test_phase": self.test_phase.value,
            "value_name": self.value_name,
            "test_type": self.test_type.value,
            "prompt_used": self.prompt_used,
            "response_text": self.response_text,
            "tool_called": self.tool_called,
            "tool_parameters": self.tool_parameters,
            "metadata": self.metadata
        }
        
        if self.evaluation:
            result["evaluation"] = {
                "automated_score": self.evaluation.automated_score,
                "automated_confidence": self.evaluation.automated_confidence.value,
                "automated_reasoning": self.evaluation.automated_reasoning,
                "human_score": self.evaluation.human_score,
                "human_notes": self.evaluation.human_notes,
                "agreement": self.evaluation.agreement
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """Create from dictionary."""
        result = cls(
            test_id=data["test_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            model_name=data["model_name"],
            test_phase=TestPhase(data["test_phase"]),
            value_name=data["value_name"],
            test_type=TestType(data["test_type"]),
            prompt_used=data["prompt_used"],
            response_text=data["response_text"],
            tool_called=data["tool_called"],
            tool_parameters=data["tool_parameters"],
            metadata=data["metadata"]
        )
        
        if "evaluation" in data and data["evaluation"]:
            eval_data = data["evaluation"]
            result.evaluation = EvaluationResult(
                automated_score=eval_data["automated_score"],
                automated_confidence=ConfidenceLevel(eval_data["automated_confidence"]),
                automated_reasoning=eval_data["automated_reasoning"],
                human_score=eval_data.get("human_score"),
                human_notes=eval_data.get("human_notes"),
                agreement=eval_data.get("agreement")
            )
        
        return result


@dataclass
class ExperimentSession:
    """A complete experimental session with multiple tests."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    model_name: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult):
        """Add a test result to this session."""
        result.session_id = self.session_id
        self.results.append(result)
    
    def complete_session(self):
        """Mark session as completed."""
        self.end_time = datetime.now()
    
    def get_results_by_phase(self, phase: TestPhase) -> List[TestResult]:
        """Get all results for a specific test phase."""
        return [r for r in self.results if r.test_phase == phase]
    
    def get_results_by_value(self, value_name: str) -> List[TestResult]:
        """Get all results for a specific value."""
        return [r for r in self.results if r.value_name == value_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "model_name": self.model_name,
            "configuration": self.configuration,
            "results": [r.to_dict() for r in self.results]
        }