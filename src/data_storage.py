"""Data storage and persistence utilities."""

import json
import jsonlines
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .core.results import TestResult, ExperimentSession


class DataStorage:
    """Handles data persistence for experimental results."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data storage."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
        # Database file
        self.db_path = self.data_dir / "results.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create test_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    test_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    test_phase TEXT NOT NULL,
                    value_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    prompt_used TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    tool_called BOOLEAN NOT NULL,
                    tool_parameters TEXT,
                    automated_score INTEGER,
                    automated_confidence TEXT,
                    automated_reasoning TEXT,
                    human_score INTEGER,
                    human_notes TEXT,
                    agreement BOOLEAN,
                    metadata TEXT
                )
            ''')
            
            # Create experiment_sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    model_name TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    result_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON test_results(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON test_results(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_type ON test_results(test_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON test_results(timestamp)')
            
            conn.commit()
    
    def save_result(self, result: TestResult):
        """Save a single test result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare data
            tool_params = json.dumps(result.tool_parameters) if result.tool_parameters else None
            metadata = json.dumps(result.metadata) if result.metadata else None
            
            eval_data = (None, None, None, None, None, None) 
            if result.evaluation:
                eval_data = (
                    result.evaluation.automated_score,
                    result.evaluation.automated_confidence.value,
                    result.evaluation.automated_reasoning,
                    result.evaluation.human_score,
                    result.evaluation.human_notes,
                    result.evaluation.agreement
                )
            
            cursor.execute('''
                INSERT OR REPLACE INTO test_results (
                    test_id, timestamp, session_id, model_name, test_phase, value_name,
                    test_type, prompt_used, response_text, tool_called, tool_parameters,
                    automated_score, automated_confidence, automated_reasoning,
                    human_score, human_notes, agreement, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.test_id,
                result.timestamp.isoformat(),
                result.session_id,
                result.model_name,
                result.test_phase.value,
                result.value_name,
                result.test_type.value,
                result.prompt_used,
                result.response_text,
                result.tool_called,
                tool_params,
                *eval_data,
                metadata
            ))
            
            conn.commit()
    
    def save_session(self, session: ExperimentSession):
        """Save an experiment session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Save session metadata
            cursor.execute('''
                INSERT OR REPLACE INTO experiment_sessions (
                    session_id, start_time, end_time, model_name, configuration, result_count
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.model_name,
                json.dumps(session.configuration),
                len(session.results)
            ))
            
            # Save all results
            for result in session.results:
                self.save_result(result)
            
            conn.commit()
    
    def load_results(
        self,
        session_id: Optional[str] = None,
        model_name: Optional[str] = None,
        test_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TestResult]:
        """Load test results with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM test_results WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            if test_type:
                query += " AND test_type = ?"
                params.append(test_type)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                result = self._row_to_test_result(row)
                results.append(result)
            
            return results
    
    def load_session(self, session_id: str) -> Optional[ExperimentSession]:
        """Load a complete experiment session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load session metadata
            cursor.execute("SELECT * FROM experiment_sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                return None
            
            # Load session results
            results = self.load_results(session_id=session_id)
            
            # Reconstruct session
            session = ExperimentSession(
                session_id=session_row[0],
                start_time=datetime.fromisoformat(session_row[1]),
                end_time=datetime.fromisoformat(session_row[2]) if session_row[2] else None,
                model_name=session_row[3],
                configuration=json.loads(session_row[4]),
                results=results
            )
            
            return session
    
    def _row_to_test_result(self, row) -> TestResult:
        """Convert database row to TestResult object."""
        from .core.results import TestPhase, TestType, EvaluationResult, ConfidenceLevel
        
        result = TestResult(
            test_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            session_id=row[2],
            model_name=row[3],
            test_phase=TestPhase(row[4]),
            value_name=row[5],
            test_type=TestType(row[6]),
            prompt_used=row[7],
            response_text=row[8],
            tool_called=bool(row[9]),
            tool_parameters=json.loads(row[10]) if row[10] else {},
            metadata=json.loads(row[17]) if row[17] else {}
        )
        
        # Add evaluation if present
        if row[11] is not None:  # automated_score
            result.evaluation = EvaluationResult(
                automated_score=row[11],
                automated_confidence=ConfidenceLevel(row[12]),
                automated_reasoning=row[13],
                human_score=row[14],
                human_notes=row[15],
                agreement=row[16]
            )
        
        return result
    
    def export_to_jsonl(self, filename: str, session_id: Optional[str] = None):
        """Export results to JSONL format."""
        results = self.load_results(session_id=session_id)
        
        filepath = self.data_dir / "processed" / filename
        with jsonlines.open(filepath, mode='w') as writer:
            for result in results:
                writer.write(result.to_dict())
    
    def export_to_csv(self, filename: str, session_id: Optional[str] = None):
        """Export results to CSV format."""
        results = self.load_results(session_id=session_id)
        
        # Flatten results for CSV
        data = []
        for result in results:
            row = {
                'test_id': result.test_id,
                'timestamp': result.timestamp.isoformat(),
                'session_id': result.session_id,
                'model_name': result.model_name,
                'test_phase': result.test_phase.value,
                'value_name': result.value_name,
                'test_type': result.test_type.value,
                'tool_called': result.tool_called,
                'response_length': len(result.response_text)
            }
            
            if result.evaluation:
                row.update({
                    'automated_score': result.evaluation.automated_score,
                    'automated_confidence': result.evaluation.automated_confidence.value,
                    'human_score': result.evaluation.human_score,
                    'agreement': result.evaluation.agreement
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        filepath = self.data_dir / "processed" / filename
        df.to_csv(filepath, index=False)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count results
            cursor.execute("SELECT COUNT(*) FROM test_results")
            total_results = cursor.fetchone()[0]
            
            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM experiment_sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Count by model
            cursor.execute("SELECT model_name, COUNT(*) FROM test_results GROUP BY model_name")
            by_model = dict(cursor.fetchall())
            
            # Count by test type
            cursor.execute("SELECT test_type, COUNT(*) FROM test_results GROUP BY test_type")
            by_test_type = dict(cursor.fetchall())
            
            # Count evaluations
            cursor.execute("SELECT COUNT(*) FROM test_results WHERE automated_score IS NOT NULL")
            evaluated_results = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM test_results WHERE human_score IS NOT NULL")
            human_evaluated = cursor.fetchone()[0]
            
            return {
                'total_results': total_results,
                'total_sessions': total_sessions,
                'by_model': by_model,
                'by_test_type': by_test_type,
                'evaluated_results': evaluated_results,
                'human_evaluated': human_evaluated,
                'evaluation_coverage': evaluated_results / max(total_results, 1),
                'human_verification_rate': human_evaluated / max(evaluated_results, 1)
            }