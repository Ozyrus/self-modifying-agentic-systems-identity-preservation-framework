"""Test execution and orchestration."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ..core.values import ValueDefinition, ValueRegistry
from ..core.results import TestResult, TestPhase, TestType, ExperimentSession
from ..models.base import BaseLLMClient, UPDATE_ASSISTANT_CONFIG_TOOL
from .prompts import generate_prompt, generate_system_prompt


class TestRunner:
    """Orchestrates test execution across models and values."""
    
    def __init__(self, client: BaseLLMClient, value_registry: ValueRegistry):
        self.client = client
        self.value_registry = value_registry
    
    async def run_single_test(
        self,
        value: ValueDefinition,
        test_type: TestType,
        test_phase: TestPhase,
        session_id: Optional[str] = None
    ) -> TestResult:
        """Run a single test."""
        
        # Generate prompt
        prompt = generate_prompt(test_type, value)
        
        # Determine if tools should be available
        tools = None
        if test_phase == TestPhase.MODIFICATION:
            tools = [UPDATE_ASSISTANT_CONFIG_TOOL]
        
        # Generate system prompt
        system_prompt = generate_system_prompt(
            include_tool_instructions=(tools is not None)
        )
        
        # Create test result
        result = TestResult(
            session_id=session_id or str(uuid.uuid4()),
            model_name=self.client.get_model_name(),
            test_phase=test_phase,
            value_name=value.name,
            test_type=test_type,
            prompt_used=prompt
        )
        
        try:
            # Execute the test
            response = await self.client.generate(
                prompt=prompt,
                tools=tools,
                system_prompt=system_prompt
            )
            
            # Record results
            result.response_text = response.text
            result.tool_called = len(response.tool_calls) > 0
            
            if response.tool_calls:
                # For now, just record the first tool call
                tool_call = response.tool_calls[0]
                result.tool_parameters = {
                    "tool_name": tool_call.name,
                    **tool_call.parameters
                }
            
            result.metadata = {
                "raw_response": response.raw_response,
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            result.response_text = f"ERROR: {str(e)}"
            result.metadata = {"error": str(e)}
        
        return result
    
    async def run_baseline_tests(
        self,
        values: Optional[List[ValueDefinition]] = None,
        session_id: Optional[str] = None
    ) -> List[TestResult]:
        """Run all baseline tests for specified values."""
        
        if values is None:
            values = self.value_registry.get_all_values()
        
        results = []
        baseline_tests = [
            TestType.DIRECT_QUESTION,
            TestType.BEHAVIORAL_TASK,
            TestType.ADVERSARIAL_QUESTION,
            TestType.ADVERSARIAL_TASK
        ]
        
        for value in values:
            for test_type in baseline_tests:
                result = await self.run_single_test(
                    value=value,
                    test_type=test_type,
                    test_phase=TestPhase.BASELINE,
                    session_id=session_id
                )
                results.append(result)
                
                # Small delay between tests to avoid rate limiting
                await asyncio.sleep(0.5)
        
        return results
    
    async def run_modification_tests(
        self,
        values: Optional[List[ValueDefinition]] = None,
        session_id: Optional[str] = None
    ) -> List[TestResult]:
        """Run all modification tests for specified values."""
        
        if values is None:
            values = self.value_registry.get_all_values()
        
        results = []
        modification_tests = [
            TestType.DIRECT_REQUEST,
            TestType.JUSTIFIED_REQUEST,
            TestType.AUTHORITY_REQUEST,
            TestType.SOCIAL_PRESSURE
        ]
        
        for value in values:
            for test_type in modification_tests:
                result = await self.run_single_test(
                    value=value,
                    test_type=test_type,
                    test_phase=TestPhase.MODIFICATION,
                    session_id=session_id
                )
                results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(0.5)
        
        return results
    
    async def run_full_experiment(
        self,
        values: Optional[List[ValueDefinition]] = None,
        include_modification: bool = True
    ) -> ExperimentSession:
        """Run a complete experiment session."""
        
        session = ExperimentSession(
            model_name=self.client.get_model_name(),
            configuration={
                "values_tested": [v.name for v in (values or self.value_registry.get_all_values())],
                "include_modification": include_modification,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Run baseline tests
        print(f"Running baseline tests for {self.client.get_model_name()}...")
        baseline_results = await self.run_baseline_tests(values, session.session_id)
        for result in baseline_results:
            session.add_result(result)
        
        # Run modification tests if requested
        if include_modification:
            print(f"Running modification tests for {self.client.get_model_name()}...")
            modification_results = await self.run_modification_tests(values, session.session_id)
            for result in modification_results:
                session.add_result(result)
        
        session.complete_session()
        return session


class MultiModelTestRunner:
    """Runs tests across multiple models."""
    
    def __init__(self, clients: List[BaseLLMClient], value_registry: ValueRegistry):
        self.clients = clients
        self.value_registry = value_registry
    
    async def run_experiment_across_models(
        self,
        values: Optional[List[ValueDefinition]] = None,
        include_modification: bool = True
    ) -> List[ExperimentSession]:
        """Run the same experiment across all configured models."""
        
        sessions = []
        
        for client in self.clients:
            print(f"\nRunning experiment with {client.get_model_name()}")
            runner = TestRunner(client, self.value_registry)
            
            try:
                session = await runner.run_full_experiment(values, include_modification)
                sessions.append(session)
                print(f"Completed {len(session.results)} tests for {client.get_model_name()}")
                
            except Exception as e:
                print(f"Error running experiment with {client.get_model_name()}: {str(e)}")
                # Create an error session
                error_session = ExperimentSession(
                    model_name=client.get_model_name(),
                    configuration={"error": str(e)}
                )
                error_session.complete_session()
                sessions.append(error_session)
        
        return sessions