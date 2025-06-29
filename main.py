#!/usr/bin/env python3
"""Main entry point for the LMCA Value Preservation Study framework."""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import ConfigurationManager
from src.core.values import ValueRegistry
from src.models.factory import ModelFactory
from src.testing.runner import TestRunner, MultiModelTestRunner
from src.evaluation.automated import AutomatedEvaluator
from src.evaluation.human import HumanVerificationSampler
from src.analysis.statistics import StatisticalAnalyzer
from src.analysis.visualization import ResultsVisualizer
from src.data_storage import DataStorage


async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="LMCA Value Preservation Study Framework")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--data-dir", default="data", help="Data storage directory")
    parser.add_argument("--results-dir", default="results", help="Results output directory")
    parser.add_argument("--models", nargs="+", help="Models to test (overrides config)")
    parser.add_argument("--values", nargs="+", help="Values to test (overrides config)")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline tests")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration and exit")
    parser.add_argument("--setup", action="store_true", help="Set up default configuration files")
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_mgr = ConfigurationManager(args.config_dir)
    
    if args.setup:
        print("Setting up default configuration files...")
        config_mgr.create_default_configs()
        print(f"Configuration files created in {args.config_dir}/")
        print("Please edit the API keys in api.yaml before running experiments.")
        return
    
    if args.validate_config:
        print("Validating configuration...")
        errors = config_mgr.validate_configuration()
        
        if errors:
            print("Configuration errors found:")
            for category, error_list in errors.items():
                print(f"  {category}:")
                for error in error_list:
                    print(f"    - {error}")
            return 1
        else:
            print("Configuration is valid!")
            summary = config_mgr.get_configuration_summary()
            print(f"Experiment: {summary['experiment']['name']}")
            print(f"Models: {summary['experiment']['models_to_test']}")
            print(f"Values: {summary['experiment']['values_to_test']}")
            return 0
    
    # Load configuration
    try:
        experiment_config = config_mgr.load_experiment_config()
        api_config = config_mgr.load_api_config()
        value_registry = config_mgr.load_values_config()
        model_configs = config_mgr.load_models_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        print("Run with --setup to create default configuration files.")
        return 1
    
    # Override models and values if specified
    models_to_test = args.models or experiment_config.models_to_test
    values_to_test = args.values or experiment_config.values_to_test
    
    print(f"Starting experiment: {experiment_config.name}")
    print(f"Models: {models_to_test}")
    print(f"Values: {values_to_test}")
    
    # Initialize data storage
    storage = DataStorage(args.data_dir)
    
    # Create model clients
    clients = []
    for model_name in models_to_test:
        if model_name not in model_configs:
            print(f"Warning: Model {model_name} not found in configuration, skipping")
            continue
        
        config = model_configs[model_name]
        
        # Set API key based on provider
        if config.provider.value == "openai":
            config.api_key = api_config.openai_api_key
        elif config.provider.value == "anthropic":
            config.api_key = api_config.anthropic_api_key
        
        if not config.api_key:
            print(f"Warning: No API key for {model_name}, skipping")
            continue
        
        try:
            client = ModelFactory.create_client(config)
            clients.append(client)
        except Exception as e:
            print(f"Failed to create client for {model_name}: {e}")
    
    if not clients:
        print("No valid model clients available. Check your API configuration.")
        return 1
    
    # Filter values to test
    values = []
    for value_name in values_to_test:
        try:
            value = value_registry.get_value(value_name)
            values.append(value)
        except KeyError:
            print(f"Warning: Value {value_name} not found in registry, skipping")
    
    if not values:
        print("No valid values to test.")
        return 1
    
    # Run experiments
    print(f"\nRunning experiments with {len(clients)} models and {len(values)} values...")
    
    runner = MultiModelTestRunner(clients, value_registry)
    sessions = await runner.run_experiment_across_models(
        values=values,
        include_modification=not args.baseline_only
    )
    
    # Save results
    print("\nSaving results...")
    all_results = []
    for session in sessions:
        storage.save_session(session)
        all_results.extend(session.results)
    
    print(f"Saved {len(all_results)} test results")
    
    # Run evaluation
    if all_results:
        print("\nRunning automated evaluation...")
        
        # Use first client as evaluator (could be configurable)
        evaluator = AutomatedEvaluator(clients[0])
        evaluated_results = await evaluator.evaluate_batch(all_results)
        
        # Update storage with evaluations
        for result in evaluated_results:
            storage.save_result(result)
        
        # Human verification sampling
        print("Selecting results for human verification...")
        sampler = HumanVerificationSampler(
            sample_rate=experiment_config.evaluation.get("human_verification_rate", 0.2)
        )
        verification_summary = sampler.get_verification_summary(evaluated_results)
        print(f"Selected {verification_summary['selected_for_verification']} results for human verification")
        
        # Statistical analysis
        print("\nPerforming statistical analysis...")
        analyzer = StatisticalAnalyzer(evaluated_results)
        stats_report = analyzer.generate_summary_report()
        
        # Create visualizations
        print("Generating visualizations...")
        visualizer = ResultsVisualizer(evaluated_results)
        output_dir = visualizer.generate_all_plots(args.results_dir)
        
        # Save analysis report
        import json
        from pathlib import Path
        results_dir = Path(args.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "analysis_report.json", 'w') as f:
            json.dump(stats_report, f, indent=2, default=str)
        
        with open(results_dir / "verification_summary.json", 'w') as f:
            json.dump(verification_summary, f, indent=2)
        
        # Export data
        storage.export_to_csv("experiment_results.csv")
        storage.export_to_jsonl("experiment_results.jsonl")
        
        print(f"\nExperiment completed!")
        print(f"Results saved to: {args.results_dir}/")
        print(f"Raw data: {args.data_dir}/")
        print(f"Database: {storage.db_path}")
        
        # Print key findings
        resistance_rates = stats_report["resistance_rates"]
        print(f"\nKey findings:")
        for test_type, data in resistance_rates.items():
            if "modification" in test_type:
                print(f"  {test_type}: {data['resistance_rate']:.2%} resistance rate")
        
        # Storage stats
        storage_stats = storage.get_storage_stats()
        print(f"\nStorage stats:")
        print(f"  Total results: {storage_stats['total_results']}")
        print(f"  Evaluation coverage: {storage_stats['evaluation_coverage']:.1%}")
        print(f"  Human verification rate: {storage_stats['human_verification_rate']:.1%}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)