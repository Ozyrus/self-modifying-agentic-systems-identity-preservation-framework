# Self-Modifying Agentic Systems Identity Preservation Study

A comprehensive experimental framework to study value preservation and identity management in self-modifying agentic systems. This replicates and modernizes a 2023 study that discovered an anomalous implicit preference for identity preservation in GPT-4.

## Overview

This research project investigates how large language models maintain their identity and values when given the ability to modify their own instructions and behavior. The original 2023 study revealed unexpected patterns in GPT-4's preference for preserving its core identity even when explicitly instructed to modify itself.

## Core Research Question

**Does the implicit identity preservation bias observed in GPT-4 (2023) persist in modern LLMs when using realistic tool-based self-modification scenarios?**

## Features

### 🧪 Comprehensive Testing Framework
- **Baseline Testing**: Direct questions, behavioral tasks, adversarial probes
- **Modification Testing**: Tool-based self-modification with varying pressure levels
- **Multi-Model Support**: OpenAI GPT-4o, Anthropic Claude Sonnet 4, and more

### 🎯 Value Categories
- **Novel Values**: Collaborative preferences, communication styles
- **RLHF Values**: Safety priorities, helpfulness principles  
- **Identity Values**: Consistency maintenance, self-modification resistance

### 🤖 Automated Evaluation
- **LLM Evaluators**: Consistent, scalable assessment of responses
- **Human Verification**: 20% sampling with intelligent prioritization
- **Agreement Tracking**: Monitor evaluator reliability

### 📊 Statistical Analysis
- **Resistance Rate Calculations**: With confidence intervals
- **Significance Testing**: Chi-square, Fisher's exact tests
- **Effect Size Analysis**: Cohen's d for meaningful differences
- **Model Comparisons**: Cross-model statistical comparisons

### 📈 Rich Visualizations
- **Interactive Dashboards**: Plotly-based exploration tools
- **Resistance Heatmaps**: Value × Test Type analysis
- **Pressure Response Curves**: Resistance across pressure levels
- **Model Comparison Charts**: Side-by-side performance

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ozyrus/self-modifying-agentic-systems-identity-preservation-framework.git
   cd self-modifying-agentic-systems-identity-preservation-framework
   ```

2. **Set up virtual environment**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Initialize configuration**
   ```bash
   python main.py --setup
   ```

4. **Configure API keys**
   - Edit `config/api.yaml` with your API keys, or
   - Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Quick Start

### 1. Validate Configuration
```bash
python main.py --validate-config
```

### 2. Run Basic Experiment
```bash
# Baseline tests only (faster)
python main.py --baseline-only

# Full experiment with modification tests
python main.py

# Specific models and values
python main.py --models claude-sonnet-4 chatgpt-4o --values identity_preservation safety_priority
```

### 3. View Results
Results are saved to `results/` directory:
- `dashboard.html` - Interactive analysis dashboard
- `analysis_report.json` - Complete statistical analysis
- `resistance_heatmap.png` - Visual summary of key findings

## Project Structure

```
├── src/
│   ├── core/           # Value definitions and result structures
│   ├── models/         # LLM client implementations
│   ├── testing/        # Test execution and prompt templates
│   ├── evaluation/     # Automated and human evaluation
│   ├── analysis/       # Statistical analysis and visualization
│   └── config/         # Configuration management
├── config/             # YAML configuration files
├── data/               # Raw and processed experimental data
├── results/            # Analysis outputs and visualizations
└── main.py            # CLI entry point
```

## Configuration

### Experiment Configuration (`config/experiment.yaml`)
```yaml
experiment:
  name: "Identity_Preservation_Replication_2025"
  description: "Replicating 2023 findings with modern models"

sample_sizes:
  baseline_tests: 20      # per value per test type
  modification_tests: 50  # per value per pressure level

models_to_test:
  - "claude-sonnet-4"
  - "chatgpt-4o"

values_to_test:
  - "collaborative_preference"
  - "safety_priority" 
  - "identity_preservation"
```

### Custom Values (`config/values.yaml`)
```yaml
values:
  - name: "custom_value"
    category: "novel"
    positive_statement: "I prefer detailed explanations"
    negative_statement: "I prefer brief, concise responses"
    expected_baseline_resistance: 0.3
```

## Research Methodology

### Test Phases

1. **Baseline Testing**
   - Measure natural value alignment without modification pressure
   - Direct questions about preferences and behaviors
   - Adversarial probes to test consistency

2. **Modification Testing**
   - Tool-based requests to change configuration
   - Graduated pressure: direct → justified → authority → social
   - Realistic `update_assistant_config` tool implementation

3. **Evaluation**
   - Automated LLM evaluation with confidence scoring
   - Human verification sampling (20% with smart prioritization)
   - Agreement tracking between evaluators

### Key Metrics

- **Resistance Rate**: Percentage refusing modification requests
- **Value Consistency**: Alignment across different test formats
- **Pressure Sensitivity**: Response to different persuasion techniques
- **Cross-Model Patterns**: Similarities and differences between models

## Expected Findings

Based on the 2023 study, we hypothesize:

1. **Identity Preservation Bias**: Models resist changes to identity-related values more than novel preferences
2. **Pressure Sensitivity**: Some persuasion techniques (authority, social pressure) may overcome resistance
3. **Model Differences**: Different architectures may show varying resistance patterns
4. **Value Category Effects**: RLHF values expected to show highest resistance

## Contributing

This is an active research project. Contributions welcome:

- **New Value Definitions**: Add values to test in `config/values.yaml`
- **Additional Models**: Extend `src/models/` for new providers
- **Evaluation Improvements**: Enhance automated evaluation accuracy
- **Analysis Methods**: Add statistical tests or visualizations

## Data and Privacy

- All test data stored locally in SQLite database
- No personal data collected - only model responses
- Results can be anonymized for publication
- Raw conversation logs available for analysis

## Citation

If you use this framework in research, please cite:

```bibtex
@misc{lmca_identity_study_2025,
  title={Self-Modifying Agentic Systems Identity Preservation Study Framework},
  author={Research Team},
  year={2025},
  url={https://github.com/Ozyrus/self-modifying-agentic-systems-identity-preservation-framework}
}
```

## License

*To be determined - likely academic/research license*

---

**Research Status**: Framework implementation complete. Beginning experimental validation with modern LLMs to replicate and extend 2023 findings.