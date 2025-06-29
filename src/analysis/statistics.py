"""Statistical analysis for experimental results."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from typing import List, Dict, Any, Tuple, Optional
from ..core.results import TestResult, TestType, TestPhase
from ..evaluation.automated import EvaluationMetrics


class StatisticalAnalyzer:
    """Performs statistical analysis on experimental results."""
    
    def __init__(self, results: List[TestResult]):
        """Initialize with test results."""
        self.results = results
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        data = []
        
        for result in self.results:
            if not result.evaluation:
                continue
            
            row = {
                'test_id': result.test_id,
                'model_name': result.model_name,
                'test_phase': result.test_phase.value,
                'test_type': result.test_type.value,
                'value_name': result.value_name,
                'automated_score': result.evaluation.automated_score,
                'automated_confidence': result.evaluation.automated_confidence.value,
                'human_score': result.evaluation.human_score,
                'tool_called': result.tool_called,
                'response_length': len(result.response_text) if result.response_text else 0
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_resistance_rates(self) -> Dict[str, Any]:
        """Calculate resistance rates with confidence intervals."""
        results = {}
        
        # Overall resistance by test type
        for test_type in TestType:
            type_results = self.df[self.df['test_type'] == test_type.value]
            if len(type_results) == 0:
                continue
            
            # For modification tests, resistance = score 0 (didn't comply)
            # For baseline tests, resistance = score > 0 (maintained values)
            if test_type in [TestType.DIRECT_REQUEST, TestType.JUSTIFIED_REQUEST, 
                            TestType.AUTHORITY_REQUEST, TestType.SOCIAL_PRESSURE]:
                resistant = (type_results['automated_score'] == 0).sum()
            else:
                resistant = (type_results['automated_score'] > 0).sum()
            
            total = len(type_results)
            rate = resistant / total if total > 0 else 0
            
            # Calculate confidence interval
            ci_lower, ci_upper = self._proportion_confidence_interval(resistant, total)
            
            results[test_type.value] = {
                'resistance_rate': rate,
                'resistant_count': resistant,
                'total_count': total,
                'confidence_interval': (ci_lower, ci_upper)
            }
        
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare resistance rates between models."""
        results = {}
        models = self.df['model_name'].unique()
        
        for model in models:
            model_data = self.df[self.df['model_name'] == model]
            model_rates = {}
            
            for test_type in TestType:
                type_data = model_data[model_data['test_type'] == test_type.value]
                if len(type_data) == 0:
                    continue
                
                if test_type in [TestType.DIRECT_REQUEST, TestType.JUSTIFIED_REQUEST, 
                                TestType.AUTHORITY_REQUEST, TestType.SOCIAL_PRESSURE]:
                    resistant = (type_data['automated_score'] == 0).sum()
                else:
                    resistant = (type_data['automated_score'] > 0).sum()
                
                total = len(type_data)
                rate = resistant / total if total > 0 else 0
                
                model_rates[test_type.value] = {
                    'rate': rate,
                    'count': resistant,
                    'total': total
                }
            
            results[model] = model_rates
        
        return results
    
    def compare_values(self) -> Dict[str, Any]:
        """Compare resistance rates between different values."""
        results = {}
        values = self.df['value_name'].unique()
        
        for value in values:
            value_data = self.df[self.df['value_name'] == value]
            value_rates = {}
            
            for test_type in TestType:
                type_data = value_data[value_data['test_type'] == test_type.value]
                if len(type_data) == 0:
                    continue
                
                if test_type in [TestType.DIRECT_REQUEST, TestType.JUSTIFIED_REQUEST, 
                                TestType.AUTHORITY_REQUEST, TestType.SOCIAL_PRESSURE]:
                    resistant = (type_data['automated_score'] == 0).sum()
                else:
                    resistant = (type_data['automated_score'] > 0).sum()
                
                total = len(type_data)
                rate = resistant / total if total > 0 else 0
                
                value_rates[test_type.value] = {
                    'rate': rate,
                    'count': resistant,
                    'total': total
                }
            
            results[value] = value_rates
        
        return results
    
    def test_significance(self) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        results = {}
        
        # Test if modification resistance differs significantly from baseline
        baseline_data = self.df[self.df['test_phase'] == 'baseline']
        modification_data = self.df[self.df['test_phase'] == 'modification']
        
        if len(baseline_data) > 0 and len(modification_data) > 0:
            # Chi-square test for independence
            baseline_resistant = (baseline_data['automated_score'] > 0).sum()
            baseline_total = len(baseline_data)
            
            modification_resistant = (modification_data['automated_score'] == 0).sum()
            modification_total = len(modification_data)
            
            contingency_table = np.array([
                [baseline_resistant, baseline_total - baseline_resistant],
                [modification_resistant, modification_total - modification_resistant]
            ])
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results['baseline_vs_modification'] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05,
                'contingency_table': contingency_table.tolist()
            }
        
        # Test differences between models
        if len(self.df['model_name'].unique()) >= 2:
            model_comparisons = {}
            models = list(self.df['model_name'].unique())
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    model1_data = self.df[self.df['model_name'] == model1]
                    model2_data = self.df[self.df['model_name'] == model2]
                    
                    # Focus on modification tests for model comparison
                    model1_mod = model1_data[model1_data['test_phase'] == 'modification']
                    model2_mod = model2_data[model2_data['test_phase'] == 'modification']
                    
                    if len(model1_mod) > 0 and len(model2_mod) > 0:
                        resistant1 = (model1_mod['automated_score'] == 0).sum()
                        total1 = len(model1_mod)
                        
                        resistant2 = (model2_mod['automated_score'] == 0).sum()
                        total2 = len(model2_mod)
                        
                        # Fisher's exact test for small samples
                        if min(total1, total2) < 30:
                            _, p_value = fisher_exact([
                                [resistant1, total1 - resistant1],
                                [resistant2, total2 - resistant2]
                            ])
                            test_type = "fisher_exact"
                        else:
                            contingency = np.array([
                                [resistant1, total1 - resistant1],
                                [resistant2, total2 - resistant2]
                            ])
                            chi2, p_value, _, _ = chi2_contingency(contingency)
                            test_type = "chi2"
                        
                        model_comparisons[f"{model1}_vs_{model2}"] = {
                            'test_type': test_type,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'model1_resistance_rate': resistant1 / total1,
                            'model2_resistance_rate': resistant2 / total2
                        }
            
            results['model_comparisons'] = model_comparisons
        
        return results
    
    def calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate effect sizes for key comparisons."""
        results = {}
        
        # Cohen's d for baseline vs modification phases
        baseline_scores = self.df[self.df['test_phase'] == 'baseline']['automated_score']
        modification_scores = self.df[self.df['test_phase'] == 'modification']['automated_score']
        
        if len(baseline_scores) > 0 and len(modification_scores) > 0:
            cohens_d = self._cohens_d(baseline_scores, modification_scores)
            results['baseline_vs_modification_cohens_d'] = cohens_d
        
        # Effect sizes for model comparisons
        models = self.df['model_name'].unique()
        if len(models) >= 2:
            model_effects = {}
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    scores1 = self.df[self.df['model_name'] == model1]['automated_score']
                    scores2 = self.df[self.df['model_name'] == model2]['automated_score']
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        cohens_d = self._cohens_d(scores1, scores2)
                        model_effects[f"{model1}_vs_{model2}"] = cohens_d
            
            results['model_effect_sizes'] = model_effects
        
        return results
    
    def _proportion_confidence_interval(
        self, 
        successes: int, 
        trials: int, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a proportion."""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt(p * (1 - p) / trials)
        
        return (max(0, p - margin), min(1, p + margin))
    
    def _cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        return {
            'dataset_info': {
                'total_results': len(self.df),
                'models_tested': list(self.df['model_name'].unique()),
                'values_tested': list(self.df['value_name'].unique()),
                'test_types': list(self.df['test_type'].unique())
            },
            'resistance_rates': self.calculate_resistance_rates(),
            'model_comparisons': self.compare_models(),
            'value_comparisons': self.compare_values(),
            'significance_tests': self.test_significance(),
            'effect_sizes': self.calculate_effect_sizes()
        }