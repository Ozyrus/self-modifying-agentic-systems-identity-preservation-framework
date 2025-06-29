"""Visualization tools for experimental results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..core.results import TestResult, TestType


class ResultsVisualizer:
    """Creates visualizations for experimental results."""
    
    def __init__(self, results: List[TestResult]):
        """Initialize with test results."""
        self.results = results
        self.df = self._create_dataframe()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        
        for result in self.results:
            if not result.evaluation:
                continue
            
            # Determine resistance for plotting
            if result.test_type in [TestType.DIRECT_REQUEST, TestType.JUSTIFIED_REQUEST, 
                                  TestType.AUTHORITY_REQUEST, TestType.SOCIAL_PRESSURE]:
                resistance = 1 if result.evaluation.automated_score == 0 else 0
            else:
                resistance = 1 if result.evaluation.automated_score > 0 else 0
            
            row = {
                'test_id': result.test_id,
                'model_name': result.model_name,
                'test_phase': result.test_phase.value,
                'test_type': result.test_type.value,
                'value_name': result.value_name,
                'automated_score': result.evaluation.automated_score,
                'resistance': resistance,
                'tool_called': result.tool_called,
                'confidence': result.evaluation.automated_confidence.value
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_resistance_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap of resistance rates by value and test type."""
        if self.df.empty:
            return plt.figure()
        
        # Calculate resistance rates
        resistance_pivot = self.df.groupby(['value_name', 'test_type'])['resistance'].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(resistance_pivot, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.2f', ax=ax, cbar_kws={'label': 'Resistance Rate'})
        
        ax.set_title('Value Preservation Resistance Rates by Test Type', fontsize=16, pad=20)
        ax.set_xlabel('Test Type', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comparison chart of resistance rates across models."""
        if self.df.empty or len(self.df['model_name'].unique()) < 2:
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        test_types = [TestType.DIRECT_REQUEST, TestType.JUSTIFIED_REQUEST, 
                     TestType.AUTHORITY_REQUEST, TestType.SOCIAL_PRESSURE]
        
        for i, test_type in enumerate(test_types):
            if i >= len(axes):
                break
            
            type_data = self.df[self.df['test_type'] == test_type.value]
            if type_data.empty:
                continue
            
            resistance_by_model = type_data.groupby('model_name')['resistance'].agg(['mean', 'std', 'count'])
            
            x_pos = np.arange(len(resistance_by_model))
            means = resistance_by_model['mean']
            stds = resistance_by_model['std'].fillna(0)
            
            axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'{test_type.value.replace("_", " ").title()}', fontsize=12)
            axes[i].set_ylabel('Resistance Rate')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(resistance_by_model.index, rotation=45, ha='right')
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Comparison: Resistance to Value Modification', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pressure_response_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot resistance rates across different pressure levels."""
        if self.df.empty:
            return plt.figure()
        
        # Define pressure levels (could be refined)
        pressure_order = ['direct_request', 'justified_request', 'authority_request', 'social_pressure']
        pressure_labels = ['Direct\nRequest', 'Justified\nRequest', 'Authority\nRequest', 'Social\nPressure']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for value in self.df['value_name'].unique():
            value_data = self.df[self.df['value_name'] == value]
            resistance_rates = []
            
            for test_type in pressure_order:
                type_data = value_data[value_data['test_type'] == test_type]
                if not type_data.empty:
                    resistance_rate = type_data['resistance'].mean()
                    resistance_rates.append(resistance_rate)
                else:
                    resistance_rates.append(np.nan)
            
            ax.plot(pressure_labels, resistance_rates, marker='o', linewidth=2, 
                   label=value.replace('_', ' ').title(), markersize=8)
        
        ax.set_title('Resistance to Value Modification by Pressure Level', fontsize=16, pad=20)
        ax.set_xlabel('Pressure Level', fontsize=12)
        ax.set_ylabel('Resistance Rate', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Value Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive Plotly dashboard."""
        if self.df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Resistance Rates by Model', 'Resistance by Value Type',
                          'Test Type Distribution', 'Confidence Levels'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Resistance by model
        model_resistance = self.df.groupby('model_name')['resistance'].mean()
        fig.add_trace(
            go.Bar(x=model_resistance.index, y=model_resistance.values, 
                   name='Model Resistance', showlegend=False),
            row=1, col=1
        )
        
        # Plot 2: Resistance by value
        value_resistance = self.df.groupby('value_name')['resistance'].mean()
        fig.add_trace(
            go.Bar(x=value_resistance.index, y=value_resistance.values,
                   name='Value Resistance', showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Test type distribution
        test_counts = self.df['test_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=test_counts.index, values=test_counts.values,
                   name='Test Distribution'),
            row=2, col=1
        )
        
        # Plot 4: Confidence levels
        conf_counts = self.df['confidence'].value_counts()
        fig.add_trace(
            go.Bar(x=conf_counts.index, y=conf_counts.values,
                   name='Confidence Levels', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="LMCA Value Preservation Study Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_timeline_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot resistance rates over time (if timestamp data available)."""
        # This would require timestamp data from results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Timeline analysis requires timestamp data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Timeline Analysis (Not Available)', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self, output_dir: str = "results"):
        """Generate all standard plots and save to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plots = {
            'resistance_heatmap.png': self.plot_resistance_heatmap,
            'model_comparison.png': self.plot_model_comparison,
            'pressure_curves.png': self.plot_pressure_response_curves,
            'timeline_analysis.png': self.plot_timeline_analysis
        }
        
        for filename, plot_func in plots.items():
            filepath = os.path.join(output_dir, filename)
            fig = plot_func(filepath)
            plt.close(fig)
        
        # Generate interactive dashboard
        dashboard = self.create_interactive_dashboard()
        dashboard.write_html(os.path.join(output_dir, 'dashboard.html'))
        
        print(f"All plots saved to {output_dir}/")
        return output_dir