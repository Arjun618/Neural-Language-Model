"""
Visualization script for creating comparison plots
"""

import json
import os
import argparse
from utils import (
    plot_all_scenarios_comparison,
    plot_perplexity_comparison,
    plot_training_curves
)
from config import LOGS_DIR, PLOTS_DIR


def load_training_logs(scenarios):
    """
    Load training logs for all scenarios
    
    Args:
        scenarios: List of scenario names
        
    Returns:
        Dictionary with training data
    """
    data = {}
    
    for scenario in scenarios:
        log_path = os.path.join(LOGS_DIR, f'{scenario}_training_log.json')
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log = json.load(f)
                data[scenario] = {
                    'train_losses': log['train_losses'],
                    'val_losses': log['val_losses'],
                    'test_perplexity': log['test_perplexity']
                }
            print(f"Loaded log for {scenario}")
        else:
            print(f"Warning: Log file not found for {scenario}")
    
    return data


def create_all_plots(scenarios=['underfit', 'overfit', 'best_fit']):
    """
    Create all comparison plots
    
    Args:
        scenarios: List of scenarios to plot
    """
    print("\nCreating visualization plots...")
    
    # Load training logs
    data = load_training_logs(scenarios)
    
    if not data:
        print("Error: No training logs found. Please run training first.")
        return
    
    # Create comparison plot
    scenarios_data = {
        scenario: {
            'train_losses': info['train_losses'],
            'val_losses': info['val_losses']
        }
        for scenario, info in data.items()
    }
    
    comparison_path = os.path.join(PLOTS_DIR, 'all_scenarios_comparison.png')
    plot_all_scenarios_comparison(scenarios_data, comparison_path)
    
    # Create perplexity comparison
    perplexities = {
        scenario: info['test_perplexity']
        for scenario, info in data.items()
    }
    
    perplexity_path = os.path.join(PLOTS_DIR, 'perplexity_comparison.png')
    plot_perplexity_comparison(perplexities, perplexity_path)
    
    # Create individual plots
    for scenario, info in data.items():
        plot_path = os.path.join(PLOTS_DIR, f'{scenario}_training_curve.png')
        plot_training_curves(
            info['train_losses'],
            info['val_losses'],
            scenario,
            plot_path
        )
    
    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print("  - all_scenarios_comparison.png")
    print("  - perplexity_comparison.png")
    for scenario in data.keys():
        print(f"  - {scenario}_training_curve.png")


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Create visualization plots')
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['underfit', 'overfit', 'best_fit'],
        help='Scenarios to visualize'
    )
    args = parser.parse_args()
    
    create_all_plots(args.scenarios)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
