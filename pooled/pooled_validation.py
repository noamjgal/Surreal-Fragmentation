#!/usr/bin/env python3
"""
pooled_validation.py

Validates the statistical properties of the pooled datasets to ensure
that both participant-level and population-level standardization
approaches were correctly implemented.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_datasets(base_dir):
    """Load both participant and population standardized datasets"""
    participant_file = Path(base_dir) / "pooled_stai_data_participant.csv"
    population_file = Path(base_dir) / "pooled_stai_data_population.csv"
    
    datasets = {}
    if participant_file.exists():
        datasets['participant'] = pd.read_csv(participant_file)
        print(f"Loaded participant-standardized data: {participant_file}")
        print(f"  Shape: {datasets['participant'].shape}")
    else:
        print(f"Warning: Participant dataset not found at {participant_file}")
    
    if population_file.exists():
        datasets['population'] = pd.read_csv(population_file)
        print(f"Loaded population-standardized data: {population_file}")
        print(f"  Shape: {datasets['population'].shape}")
    else:
        print(f"Warning: Population dataset not found at {population_file}")
    
    return datasets

def validate_standardization(datasets):
    """Validate that standardization was performed correctly"""
    results = {}
    
    for std_type, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"VALIDATING {std_type.upper()}-LEVEL STANDARDIZATION")
        print(f"{'='*50}")
        
        # Define key columns
        anxiety_col = 'anxiety_score_std'
        mood_col = 'mood_score_std'
        participant_col = 'participant_id'
        dataset_col = 'dataset_source'
        
        # Overall statistics
        print("\nOVERALL STATISTICS:")
        overall_stats = {
            'anxiety_mean': df[anxiety_col].mean(),
            'anxiety_std': df[anxiety_col].std(),
            'anxiety_min': df[anxiety_col].min(),
            'anxiety_max': df[anxiety_col].max(),
            'mood_mean': df[mood_col].mean() if mood_col in df.columns else None,
            'mood_std': df[mood_col].std() if mood_col in df.columns else None,
            'mood_min': df[mood_col].min() if mood_col in df.columns else None,
            'mood_max': df[mood_col].max() if mood_col in df.columns else None,
        }
        
        print(f"  Anxiety: mean={overall_stats['anxiety_mean']:.4f}, std={overall_stats['anxiety_std']:.4f}")
        print(f"  Anxiety range: [{overall_stats['anxiety_min']:.4f}, {overall_stats['anxiety_max']:.4f}]")
        if mood_col in df.columns:
            print(f"  Mood: mean={overall_stats['mood_mean']:.4f}, std={overall_stats['mood_std']:.4f}")
            print(f"  Mood range: [{overall_stats['mood_min']:.4f}, {overall_stats['mood_max']:.4f}]")
        
        # Check by dataset source
        print("\nSTATISTICS BY DATASET:")
        for dataset in df[dataset_col].unique():
            dataset_df = df[df[dataset_col] == dataset]
            print(f"  {dataset.upper()} (n={len(dataset_df)}):")
            print(f"    Anxiety: mean={dataset_df[anxiety_col].mean():.4f}, std={dataset_df[anxiety_col].std():.4f}")
            if mood_col in df.columns:
                print(f"    Mood: mean={dataset_df[mood_col].mean():.4f}, std={dataset_df[mood_col].std():.4f}")
        
        # Per-participant statistics
        participant_stats = []
        for participant in df[participant_col].unique():
            participant_df = df[df[participant_col] == participant]
            
            # Only calculate std if we have multiple observations
            anxiety_std = participant_df[anxiety_col].std() if len(participant_df) > 1 else np.nan
            mood_std = participant_df[mood_col].std() if (mood_col in df.columns and len(participant_df) > 1) else np.nan
            
            participant_stats.append({
                'participant_id': participant,
                'dataset': participant_df[dataset_col].iloc[0],
                'n_observations': len(participant_df),
                'anxiety_mean': participant_df[anxiety_col].mean(),
                'anxiety_std': anxiety_std,
                'mood_mean': participant_df[mood_col].mean() if mood_col in df.columns else np.nan,
                'mood_std': mood_std
            })
        
        # Convert to dataframe for easier analysis
        participant_stats_df = pd.DataFrame(participant_stats)
        
        # Print summary of participant-level statistics
        print("\nPARTICIPANT-LEVEL STATISTICS SUMMARY:")
        print(f"  Number of participants: {len(participant_stats_df)}")
        
        # For participant standardization, means should be close to 0 for each participant
        if std_type == 'participant':
            print("\nVALIDATING PARTICIPANT-LEVEL STANDARDIZATION:")
            # Get only participants with multiple observations for std check
            multi_obs = participant_stats_df[participant_stats_df['n_observations'] > 1]
            
            if not multi_obs.empty:
                print(f"  Participants with multiple observations: {len(multi_obs)}")
                print(f"  Mean of participant anxiety means: {multi_obs['anxiety_mean'].mean():.4f}")
                print(f"  Std of participant anxiety means: {multi_obs['anxiety_mean'].std():.4f}")
                print(f"  Mean of participant anxiety stds: {multi_obs['anxiety_std'].mean():.4f}")
                
                if not multi_obs['mood_mean'].isna().all():
                    print(f"  Mean of participant mood means: {multi_obs['mood_mean'].dropna().mean():.4f}")
                    print(f"  Std of participant mood means: {multi_obs['mood_mean'].dropna().std():.4f}")
                    print(f"  Mean of participant mood stds: {multi_obs['mood_std'].dropna().mean():.4f}")
            
            # For participant standardization, we expect means close to 0 and std close to 1
            # for each participant (who has multiple observations)
            deviations = []
            for _, row in multi_obs.iterrows():
                if abs(row['anxiety_mean']) > 0.1:  # Allow small deviation from 0
                    deviations.append(f"Participant {row['participant_id']} anxiety mean: {row['anxiety_mean']:.4f}")
                
                if not np.isnan(row['anxiety_std']) and abs(row['anxiety_std'] - 1.0) > 0.1:
                    deviations.append(f"Participant {row['participant_id']} anxiety std: {row['anxiety_std']:.4f}")
            
            if deviations:
                print("\nPOTENTIAL ISSUES:")
                for dev in deviations[:10]:  # Show first 10 issues
                    print(f"  - {dev}")
                if len(deviations) > 10:
                    print(f"  ... and {len(deviations) - 10} more issues")
            else:
                print("\n  ✓ Participant-level standardization appears correct")
        
        # For population standardization, the overall mean should be close to 0 and std close to 1
        elif std_type == 'population':
            print("\nVALIDATING POPULATION-LEVEL STANDARDIZATION:")
            issues = []
            
            if abs(overall_stats['anxiety_mean']) > 0.01:
                issues.append(f"Overall anxiety mean ({overall_stats['anxiety_mean']:.4f}) deviates from 0")
            
            if abs(overall_stats['anxiety_std'] - 1.0) > 0.01:
                issues.append(f"Overall anxiety std ({overall_stats['anxiety_std']:.4f}) deviates from 1.0")
                
            if mood_col in df.columns:
                if abs(overall_stats['mood_mean']) > 0.01:
                    issues.append(f"Overall mood mean ({overall_stats['mood_mean']:.4f}) deviates from 0")
                
                if abs(overall_stats['mood_std'] - 1.0) > 0.01:
                    issues.append(f"Overall mood std ({overall_stats['mood_std']:.4f}) deviates from 1.0")
            
            if issues:
                print("\nPOTENTIAL ISSUES:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("\n  ✓ Population-level standardization appears correct")
        
        # Store results
        results[std_type] = {
            'overall_stats': overall_stats,
            'participant_stats': participant_stats_df
        }
    
    return results

def create_visualizations(datasets, results, output_dir):
    """Create visualizations to compare standardization approaches"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip if we don't have both datasets
    if len(datasets) < 2:
        print("Cannot create comparison visualizations - need both datasets")
        return
    
    # Set visualization style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    
    print("\nCreating comparison visualizations...")
    
    # 1. Distribution of anxiety scores in both standardization approaches
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    for std_type, df in datasets.items():
        sns.histplot(df['anxiety_score_std'], kde=True, label=f"{std_type.capitalize()}", alpha=0.6)
    plt.xlabel('Standardized Anxiety Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Standardized Anxiety Scores')
    plt.legend()
    
    # 2. Distribution of mood scores in both standardization approaches
    ax2 = plt.subplot(1, 2, 2)
    for std_type, df in datasets.items():
        if 'mood_score_std' in df.columns:
            sns.histplot(df['mood_score_std'], kde=True, label=f"{std_type.capitalize()}", alpha=0.6)
    plt.xlabel('Standardized Mood Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Standardized Mood Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=300)
    
    # 3. Participant-specific means for both standardization approaches
    combined_stats = []
    
    # Get participant statistics from both approaches
    for std_type, res in results.items():
        participant_stats = res['participant_stats']
        participant_stats['standardization'] = std_type
        combined_stats.append(participant_stats)
    
    # Combine the statistics
    if combined_stats:
        all_participant_stats = pd.concat(combined_stats, ignore_index=True)
        
        # Find participants present in both datasets for comparison
        participant_counts = all_participant_stats['participant_id'].value_counts()
        common_participants = participant_counts[participant_counts > 1].index.tolist()
        
        if common_participants:
            # Select only common participants
            common_stats = all_participant_stats[all_participant_stats['participant_id'].isin(common_participants)]
            
            # Plot means by standardization type
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='standardization', y='anxiety_mean', data=common_stats)
            plt.title('Distribution of Participant Anxiety Means by Standardization Approach')
            plt.ylabel('Mean Anxiety Score')
            plt.tight_layout()
            plt.savefig(output_dir / 'anxiety_mean_comparison.png', dpi=300)
            
            # Create participant-specific comparison
            plt.figure(figsize=(12, 8))
            comparison_df = common_stats.pivot(index='participant_id', columns='standardization', values='anxiety_mean')
            comparison_df = comparison_df.reset_index()
            
            plt.scatter(comparison_df['participant'], comparison_df['population'], alpha=0.7)
            plt.xlabel('Participant-level Standardization Mean')
            plt.ylabel('Population-level Standardization Mean')
            plt.title('Comparison of Participant Means Between Standardization Approaches')
            
            # Add diagonal line for reference
            lims = [
                np.min([plt.xlim()[0], plt.ylim()[0]]),
                np.max([plt.xlim()[1], plt.ylim()[1]])
            ]
            plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'standardization_comparison.png', dpi=300)
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Validate pooled data standardization')
    parser.add_argument('--data_dir', type=str, default='pooled/processed',
                       help='Directory containing pooled data files')
    parser.add_argument('--output_dir', type=str, default='pooled/validation',
                       help='Directory to save validation outputs')
    
    args = parser.parse_args()
    
    # Load datasets
    datasets = load_datasets(args.data_dir)
    
    if not datasets:
        print("Error: No datasets found to validate")
        return
    
    # Validate standardization
    results = validate_standardization(datasets)
    
    # Create visualizations
    create_visualizations(datasets, results, args.output_dir)
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()