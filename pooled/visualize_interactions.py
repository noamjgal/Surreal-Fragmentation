import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multi_focus import run_multilevel_models
import statsmodels.api as sm
import os

# Create directory for visualizations if it doesn't exist
os.makedirs('pooled/visualizations', exist_ok=True)

# Load the dataset
file_path = "pooled/processed/pooled_stai_data_population_renamed.csv"
df = pd.read_csv(file_path)
'''
columns:
Index(['Participant ID', 'Dataset Source', 'Anxiety (Z)', 'Anxiety (Raw)',
       'Depressed Mood (Z)', 'Depressed Mood (Raw)', 'Gender', 'Location Type',
       'Age Group', 'Digital Fragmentation', 'Mobility Fragmentation',
       'Digital Mobile Fragmentation', 'Digital Home Fragmentation',
       'Digital Home Mobility Delta', 'Digital Duration', 'Mobile Duration',
       'Digital Mobile Duration', 'Digital Home Duration',
       'Active Transport Duration', 'Mechanized Transport Duration',
       'Home Duration', 'Out of Home Duration'],
      dtype='object')
'''

# Define variables (same as in multi_focus.py)
dep_vars = ['Anxiety (Z)', 'Depressed Mood (Z)']
main_ind_var = 'Digital Mobile Fragmentation'
control_vars = ['Age Group', 'Mobile Duration']

# Add Intercept column required by statsmodels
df["Intercept"] = 1

# Set interaction term to gender
interaction_terms = ['Gender']

print(f"Running multilevel models with interaction terms: {interaction_terms}")

# Run the analysis
results = run_multilevel_models(
    data=df,
    dependent_vars=dep_vars,
    main_indep_var=main_ind_var,
    control_vars=control_vars,
    interaction_terms=interaction_terms,
    output_file="pooled/results/multilevel_fragmentation_results_with_gender_interaction.csv"
)

print("Analysis complete! Now generating visualizations...")

# Create a unique participant identifier
df['unique_participant_id'] = df['Dataset Source'] + '_' + df['Participant ID'].astype(str)

# Calculate person-means for the main independent variable
person_means = df.groupby('unique_participant_id')[main_ind_var].mean().reset_index()
person_means.columns = ['unique_participant_id', f'{main_ind_var}_mean']
df = pd.merge(df, person_means, on='unique_participant_id', how='left')

# Calculate person-mean-centered values, handling NaN values
df[f'{main_ind_var}_centered'] = np.where(
    df[main_ind_var].notna(),
    df[main_ind_var] - df[f'{main_ind_var}_mean'],
    np.nan
)

# Do the same for time-varying control variables
time_varying_controls = ['Mobile Duration']
for control in time_varying_controls:
    person_means = df.groupby('unique_participant_id')[control].mean().reset_index()
    person_means.columns = ['unique_participant_id', f'{control}_mean']
    df = pd.merge(df, person_means, on='unique_participant_id', how='left')
    df[f'{control}_centered'] = np.where(
        df[control].notna(),
        df[control] - df[f'{control}_mean'],
        np.nan
    )

# Function to create interaction plots
def create_interaction_plots(data, dependent_var, main_indep_var, interaction_var):
    """
    Create plots showing the interaction between main_indep_var and interaction_var 
    for the dependent_var, using confidence intervals instead of individual points.
    """
    # Remove missing values
    plot_data = data.dropna(subset=[dependent_var, f'{main_indep_var}_mean', interaction_var])
    
    # Get unique values of the interaction variable
    interaction_values = plot_data[interaction_var].unique()
    
    # Create between-person effect plot
    plt.figure(figsize=(10, 6))
    
    # Plot regression line with confidence interval for each level of the interaction variable
    for value in interaction_values:
        subset = plot_data[plot_data[interaction_var] == value]
        
        # Fit regression model
        X = sm.add_constant(subset[f'{main_indep_var}_mean'])
        model = sm.OLS(subset[dependent_var], X).fit()
        
        # Create a range of x values for the regression line
        x_range = np.linspace(subset[f'{main_indep_var}_mean'].min(), 
                             subset[f'{main_indep_var}_mean'].max(), 100)
        X_pred = sm.add_constant(x_range)
        
        # Get predictions and confidence intervals
        y_pred = model.get_prediction(X_pred)
        y_mean = y_pred.predicted_mean
        y_ci = y_pred.conf_int(alpha=0.05)  # 95% confidence interval
        
        # Plot regression line with confidence interval
        plt.plot(x_range, y_mean, linewidth=2, 
                label=f'{interaction_var}={value} (slope={model.params.iloc[1]:.3f})')
        plt.fill_between(x_range, y_ci[:, 0], y_ci[:, 1], alpha=0.2)
    
    plt.xlabel(f'{main_indep_var.replace("_", " ").title()} (Between-Person)')
    plt.ylabel(f'{dependent_var.replace("_", " ").title()}')
    plt.title(f'Interaction Effect: {interaction_var.replace("_", " ").title()} × '
             f'{main_indep_var.replace("_", " ").title()} (Between-Person)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'pooled/visualizations/between_interaction_{dependent_var}_{main_indep_var}_{interaction_var}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create within-person effect plot (if there's enough within-person variation)
    plt.figure(figsize=(10, 6))
    
    # Check if we have enough participants with multiple observations
    participant_counts = plot_data.groupby('unique_participant_id').size()
    participants_with_multiple = participant_counts[participant_counts > 1].index
    
    if len(participants_with_multiple) >= 5:
        # Plot regression lines with confidence intervals for within-person effects
        for value in interaction_values:
            subset = plot_data[plot_data[interaction_var] == value]
            # Make sure centered values don't have NaN
            subset = subset.dropna(subset=[f'{main_indep_var}_centered', dependent_var])
            
            # Only proceed if we have enough data
            if len(subset) >= 10:
                # Create overall within-person effect line with confidence interval
                X = sm.add_constant(subset[f'{main_indep_var}_centered'])
                model = sm.OLS(subset[dependent_var], X).fit()
                
                # Create a range of x values for the regression line
                x_range = np.linspace(subset[f'{main_indep_var}_centered'].min(), 
                                     subset[f'{main_indep_var}_centered'].max(), 100)
                X_pred = sm.add_constant(x_range)
                
                # Get predictions and confidence intervals
                y_pred = model.get_prediction(X_pred)
                y_mean = y_pred.predicted_mean
                y_ci = y_pred.conf_int(alpha=0.05)  # 95% confidence interval
                
                # Plot regression line with confidence interval
                plt.plot(x_range, y_mean, linewidth=3, 
                        label=f'{interaction_var}={value} (slope={model.params.iloc[1]:.3f})')
                plt.fill_between(x_range, y_ci[:, 0], y_ci[:, 1], alpha=0.2)
        
        plt.xlabel(f'{main_indep_var.replace("_", " ").title()} (Within-Person Deviation)')
        plt.ylabel(f'{dependent_var.replace("_", " ").title()}')
        plt.title(f'Within-Person Effect: {interaction_var.replace("_", " ").title()} × '
                 f'{main_indep_var.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'pooled/visualizations/within_interaction_{dependent_var}_{main_indep_var}_{interaction_var}.png', 
                   dpi=300, bbox_inches='tight')
    else:
        print(f"Not enough within-person variation for {dependent_var} with {interaction_var}")
    
    plt.close()

# Function to create marginal effects plots using model predictions
def create_marginal_effects_plot(results_file, dependent_vars, main_indep_var, interaction_var):
    """
    Create marginal effects plots based on the model results.
    """
    # Read the results file
    results = pd.read_csv(results_file)
    
    # Process each dependent variable
    for dep_var in dependent_vars:
        print(f"\nCreating marginal effects plot for {dep_var}")
        
        # Extract the model parameters
        # First, find the rows with main effect and interaction effect
        rows = results['Metric'].tolist()
        
        # Find the main effect (for reference group - female)
        main_effect_row = -1
        for i, row in enumerate(rows):
            if row == f'Between {main_indep_var}':
                main_effect_row = i
                break
        
        if main_effect_row == -1:
            print(f"Cannot find main effect row for {main_indep_var}")
            continue
        
        # Find the interaction effect (male)
        interaction_row = -1
        for i, row in enumerate(rows):
            if row == f'Between {interaction_var} [male] × {main_indep_var}':
                interaction_row = i
                break
        
        if interaction_row == -1:
            print(f"Cannot find interaction row for {interaction_var} × {main_indep_var}")
            continue
        
        # Extract the coefficients and standard errors
        main_effect = float(results.iloc[main_effect_row][dep_var])
        main_effect_se = float(results.iloc[main_effect_row + 1][dep_var])
        
        interaction_effect = float(results.iloc[interaction_row][dep_var])
        interaction_effect_se = float(results.iloc[interaction_row + 1][dep_var])
        
        print(f"Main effect (female): {main_effect:.3f}, SE: {main_effect_se:.3f}")
        print(f"Interaction effect (male): {interaction_effect:.3f}, SE: {interaction_effect_se:.3f}")
        
        # Calculate marginal effects for each gender
        # For female (reference group): just the main effect
        female_effect = main_effect
        female_se = main_effect_se
        
        # For male: main effect + interaction effect
        # We need to correctly calculate the standard error for the sum
        # In the absence of the covariance matrix, we'll use an approximation
        # that assumes independence between parameters
        male_effect = main_effect + interaction_effect
        
        # Apply variance sum formula with an estimated correlation
        # We'll conservatively use a zero correlation here
        estimated_correlation = 0
        male_se = np.sqrt(main_effect_se**2 + interaction_effect_se**2 + 
                         2 * estimated_correlation * main_effect_se * interaction_effect_se)
        
        print(f"Marginal effect for female: {female_effect:.3f}, SE: {female_se:.3f}")
        print(f"Marginal effect for male: {male_effect:.3f}, SE: {male_se:.3f}")
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Set up the plot data
        categories = ['female', 'male']
        effects = [female_effect, male_effect]
        errors = [1.96 * female_se, 1.96 * male_se]  # 95% CI
        
        # Create bar plot
        plt.bar(categories, effects, yerr=errors, align='center', alpha=0.7, 
               ecolor='black', capsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel(interaction_var)
        plt.ylabel(f'Effect of {main_indep_var} on {dep_var}')
        plt.title(f'Marginal Effects of {main_indep_var} by {interaction_var}')
        
        # Add effect size labels with confidence intervals
        for i, v in enumerate(effects):
            plt.text(i, v + (0.05 if v >= 0 else -0.15), 
                    f"{v:.3f}\n95% CI: [{v-errors[i]:.3f}, {v+errors[i]:.3f}]", 
                    ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'pooled/visualizations/marginal_effects_{dep_var}_{main_indep_var}_{interaction_var}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved marginal effects plot to pooled/visualizations/marginal_effects_{dep_var}_{main_indep_var}_{interaction_var}.png")

        # Create a predicted values plot as an alternative visualization
        plt.figure(figsize=(10, 6))
        
        # Create data for predictions
        x_range = np.linspace(-2, 2, 100)  # Standardized range for the main effect
        
        # Predict values for female (just main effect)
        female_y = main_effect * x_range
        female_ci_lower = female_y - 1.96 * main_effect_se
        female_ci_upper = female_y + 1.96 * main_effect_se
        
        # Predict values for male (main effect + interaction)
        male_y = (main_effect + interaction_effect) * x_range
        male_ci_lower = male_y - 1.96 * male_se
        male_ci_upper = male_y + 1.96 * male_se
        
        # Plot the lines with confidence intervals
        plt.plot(x_range, female_y, label='Female', linewidth=2, color='#FF9999')
        plt.fill_between(x_range, female_ci_lower, female_ci_upper, alpha=0.2, color='#FF9999')
        
        plt.plot(x_range, male_y, label='Male', linewidth=2, color='#6699CC')
        plt.fill_between(x_range, male_ci_lower, male_ci_upper, alpha=0.2, color='#6699CC')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        plt.xlabel(f'{main_indep_var} (Standardized)')
        plt.ylabel(f'Predicted {dep_var}')
        plt.title(f'Predicted Effect of {main_indep_var} by {interaction_var}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'pooled/visualizations/predicted_values_{dep_var}_{main_indep_var}_{interaction_var}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved predicted values plot to pooled/visualizations/predicted_values_{dep_var}_{main_indep_var}_{interaction_var}.png")

# Generate interaction plots
for dep_var in dep_vars:
    create_interaction_plots(df, dep_var, main_ind_var, 'Gender')

# Generate marginal effects plots
create_marginal_effects_plot(
    "pooled/results/multilevel_fragmentation_results_with_gender_interaction.csv",
    dep_vars,
    main_ind_var,
    'Gender'
)

print("Visualizations created! Check the pooled/visualizations directory.") 