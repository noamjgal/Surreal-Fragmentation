import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import os
import warnings
from tqdm import tqdm

file_path = "pooled/processed/pooled_stai_data_population_renamed.csv"

df = pd.read_csv(file_path)

print(df.columns)

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

dep_vars = ['Anxiety (Z)', 'Depressed Mood (Z)']
main_ind_var = 'Digital Fragmentation'
control_vars = ['Age Group', 'Gender', 'Digital Duration']


'''notes about participant ids, there is overlap between participants in the two subsidiary datasets, we therefore need to treat participant ids from the tlv dataset as a separate individual from participant ids in the surreal dataset'''

def run_multilevel_models(data, dependent_vars, main_indep_var, control_vars, output_file="multilevel_results.csv", 
                      interaction_terms=None):
    """
    Run multilevel regression models for each dependent variable and output comprehensive results to CSV.
    
    Args:
        data (pd.DataFrame): The dataset with all relevant variables
        dependent_vars (list): List of dependent variables to model
        main_indep_var (str): Main independent variable of interest
        control_vars (list): List of control variables to include in the model
        output_file (str): Output CSV filename
        interaction_terms (list, optional): List of variables to interact with the main independent variable.
                                           For example, ['gender_standardized'] would add 
                                           gender_standardized:main_indep_var interaction.
        
    Returns:
        pd.DataFrame: DataFrame with model results
    
    Notes:
        - For categorical predictors (demographic variables), statsmodels automatically creates dummy variables
          where one level serves as the reference category.
        - Time-varying predictors are separated into between-person and within-person components.
        - Interaction terms are handled by creating interaction variables in the formula.
    """
    # Set default for interaction_terms if None
    if interaction_terms is None:
        interaction_terms = []
    
    # Create a unique participant identifier that differentiates between datasets
    data['unique_participant_id'] = data['Dataset Source'] + '_' + data['Participant ID'].astype(str)
    
    # Check that we have the expected number of unique participants
    expected_count = len(data.groupby(['Dataset Source', 'Participant ID']).size())
    actual_count = len(data['unique_participant_id'].unique())
    
    if expected_count != actual_count:
        warnings.warn(f"Expected {expected_count} unique participants but found {actual_count}. Check participant ID creation.")
    
    # Track model convergence
    convergence_issues = []
    
    # Store results for all dependent variables
    all_model_results = {}
    
    # For between and within person effects, we need to compute means and deviations
    # Group by unique participant ID to calculate person-means
    # First drop missing values in the main independent variable
    valid_data = data.dropna(subset=[main_indep_var])
    
    # Check if we have enough data after removing missing values for the main independent variable
    if len(valid_data) < 10:
        warnings.warn(f"Not enough data for {main_indep_var} after removing missing values. Cannot proceed.")
        return None
    
    print(f"Removed {len(data) - len(valid_data)} rows with missing values in {main_indep_var}")
    print(f"Remaining data: {len(valid_data)} rows")
    
    # Use only valid data for the rest of the analysis
    data = valid_data
    
    person_means = data.groupby('unique_participant_id')[main_indep_var].mean().reset_index()
    person_means.columns = ['unique_participant_id', f'{main_indep_var}_mean']
    
    # Merge person means back to original data
    data = pd.merge(data, person_means, on='unique_participant_id', how='left')
    
    # Calculate person-mean-centered (within-person) values
    data[f'{main_indep_var}_centered'] = data[main_indep_var] - data[f'{main_indep_var}_mean']
    
    # Also person-center the control variables that vary within person
    time_varying_controls = [var for var in control_vars 
                            if var not in ['Gender', 'Age Group', 'Location Type']]
    
    for control in time_varying_controls:
        # Check if this control variable has any missing values
        missing_control = data[control].isna().sum()
        if missing_control > 0:
            print(f"Warning: {missing_control} missing values in control variable {control}")
            # For control variables, we'll impute missing values with the mean
            control_mean = data[control].mean()
            data[control] = data[control].fillna(control_mean)
            print(f"  Imputed with mean value: {control_mean:.3f}")
        
        person_means = data.groupby('unique_participant_id')[control].mean().reset_index()
        person_means.columns = ['unique_participant_id', f'{control}_mean']
        data = pd.merge(data, person_means, on='unique_participant_id', how='left')
        data[f'{control}_centered'] = data[control] - data[f'{control}_mean']
    
    # Process each dependent variable
    for dep_var in tqdm(dependent_vars, desc="Processing dependent variables"):
        print(f"\nModeling dependent variable: {dep_var}")
        
        # Store models for later access
        model_fits = {}
        
        # Drop rows with missing values in the dependent variable
        model_data = data.dropna(subset=[dep_var])
        
        # Ensure column names are patsy-compatible (no spaces or special characters)
        model_data = model_data.copy()
        model_data.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in model_data.columns]
        dep_var_clean = dep_var.replace(' ', '_').replace('(', '').replace(')', '')
        main_indep_var_clean = main_indep_var.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Also clean control variable names
        control_vars_clean = [var.replace(' ', '_').replace('(', '').replace(')', '') for var in control_vars]
        time_varying_controls_clean = [var.replace(' ', '_').replace('(', '').replace(')', '') for var in time_varying_controls]
        
        # Check if we have enough data after dropping missing values
        if len(model_data) < 10:
            warnings.warn(f"Not enough data for {dep_var} after removing missing values. Skipping.")
            continue
            
        # Calculate baseline ICC to understand the nested structure
        try:
            null_formula = f"{dep_var_clean} ~ 1"
            md_null = smf.mixedlm(null_formula, model_data, groups=model_data["unique_participant_id"])
            mdf_null = md_null.fit(reml=True)
            
            # Extract variance components
            random_effect_var = mdf_null.cov_re.iloc[0, 0]
            residual_var = mdf_null.scale
            icc = random_effect_var / (random_effect_var + residual_var)
            
            print(f"ICC for {dep_var}: {icc:.3f}")
        except Exception as e:
            print(f"Error calculating ICC for {dep_var}: {str(e)}")
            icc = np.nan
        
        # --------- Model: With Controls ---------
        # Add control variables to the formula
        control_between = " + ".join([var for var in control_vars_clean if var not in time_varying_controls_clean])
        
        # For time-varying controls, add both their between and within components
        if time_varying_controls_clean:
            control_between += " + " + " + ".join([f"{var}_mean" for var in time_varying_controls_clean])
            control_within = " + " + " + ".join([f"{var}_centered" for var in time_varying_controls_clean])
        else:
            control_within = ""
        
        # Start with basic formula
        full_formula = f"{dep_var_clean} ~ {main_indep_var_clean}_mean + {main_indep_var_clean}_centered + {control_between}{control_within}"
        
        # Add interaction terms if specified
        interaction_parts = []
        for interaction_var in interaction_terms:
            # For categorical variables, we interact with both between and within components of main variable
            if interaction_var in ['Gender', 'Age Group', 'Location Type']:
                # Create interaction with between-person effect
                interaction_parts.append(f"{interaction_var}:{main_indep_var_clean}_mean")
                # Create interaction with within-person effect
                interaction_parts.append(f"{interaction_var}:{main_indep_var_clean}_centered")
                
                # Log the interaction terms
                print(f"Adding interaction terms: {interaction_var}:{main_indep_var_clean}_mean and {interaction_var}:{main_indep_var_clean}_centered")
            else:
                # For continuous variables, different approach needed
                warnings.warn(f"Interaction with continuous variable {interaction_var} not currently supported")
        
        # Add interaction terms to formula if any exist
        if interaction_parts:
            interaction_formula = " + " + " + ".join(interaction_parts)
            full_formula += interaction_formula
        
        try:
            # Fit the model with controls
            print(f"Fitting model with formula: {full_formula}")
            print(f"Model data shape: {model_data.shape}")
            
            # Log variable counts for debugging
            for var in [main_indep_var_clean, f"{main_indep_var_clean}_mean", f"{main_indep_var_clean}_centered"] + control_vars_clean:
                if var in model_data.columns:
                    non_missing = model_data[var].notna().sum()
                    print(f"  Variable {var}: {non_missing}/{len(model_data)} non-missing values")
            
            # Make sure we have all required columns
            required_vars = [f"{main_indep_var_clean}_mean", f"{main_indep_var_clean}_centered"]
            for var in control_vars_clean:
                if var in time_varying_controls_clean:
                    required_vars.extend([f"{var}_mean", f"{var}_centered"])
                else:
                    required_vars.append(var)
            
            missing_vars = [var for var in required_vars if var not in model_data.columns]
            if missing_vars:
                raise ValueError(f"Missing required variables in model data: {missing_vars}")
            
            md = smf.mixedlm(full_formula, model_data, groups=model_data["unique_participant_id"])
            mdf = md.fit(reml=True)
            model_fits[dep_var] = mdf
            
            # Calculate pseudo-R²
            # This gets more complex with many predictors - use a simplified approach
            pred_vars = ["Intercept", f"{main_indep_var_clean}_mean", f"{main_indep_var_clean}_centered"]
            for var in control_vars_clean:
                if var not in time_varying_controls_clean:
                    if var in model_data.columns:
                        pred_vars.append(var)
            
            for var in time_varying_controls_clean:
                pred_vars.append(f"{var}_mean")
                pred_vars.append(f"{var}_centered")
            
            # Keep only variables that actually exist in the model params
            pred_vars = [v for v in pred_vars if v in mdf.params.index]
            
            # Now calculate R²
            var_fixed = np.var(np.dot(model_data[pred_vars], mdf.params[pred_vars]))
            var_random = mdf.cov_re.iloc[0, 0]
            var_residual = mdf.scale
            marginal_r2 = var_fixed / (var_fixed + var_random + var_residual)
            conditional_r2 = (var_fixed + var_random) / (var_fixed + var_random + var_residual)
            
            # Calculate AIC and BIC manually
            k = len(mdf.params)  # Number of parameters
            n = len(model_data)  # Sample size
            llf = mdf.llf  # Log-likelihood
            aic = -2 * llf + 2 * k
            bic = -2 * llf + k * np.log(n)
            
            # Extract all coefficients, standard errors, p-values, etc.
            model_results = {
                'n_observations': len(model_data),
                'n_participants': model_data['unique_participant_id'].nunique(),
                'icc': icc,
                'log_likelihood': llf,
                'AIC': aic,
                'BIC': bic,
                'marginal_r2': marginal_r2,
                'conditional_r2': conditional_r2
            }
            
            # Add main predictor (between and within) statistics
            for param in mdf.params.index:
                if param in [f"{main_indep_var_clean}_mean", f"{main_indep_var_clean}_centered"]:
                    effect_type = "between" if param.endswith("_mean") else "within"
                    param_name = f"{effect_type}_{main_indep_var_clean}"
                    
                    model_results[f"{param_name}_coef"] = mdf.params[param]
                    model_results[f"{param_name}_se"] = mdf.bse[param]
                    # Calculate t-statistic
                    t_value = mdf.params[param] / mdf.bse[param]
                    model_results[f"{param_name}_t"] = t_value
                    # Calculate p-value
                    p_value = 2 * (1 - stats.t.cdf(abs(t_value), mdf.df_resid))
                    model_results[f"{param_name}_p"] = p_value
                    # Add stars/symbols for significance levels
                    if p_value < 0.001:
                        model_results[f"{param_name}_sig"] = "***"
                    elif p_value < 0.01:
                        model_results[f"{param_name}_sig"] = "**"
                    elif p_value < 0.05:
                        model_results[f"{param_name}_sig"] = "*"
                    elif p_value < 0.1:
                        model_results[f"{param_name}_sig"] = "†"  # dagger symbol for marginal significance
                    else:
                        model_results[f"{param_name}_sig"] = ""
                    
                    model_results[f"{param_name}_ci_lower"] = mdf.conf_int().loc[param, 0]
                    model_results[f"{param_name}_ci_upper"] = mdf.conf_int().loc[param, 1]
                
                # Handle interaction terms specifically
                elif ":" in param:
                    # This is an interaction term - parse it
                    parts = param.split(":")
                    if len(parts) == 2:
                        # Handle special case for categorical variables with levels
                        if "[" in parts[0] and "]" in parts[0]:
                            # Something like 'Gender[T.male]:mobility_fragmentation_mean'
                            var_level_part = parts[0]
                            var_parts = var_level_part.split('[')
                            var_name = var_parts[0]
                            level = var_parts[1].replace('T.', '').replace(']', '')
                            var_level = f"{var_name}_{level}"
                            
                            # Check which part of main_indep_var this interacts with
                            if f"{main_indep_var_clean}_mean" in parts[1]:
                                effect_type = "between"
                                param_key = f"interaction_{var_level}_{effect_type}_{main_indep_var_clean}"
                            elif f"{main_indep_var_clean}_centered" in parts[1]:
                                effect_type = "within"
                                param_key = f"interaction_{var_level}_{effect_type}_{main_indep_var_clean}"
                            else:
                                # Some other type of interaction
                                param_key = f"interaction_{param.replace(':', '_').replace('[', '_').replace(']', '_').replace('.', '')}"
                        else:
                            # Regular interaction without categorical levels
                            if f"{main_indep_var_clean}_mean" in parts[1]:
                                effect_type = "between" 
                                param_key = f"interaction_{parts[0]}_{effect_type}_{main_indep_var_clean}"
                            elif f"{main_indep_var_clean}_centered" in parts[1]:
                                effect_type = "within"
                                param_key = f"interaction_{parts[0]}_{effect_type}_{main_indep_var_clean}"
                            else:
                                # Some other type of interaction
                                param_key = f"interaction_{param.replace(':', '_')}"
                        
                        # Now store the interaction parameter information
                        model_results[f"{param_key}_coef"] = mdf.params[param]
                        model_results[f"{param_key}_se"] = mdf.bse[param]
                        t_value = mdf.params[param] / mdf.bse[param]
                        model_results[f"{param_key}_t"] = t_value
                        p_value = 2 * (1 - stats.t.cdf(abs(t_value), mdf.df_resid))
                        model_results[f"{param_key}_p"] = p_value
                        
                        # Add significance indicator
                        if p_value < 0.001:
                            model_results[f"{param_key}_sig"] = "***"
                        elif p_value < 0.01:
                            model_results[f"{param_key}_sig"] = "**"
                        elif p_value < 0.05:
                            model_results[f"{param_key}_sig"] = "*"
                        elif p_value < 0.1:
                            model_results[f"{param_key}_sig"] = "†"
                        else:
                            model_results[f"{param_key}_sig"] = ""
                            
                        model_results[f"{param_key}_ci_lower"] = mdf.conf_int().loc[param, 0]
                        model_results[f"{param_key}_ci_upper"] = mdf.conf_int().loc[param, 1]
                
                # Add control variable statistics
                elif param != "Intercept":
                    # Handle control variables - both time-invariant and time-varying (between/within)
                    is_between = param.endswith("_mean")
                    is_within = param.endswith("_centered")
                    
                    if is_between or is_within:
                        # For time-varying controls, we have both between and within effects
                        effect_type = "between" if is_between else "within"
                        # Extract base variable name from param name
                        if is_between:
                            base_var = param.replace("_mean", "")
                        else:
                            base_var = param.replace("_centered", "")
                        
                        param_name = f"{effect_type}_{base_var}"
                    else:
                        # For time-invariant controls, there's only one effect
                        # Check if this is a categorical variable with multiple levels
                        if '[' in param and ']' in param:
                            # For categorical variables with format like "Gender[T.male]"
                            var_parts = param.split('[')
                            base_var = var_parts[0]
                            level = var_parts[1].replace('T.', '').replace(']', '')
                            param_name = f"{base_var}_{level}"
                        else:
                            param_name = param
                    
                    model_results[f"{param_name}_coef"] = mdf.params[param]
                    model_results[f"{param_name}_se"] = mdf.bse[param]
                    # Calculate t-statistic
                    t_value = mdf.params[param] / mdf.bse[param]
                    model_results[f"{param_name}_t"] = t_value
                    # Calculate p-value
                    p_value = 2 * (1 - stats.t.cdf(abs(t_value), mdf.df_resid))
                    model_results[f"{param_name}_p"] = p_value
                    # Add stars/symbols for significance levels
                    if p_value < 0.001:
                        model_results[f"{param_name}_sig"] = "***"
                    elif p_value < 0.01:
                        model_results[f"{param_name}_sig"] = "**"
                    elif p_value < 0.05:
                        model_results[f"{param_name}_sig"] = "*"
                    elif p_value < 0.1:
                        model_results[f"{param_name}_sig"] = "†"  # dagger symbol for marginal significance
                    else:
                        model_results[f"{param_name}_sig"] = ""
                    
                    model_results[f"{param_name}_ci_lower"] = mdf.conf_int().loc[param, 0]
                    model_results[f"{param_name}_ci_upper"] = mdf.conf_int().loc[param, 1]
                else:
                    # Add intercept
                    model_results["Intercept_coef"] = mdf.params[param]
                    model_results["Intercept_se"] = mdf.bse[param]
                    t_value = mdf.params[param] / mdf.bse[param]
                    model_results["Intercept_t"] = t_value
                    p_value = 2 * (1 - stats.t.cdf(abs(t_value), mdf.df_resid))
                    model_results["Intercept_p"] = p_value
                    if p_value < 0.001:
                        model_results["Intercept_sig"] = "***"
                    elif p_value < 0.01:
                        model_results["Intercept_sig"] = "**"
                    elif p_value < 0.05:
                        model_results["Intercept_sig"] = "*"
                    elif p_value < 0.1:
                        model_results["Intercept_sig"] = "†"  # dagger symbol for marginal significance
                    else:
                        model_results["Intercept_sig"] = ""
                    model_results["Intercept_ci_lower"] = mdf.conf_int().loc[param, 0]
                    model_results["Intercept_ci_upper"] = mdf.conf_int().loc[param, 1]
            
            # Store results for this dependent variable
            all_model_results[dep_var] = model_results
            
        except Exception as e:
            print(f"Error fitting model for {dep_var}: {str(e)}")
            convergence_issues.append(f"{dep_var}: {str(e)}")
    
    # Create article-ready table with one column per dependent variable
    if not all_model_results:
        print("No successful model fits to report")
        return None
    
    # Find all parameter names across all models to ensure we include categorical variable levels
    all_params = set()
    for dep_var, results in all_model_results.items():
        all_params.update([k for k in results.keys() if k.endswith('_coef')])
    
    print(f"All parameter names found: {all_params}")
    
    # Debug interaction parameters
    if interaction_terms:
        print("\nLooking for interaction parameters:")
        for interaction_var in interaction_terms:
            # More flexible pattern matching for interaction parameters
            interaction_patterns = [
                f"{interaction_var}:", 
                f"interaction_{interaction_var}", 
                f"{interaction_var}_"
            ]
            found_params = []
            for pattern in interaction_patterns:
                found_params.extend([p for p in all_params if pattern in p])
            found_params = list(set(found_params))  # Remove duplicates
            print(f"Interaction variable {interaction_var}: found {len(found_params)} parameters: {found_params}")
            
            # Also check the actual model parameters
            for dep_var, model_fit in model_fits.items():
                print(f"Model parameters for {dep_var}:")
                interaction_param_found = False
                for param_name in model_fit.params.index:
                    # Use more flexible matching for parameters
                    if interaction_var in param_name or f"{interaction_var.lower()}" in param_name.lower():
                        interaction_param_found = True
                        print(f"  {param_name}: {model_fit.params[param_name]:.4f}, p-value: {2 * (1 - stats.t.cdf(abs(model_fit.params[param_name] / model_fit.bse[param_name]), model_fit.df_resid)):.4f}")
                        
                        # Make sure this parameter is in the results
                        param_base = param_name.replace("[", "_").replace("]", "").replace(".", "").replace("T", "")
                        if f"{param_base}_coef" not in all_model_results[dep_var]:
                            print(f"  WARNING: Parameter {param_name} not found in results as {param_base}_coef")
                
                if not interaction_param_found:
                    print(f"  No interaction parameters found for {dep_var}")
    
    # Define the order of rows in the table
    row_order = [
        'n_observations', 'n_participants', 'icc', 
        'Intercept_coef', 'Intercept_se', 'Intercept_p', 
        f'between_{main_indep_var_clean}_coef', f'between_{main_indep_var_clean}_se', f'between_{main_indep_var_clean}_p', 
        f'within_{main_indep_var_clean}_coef', f'within_{main_indep_var_clean}_se', f'within_{main_indep_var_clean}_p'
    ]
    
    # Add interaction terms to row order if any
    for interaction_var in interaction_terms:
        # Look for interaction parameters in the model results
        interaction_params = []
        for param in all_params:
            if param.startswith('interaction_') and param.endswith('_coef'):
                if interaction_var in param:
                    interaction_params.append(param)
        
        print(f"Found interaction parameters for {interaction_var}: {interaction_params}")
        
        for param in sorted(interaction_params):
            base_name = param[:-5]  # Remove _coef suffix
            row_order.extend([
                f'{base_name}_coef', f'{base_name}_se', f'{base_name}_p'
            ])
    
    # Add control variables to row order
    for control in control_vars:
        control_clean = control.replace(' ', '_').replace('(', '').replace(')', '')
        if control in time_varying_controls:
            # Time-varying controls have both between and within effects
            row_order.extend([
                f'between_{control_clean}_coef', f'between_{control_clean}_se', f'between_{control_clean}_p',
                f'within_{control_clean}_coef', f'within_{control_clean}_se', f'within_{control_clean}_p'
            ])
        else:
            # Check if this is a categorical variable with multiple levels
            categorical_params = [p for p in all_params if p.startswith(f"{control_clean}_") and p.endswith("_coef") and ":" not in p]
            if categorical_params:
                # Add each level of the categorical variable
                for param in categorical_params:
                    base_name = param[:-5]  # Remove _coef suffix
                    row_order.extend([
                        f'{base_name}_coef', f'{base_name}_se', f'{base_name}_p'
                    ])
            else:
                # Regular non-categorical variable
                row_order.extend([
                    f'{control_clean}_coef', f'{control_clean}_se', f'{control_clean}_p'
                ])
    
    # Add model fit metrics to row order
    row_order.extend(['log_likelihood', 'AIC', 'BIC', 'marginal_r2', 'conditional_r2'])
    
    # Create a list for each row
    rows = []
    
    # First, add descriptive metric names
    metric_names = [
        'Observations (N)', 'Participants', 'ICC', 
        'Intercept', 'SE', 'p-value',
        f'Between {main_indep_var}', 'SE', 'p-value',
        f'Within {main_indep_var}', 'SE', 'p-value'
    ]
    
    # Add interaction term names
    for interaction_var in interaction_terms:
        # Look for interaction parameters in the model results
        interaction_params = []
        for param in all_params:
            if param.startswith('interaction_') and param.endswith('_coef'):
                if interaction_var in param:
                    interaction_params.append(param)
        
        for param in sorted(interaction_params):
            base_name = param[:-5]  # Remove _coef suffix
            
            # Format a nice display name for the interaction
            if 'between' in base_name:
                effect_type = 'Between'
            elif 'within' in base_name:
                effect_type = 'Within'
            else:
                effect_type = ''
                
            # Check if it's a gender interaction
            if 'Gender_male' in base_name:
                var_display = 'Gender [male]'
            elif 'Age_Group_adult' in base_name:
                var_display = 'Age Group [adult]'
            elif 'Location_Type' in base_name:
                var_display = 'Location Type [suburb]'
            else:
                var_display = interaction_var
                
            display_name = f"{effect_type} {var_display} × {main_indep_var}"
            
            metric_names.extend([
                display_name, 'SE', 'p-value'
            ])
    
    # Add control variable names
    for control in control_vars:
        control_clean = control.replace(' ', '_').replace('(', '').replace(')', '')
        if control in time_varying_controls:
            # Time-varying controls have both between and within effects
            metric_names.extend([
                f'Between {control}', 'SE', 'p-value',
                f'Within {control}', 'SE', 'p-value'
            ])
        else:
            # Check if this is a categorical variable with multiple levels
            categorical_params = [p for p in all_params if p.startswith(f"{control_clean}_") and p.endswith("_coef") and ":" not in p]
            if categorical_params:
                # Add each level of the categorical variable
                for param in categorical_params:
                    base_name = param[:-5]  # Remove _coef suffix
                    level = base_name.split('_')[-1]  # Get the level name
                    metric_names.extend([
                        f'{control} [{level}]', 'SE', 'p-value'
                    ])
            else:
                # Regular non-categorical variable
                metric_names.extend([
                    f'{control}', 'SE', 'p-value'
                ])
    
    # Add model fit metric names
    metric_names.extend(['Log-likelihood', 'AIC', 'BIC', 'Marginal R²', 'Conditional R²'])
    
    # Add metric names as the first column
    rows.append(['Metric'] + dependent_vars)
    
    # Add each row in specified order
    for i, row_id in enumerate(row_order):
        row_values = [metric_names[i]]
        
        for dep_var in dependent_vars:
            if dep_var in all_model_results and row_id in all_model_results[dep_var]:
                value = all_model_results[dep_var][row_id]
                
                # Format numbers appropriately
                if '_coef' in row_id or '_se' in row_id or '_ci_' in row_id:
                    row_values.append(f"{value:.3f}")
                elif '_p' in row_id:
                    if value < 0.001:
                        row_values.append("<0.001")
                    else:
                        row_values.append(f"{value:.3f}")
                elif 'r2' in row_id:
                    row_values.append(f"{value:.3f}")
                elif row_id == 'icc':
                    row_values.append(f"{value:.3f}")
                else:
                    row_values.append(str(value))
            else:
                row_values.append('')
        
        rows.append(row_values)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(rows)
    
    # Use the header row (row with Metric and dependent variables)
    header = results_df.iloc[0]
    
    # Remove the header row from the data
    results_df = results_df[1:]
    
    # Set column names from the header row
    results_df.columns = header
    
    # Save to CSV without the significance legend
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    
    # Report any convergence issues
    if convergence_issues:
        print("\nConvergence issues encountered:")
        for issue in convergence_issues:
            print(f"- {issue}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Add Intercept column required by statsmodels
    df["Intercept"] = 1
    
    # Define interaction term - set this to None to run without interactions
    # Example: interaction with gender
    interaction_terms = None #['gender_standardized']
    
    print(f"Running multilevel models for {len(dep_vars)} dependent variables")
    print(f"Main independent variable: {main_ind_var}")
    print(f"Control variables: {control_vars}")
    if interaction_terms:
        print(f"Interaction terms: {main_ind_var} × {', '.join(interaction_terms)}")
    
    # Run the analyses
    results = run_multilevel_models(
        data=df,
        dependent_vars=dep_vars,
        main_indep_var=main_ind_var,
        control_vars=control_vars,
        interaction_terms=interaction_terms,
        output_file="pooled/results/multilevel_fragmentation_results_with_interactions.csv"
    )
    
    print("\nAnalysis complete!")
