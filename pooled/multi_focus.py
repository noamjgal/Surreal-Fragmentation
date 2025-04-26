import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
import os
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import patsy

# Suppress warnings to keep output clean but keep convergence warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Get the script's directory and project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # One level up from script directory

# Create directories for results and visualizations using absolute paths
os.makedirs(os.path.join(script_dir, 'results'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

def run_multilevel_models(data, dependent_vars, main_indep_var, control_vars, output_file="multilevel_results.csv", 
                      interaction_terms=None, create_visualizations=True):
    """
    Run multilevel regression models for each dependent variable and output comprehensive results to CSV.
    
    Args:
        data (pd.DataFrame): The dataset with all relevant variables
        dependent_vars (list): List of dependent variables to model
        main_indep_var (str): Main independent variable of interest
        control_vars (list): List of control variables to include in the model
        output_file (str): Output CSV filename
        interaction_terms (list, optional): List of variables to interact with the main independent variable.
        create_visualizations (bool): Whether to create interaction visualization plots
    
    Returns:
        pd.DataFrame: DataFrame with model results
    """
    # Set interaction terms to empty list if interaction_terms is None
    if interaction_terms is None:
        interaction_terms = []
    
    # Create a unique participant identifier to differentiate between duplicate participant ids in the two datasets
    data['unique_participant_id'] = data['Dataset Source'] + '_' + data['Participant ID'].astype(str)
    
    # Check that we have the expected number of unique participants
    expected_count = len(data.groupby(['Dataset Source', 'Participant ID']).size())
    actual_count = len(data['unique_participant_id'].unique())
    assert expected_count == actual_count, f"Expected {expected_count} unique participants but found {actual_count}. Check participant ID creation."
    
    # Track model convergence
    convergence_issues = []
    
    # Store results for all dependent variables
    all_model_results = {}
    
    # Store model fits for visualization later
    model_fits = {}
    
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
    
    # Explicitly convert categorical variables to category type
    for var in interaction_terms + [v for v in control_vars if v in ['Gender', 'Age Group', 'Location Type']]:
        if var in data.columns:
            print(f"Converting {var} to categorical type")
            data[var] = data[var].astype('category')
            print(f"Categories for {var}: {data[var].cat.categories.tolist()}")
    
    # Calculate person means for the main independent variable
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
            print(f"Random effect variance: {random_effect_var:.3f}")
            print(f"Residual variance: {residual_var:.3f}")
        except Exception as e:
            print(f"Error calculating ICC for {dep_var}: {str(e)}")
            icc = np.nan
            random_effect_var = np.nan
            residual_var = np.nan
        
        # --------- Model: With Controls ---------
        # Add control variables to the formula
        control_vars_list = [var for var in control_vars_clean if var not in time_varying_controls_clean]
        if control_vars_list:
            control_between = " + " + " + ".join([f"C({var})" if var in ["Gender", "Age_Group", "Location_Type"] else var for var in control_vars_list])
        else:
            control_between = ""
        
        # For time-varying controls, add both their between and within components
        if time_varying_controls_clean:
            if control_between:
                control_between += " + " + " + ".join([f"{var}_mean" for var in time_varying_controls_clean])
            else:
                control_between = " + " + " + ".join([f"{var}_mean" for var in time_varying_controls_clean])
            control_within = " + " + " + ".join([f"{var}_centered" for var in time_varying_controls_clean])
        else:
            control_within = ""
        
        # Start with basic formula
        full_formula = f"{dep_var_clean} ~ {main_indep_var_clean}_mean + {main_indep_var_clean}_centered{control_between}{control_within}"
        
        # Add interaction terms if specified
        interaction_parts = []
        for interaction_var in interaction_terms:
            # Clean the interaction variable name to make it formula-compatible
            interaction_var_clean = interaction_var.replace(' ', '_')
            
            # For categorical variables, use C() to explicitly mark as categorical
            if interaction_var in ['Gender', 'Age Group', 'Location Type']:
                # Create interaction with between-person effect
                interaction_parts.append(f"C({interaction_var_clean}):{main_indep_var_clean}_mean")
                # Create interaction with within-person effect
                interaction_parts.append(f"C({interaction_var_clean}):{main_indep_var_clean}_centered")
                
                # Log the interaction terms
                print(f"Adding interaction terms: C({interaction_var_clean}):{main_indep_var_clean}_mean and C({interaction_var_clean}):{main_indep_var_clean}_centered")
            else:
                # For continuous variables, different approach needed
                warnings.warn(f"Interaction with continuous variable {interaction_var} not currently supported")
        
        # Add interaction terms to formula if any exist
        if interaction_parts:
            interaction_formula = " + " + " + ".join(interaction_parts)
            full_formula += interaction_formula
        
        try:
            # Debug: Check the design matrix before fitting
            print(f"Fitting model with formula: {full_formula}")
            print(f"Model data shape: {model_data.shape}")
            
            # Preview the design matrix to diagnose issues
            try:
                y, X = patsy.dmatrices(full_formula, model_data, return_type='dataframe')
                print("Formula terms in design matrix:")
                print(X.columns.tolist())
                
                # Check specifically for interaction columns
                interaction_cols = [col for col in X.columns if ':' in str(col)]
                print(f"Interaction columns ({len(interaction_cols)}): {interaction_cols}")
            except Exception as e:
                print(f"Error inspecting design matrix: {str(e)}")
            
            # Log variable counts for debugging
            for var in [main_indep_var_clean, f"{main_indep_var_clean}_mean", f"{main_indep_var_clean}_centered"] + control_vars_clean:
                if var in model_data.columns:
                    non_missing = model_data[var].notna().sum()
                    print(f"  Variable {var}: {non_missing}/{len(model_data)} non-missing values")
            
            # Fit the model with controls
            md = smf.mixedlm(full_formula, model_data, groups=model_data["unique_participant_id"])
            mdf = md.fit(reml=True)
            model_fits[dep_var] = mdf
            
            print("\nModel parameters for", dep_var)
            for param in mdf.params.index:
                if ':' in param:  # This is an interaction parameter
                    print(f"  {param}: {mdf.params[param]:.4f}, p-value: {2 * (1 - stats.t.cdf(abs(mdf.params[param] / mdf.bse[param]), mdf.df_resid)):.4f}")
            
            # Calculate pseudo-R²
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
                'random_effect_var': random_effect_var,
                'residual_var': residual_var,
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
                
                # Handle interaction terms using improved extraction
                elif ':' in param:
                    # This is an interaction term - use flexible parsing
                    print(f"Extracting interaction parameter: {param}")
                    
                    # Determine which type of interaction pattern we have
                    if 'C(' in param:
                        # Format like "C(Gender)[T.male]:Digital_Home_Mobility_Delta_mean"
                        parts = param.split(':')
                        
                        # Extract categorical variable and level
                        cat_part = parts[0]
                        var_name = cat_part.split('(')[1].split(')')[0]
                        level = "base"  # Default if no level specified
                        
                        if '[T.' in cat_part:
                            level = cat_part.split('[T.')[1].split(']')[0]
                        
                        # Determine effect type (between/within)
                        indep_part = parts[1]
                        effect_type = "between" if "_mean" in indep_part else "within"
                        
                        # Create parameter key
                        param_key = f"interaction_{var_name}_{level}_{effect_type}_{main_indep_var_clean}"
                    else:
                        # Try other formats or use a generic format
                        param_key = f"interaction_{param.replace(':', '_').replace('[', '_').replace(']', '_').replace('.', '')}"
                    
                    # Store parameter information
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
                    
                    print(f"  Successfully processed interaction: {param} -> {param_key}")
                
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
                        if 'C(' in param and ')[T.' in param:
                            # For categorical variables with format like "C(Gender)[T.male]"
                            var_name = param.split('(')[1].split(')')[0]
                            level = param.split('[T.')[1].split(']')[0]
                            param_name = f"{var_name}_{level}"
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
            # Clean the interaction variable name to match how it was used in the formula
            interaction_var_clean = interaction_var.replace(' ', '_')
            
            # More flexible pattern matching for interaction parameters
            interaction_patterns = [
                f"{interaction_var_clean}", 
                f"interaction_{interaction_var_clean}", 
            ]
            found_params = []
            for pattern in interaction_patterns:
                found_params.extend([p for p in all_params if pattern in p.lower() and 'interaction' in p.lower()])
            found_params = list(set(found_params))  # Remove duplicates
            print(f"Interaction variable {interaction_var}: found {len(found_params)} parameters: {found_params}")
    
    # Define the order of rows in the table
    row_order = [
        'n_observations', 'n_participants', 'icc', 
        'random_effect_var', 'residual_var',
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
                if interaction_var.lower().replace(' ', '_') in param.lower():
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
    
    # Add model fit metric names
    row_order.extend(['log_likelihood', 'AIC', 'BIC', 'marginal_r2', 'conditional_r2'])
    
    # Create a list for each row
    rows = []
    
    # First, add descriptive metric names
    metric_names = [
        'Observations (N)', 'Participants', 'ICC', 
        'Random Effect Variance', 'Residual Variance',
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
                if interaction_var.lower().replace(' ', '_') in param.lower():
                    interaction_params.append(param)
        
        for param in sorted(interaction_params):
            base_name = param[:-5]  # Remove _coef suffix
            
            # Format a nice display name for the interaction
            parts = base_name.split('_')
            if 'between' in parts:
                effect_type = 'Between'
            elif 'within' in parts:
                effect_type = 'Within'
            else:
                effect_type = ''
                
            # Try to extract the level of the categorical variable
            if interaction_var == 'Gender' and 'male' in base_name:
                var_display = 'Gender [male]'
            elif interaction_var == 'Age Group' and 'adult' in base_name:
                var_display = 'Age Group [adult]'
            elif interaction_var == 'Location Type' and 'suburb' in base_name:
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
        if i < len(metric_names):
            row_values = [metric_names[i]]
        else:
            row_values = [f"Parameter {i}"]  # Fallback if metric_names is shorter than row_order
        
        for dep_var in dependent_vars:
            if dep_var in all_model_results and row_id in all_model_results[dep_var]:
                value = all_model_results[dep_var][row_id]
                
                # Format numbers appropriately
                if '_coef' in row_id or '_se' in row_id or '_ci_' in row_id:
                    row_values.append(format_small_number(value))
                elif '_p' in row_id:
                    # Get the corresponding parameter base name to look up significance
                    param_base = row_id.replace('_p', '')
                    sig_key = f"{param_base}_sig"
                    sig_symbol = all_model_results[dep_var].get(sig_key, "")
                    
                    # Format p-value with significance symbol
                    if value < 0.001:
                        row_values.append(f"<0.001{sig_symbol}")
                    else:
                        row_values.append(f"{value:.3f}{sig_symbol}")
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
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    # Create and save a version with significance note
    note = "Note: † p<0.1, * p<0.05, ** p<0.01, *** p<0.001"
    with open(output_file, 'a') as f:
        f.write(f"\n{note}")
    
    print(f"\nResults saved to {output_file}")
    print(f"Significance levels: {note}")
    
    # Report any convergence issues
    if convergence_issues:
        print("\nConvergence issues encountered:")
        for issue in convergence_issues:
            print(f"- {issue}")
    
    # Create visualizations if requested
    if create_visualizations and interaction_terms:
        for interaction_var in interaction_terms:
            print(f"\nCreating visualizations for interaction with {interaction_var}")
            create_interaction_plots(data, dependent_vars, main_indep_var, interaction_var)
            create_marginal_effects_plot(output_file, dependent_vars, main_indep_var, interaction_var)
    
    return results_df

# Function to create interaction plots
def create_interaction_plots(data, dependent_vars, main_indep_var, interaction_var):
    """
    Create plots showing the interaction between main_indep_var and interaction_var
    for each dependent variable.
    """
    # Get the script's directory for output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure column names are clean for patsy
    data = data.copy()
    data.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in data.columns]
    main_indep_var_clean = main_indep_var.replace(' ', '_').replace('(', '').replace(')', '')
    interaction_var_clean = interaction_var.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Process each dependent variable
    for dep_var in dependent_vars:
        dep_var_clean = dep_var.replace(' ', '_').replace('(', '').replace(')', '')
        print(f"\nCreating interaction plots for {dep_var} × {interaction_var}")
        
        # Remove missing values
        plot_data = data.dropna(subset=[dep_var_clean, f'{main_indep_var_clean}_mean', interaction_var_clean])
        
        # Get unique values of the interaction variable
        interaction_values = plot_data[interaction_var_clean].unique()
        print(f"Found {len(interaction_values)} unique values for {interaction_var}: {interaction_values}")
        
        # Create between-person effect plot
        plt.figure(figsize=(10, 6))
        
        # Plot regression line with confidence interval for each level of the interaction variable
        for value in interaction_values:
            subset = plot_data[plot_data[interaction_var_clean] == value]
            
            # Make sure we have enough data
            if len(subset) < 10:
                print(f"  Not enough data for {interaction_var}={value} (n={len(subset)})")
                continue
            
            # Fit regression model
            X = sm.add_constant(subset[f'{main_indep_var_clean}_mean'])
            
            try:
                model = sm.OLS(subset[dep_var_clean], X).fit()
                
                # Create a range of x values for the regression line
                x_min = subset[f'{main_indep_var_clean}_mean'].min()
                x_max = subset[f'{main_indep_var_clean}_mean'].max()
                x_range = np.linspace(x_min, x_max, 100)
                X_pred = sm.add_constant(x_range)
                
                # Get predictions and confidence intervals
                y_pred = model.get_prediction(X_pred)
                y_mean = y_pred.predicted_mean
                y_ci = y_pred.conf_int(alpha=0.05)  # 95% confidence interval
                
                # Format legend label based on interaction variable
                if interaction_var.lower() == 'gender':
                    legend_label = 'Male' if value == 'male' else 'Female'
                else:
                    legend_label = f'{interaction_var}={value}'
                
                # Plot regression line with confidence interval
                plt.plot(x_range, y_mean, linewidth=2, 
                        label=f'{legend_label} (slope={model.params.iloc[1]:.3f})')
                plt.fill_between(x_range, y_ci[:, 0], y_ci[:, 1], alpha=0.2)
                
                print(f"  Fitted model for {interaction_var}={value}, slope={model.params.iloc[1]:.3f}")
            except Exception as e:
                print(f"  Error fitting model for {interaction_var}={value}: {e}")
        
        plt.xlabel(f'{main_indep_var.replace("_", " ").title()} (Between-Person)')
        plt.ylabel(f'{dep_var.replace("_", " ").title()}')
        plt.title(f'Interaction Effect: {interaction_var.replace("_", " ").title()} × '
                f'{main_indep_var.replace("_", " ").title()} (Between-Person)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(script_dir, 'visualizations', f'between_interaction_{dep_var_clean}_{main_indep_var_clean}_{interaction_var_clean}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved between-person plot to {output_file}")
        plt.close()
        
        # Create within-person effect plot (if there's enough within-person variation)
        plt.figure(figsize=(10, 6))
        
        # Check if we have enough participants with multiple observations
        participant_counts = plot_data.groupby('unique_participant_id').size()
        participants_with_multiple = participant_counts[participant_counts > 1].index
        
        if len(participants_with_multiple) >= 5:
            print(f"  Found {len(participants_with_multiple)} participants with multiple observations")
            
            # Plot regression lines with confidence intervals for within-person effects
            for value in interaction_values:
                subset = plot_data[plot_data[interaction_var_clean] == value]
                # Make sure centered values don't have NaN
                subset = subset.dropna(subset=[f'{main_indep_var_clean}_centered', dep_var_clean])
                
                # Only proceed if we have enough data
                if len(subset) >= 10:
                    try:
                        # Create overall within-person effect line with confidence interval
                        X = sm.add_constant(subset[f'{main_indep_var_clean}_centered'])
                        model = sm.OLS(subset[dep_var_clean], X).fit()
                        
                        # Create a range of x values for the regression line
                        x_min = subset[f'{main_indep_var_clean}_centered'].min()
                        x_max = subset[f'{main_indep_var_clean}_centered'].max()
                        x_range = np.linspace(x_min, x_max, 100)
                        X_pred = sm.add_constant(x_range)
                        
                        # Get predictions and confidence intervals
                        y_pred = model.get_prediction(X_pred)
                        y_mean = y_pred.predicted_mean
                        y_ci = y_pred.conf_int(alpha=0.05)  # 95% confidence interval
                        
                        # Format legend label based on interaction variable
                        if interaction_var.lower() == 'gender':
                            legend_label = 'Male' if value == 'male' else 'Female'
                        else:
                            legend_label = f'{interaction_var}={value}'
                        
                        # Plot regression line with confidence interval
                        plt.plot(x_range, y_mean, linewidth=3, 
                                label=f'{legend_label} (slope={model.params.iloc[1]:.3f})')
                        plt.fill_between(x_range, y_ci[:, 0], y_ci[:, 1], alpha=0.2)
                        
                        print(f"  Fitted within-person model for {interaction_var}={value}, slope={model.params.iloc[1]:.3f}")
                    except Exception as e:
                        print(f"  Error fitting within-person model for {interaction_var}={value}: {e}")
                else:
                    print(f"  Not enough within-person data for {interaction_var}={value} (n={len(subset)})")
            
            plt.xlabel(f'{main_indep_var.replace("_", " ").title()} (Within-Person Deviation)')
            plt.ylabel(f'{dep_var.replace("_", " ").title()}')
            plt.title(f'Within-Person Effect: {interaction_var.replace("_", " ").title()} × '
                    f'{main_indep_var.replace("_", " ").title()}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, alpha=0.3)
            
            output_file = os.path.join(script_dir, 'visualizations', f'within_interaction_{dep_var_clean}_{main_indep_var_clean}_{interaction_var_clean}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved within-person plot to {output_file}")
        else:
            print(f"  Not enough within-person variation for {dep_var} with {interaction_var}")
        
        plt.close()

# Function to create marginal effects plots using model predictions
def create_marginal_effects_plot(results_file, dependent_vars, main_indep_var, interaction_var):
    """
    Create marginal effects plots based on the model results.
    """
    # Get the script's directory for output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nCreating marginal effects plot for {main_indep_var} × {interaction_var}")
    
    try:
        # Read the results file
        results = pd.read_csv(results_file)
        print(f"  Results file read successfully. Shape: {results.shape}")
    except Exception as e:
        print(f"  Error reading results file: {e}")
        print("  Trying with more flexible parsing...")
        
        try:
            results = pd.read_csv(results_file, engine='python', on_bad_lines='skip')
            print(f"  Results file read with flexible parsing. Shape: {results.shape}")
        except Exception as e2:
            print(f"  All parsing methods failed: {e2}")
            return
    
    # Process each dependent variable
    for dep_var in dependent_vars:
        dep_var_clean = dep_var.replace(' ', '_').replace('(', '').replace(')', '')
        print(f"\n  Creating marginal effects plot for {dep_var}")
        
        if dep_var not in results.columns:
            print(f"  ERROR: {dep_var} not found in results columns")
            continue
        
        # Find main effect and interaction rows using flexible pattern matching
        main_effect_row = None
        interaction_row = None
        
        # Look for main effect (between-person)
        for i, metric in enumerate(results['Metric']):
            if 'between' in str(metric).lower() and main_indep_var.lower() in str(metric).lower() and 'interaction' not in str(metric).lower():
                main_effect_row = i
                print(f"  Found main effect at row {i}: {metric}")
                break
        
        # Look for interaction effect
        for i, metric in enumerate(results['Metric']):
            metric_str = str(metric).lower()
            if ('between' in metric_str and 
                interaction_var.lower().replace(' ', '') in metric_str.replace(' ', '') and 
                main_indep_var.lower().replace(' ', '') in metric_str.replace(' ', '') and
                '×' in metric_str):
                interaction_row = i
                print(f"  Found interaction effect at row {i}: {metric}")
                break
        
        if main_effect_row is None or interaction_row is None:
            print("  ERROR: Could not find required parameters. Skipping this plot.")
            continue
        
        # Extract coefficients and standard errors
        try:
            # Get main effect coefficient and SE
            main_effect = float(str(results.iloc[main_effect_row][dep_var]).replace('<', '').split()[0])
            main_effect_se = float(str(results.iloc[main_effect_row + 1][dep_var]))
            
            # Get interaction effect coefficient and SE
            interaction_effect = float(str(results.iloc[interaction_row][dep_var]).replace('<', '').split()[0])
            interaction_effect_se = float(str(results.iloc[interaction_row + 1][dep_var]))
            
            print(f"  Main effect: {main_effect:.3f}, SE: {main_effect_se:.3f}")
            print(f"  Interaction effect: {interaction_effect:.3f}, SE: {interaction_effect_se:.3f}")
        except Exception as e:
            print(f"  Error extracting coefficients: {e}")
            print("  Attempting alternative extraction method...")
            
            try:
                # Look for p-value rows to identify coefficients
                for i, metric in enumerate(results['Metric']):
                    if str(metric) == 'p-value':
                        if i >= 2:  # Make sure we have at least two rows above
                            if 'between' in str(results.iloc[i-2]['Metric']).lower() and main_indep_var.lower() in str(results.iloc[i-2]['Metric']).lower():
                                main_effect = float(str(results.iloc[i-2][dep_var]))
                                main_effect_se = float(str(results.iloc[i-1][dep_var]))
                                print(f"  Found main effect using p-value method: {main_effect:.3f}, SE: {main_effect_se:.3f}")
                            
                            if interaction_var.lower() in str(results.iloc[i-2]['Metric']).lower() and '×' in str(results.iloc[i-2]['Metric']):
                                interaction_effect = float(str(results.iloc[i-2][dep_var]))
                                interaction_effect_se = float(str(results.iloc[i-1][dep_var]))
                                print(f"  Found interaction effect using p-value method: {interaction_effect:.3f}, SE: {interaction_effect_se:.3f}")
                
                if main_effect is None or interaction_effect is None:
                    raise ValueError("Could not extract coefficients using alternative method")
                    
            except Exception as e2:
                print(f"  Alternative extraction also failed: {e2}")
                print("  Skipping this plot.")
                continue
        
        # Calculate marginal effects
        # For reference group (typically female for gender): just the main effect
        reference_effect = main_effect
        reference_se = main_effect_se
        
        # For comparison group (typically male for gender): main effect + interaction
        comparison_effect = main_effect + interaction_effect
        
        # Calculate standard error for the combined effect (simplification assuming zero correlation)
        comparison_se = np.sqrt(main_effect_se**2 + interaction_effect_se**2)
        
        # Create the marginal effects plot
        plt.figure(figsize=(8, 6))
        
        # Set up categories based on interaction variable
        if interaction_var.lower() == 'gender':
            categories = ['Female', 'Male']
        elif interaction_var.lower() == 'age group':
            categories = ['Child/Adolescent', 'Adult']
        elif interaction_var.lower() == 'location type':
            categories = ['Urban', 'Suburban']
        else:
            categories = ['Reference', 'Comparison']
            
        effects = [reference_effect, comparison_effect]
        errors = [1.96 * reference_se, 1.96 * comparison_se]  # 95% CI
        
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
        
        output_file = os.path.join(script_dir, 'visualizations', f'marginal_effects_{dep_var_clean}_{main_indep_var.replace(" ", "_")}_{interaction_var.replace(" ", "_")}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved marginal effects plot to {output_file}")

        # Create predicted values plot
        plt.figure(figsize=(10, 6))
        
        # Create data for predictions
        x_range = np.linspace(-2, 2, 100)  # Standardized range for main effect
        
        # Predict values for reference group
        reference_y = reference_effect * x_range
        reference_ci_lower = reference_y - 1.96 * reference_se
        reference_ci_upper = reference_y + 1.96 * reference_se
        
        # Predict values for comparison group
        comparison_y = comparison_effect * x_range
        comparison_ci_lower = comparison_y - 1.96 * comparison_se
        comparison_ci_upper = comparison_y + 1.96 * comparison_se
        
        # Plot with custom colors
        plt.plot(x_range, reference_y, label=categories[0], linewidth=2, color='#FF9999')
        plt.fill_between(x_range, reference_ci_lower, reference_ci_upper, alpha=0.2, color='#FF9999')
        
        plt.plot(x_range, comparison_y, label=categories[1], linewidth=2, color='#6699CC')
        plt.fill_between(x_range, comparison_ci_lower, comparison_ci_upper, alpha=0.2, color='#6699CC')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        plt.xlabel(f'{main_indep_var} (Standardized)')
        plt.ylabel(f'Predicted {dep_var}')
        plt.title(f'Predicted Effect of {main_indep_var} by {interaction_var}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(script_dir, 'visualizations', f'predicted_values_{dep_var_clean}_{main_indep_var.replace(" ", "_")}_{interaction_var.replace(" ", "_")}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved predicted values plot to {output_file}")

def format_small_number(value):
    """Format small numbers with more significant digits."""
    if abs(value) < 0.001:
        return f"{value:.6f}"
    elif abs(value) < 0.01:
        return f"{value:.5f}"
    elif abs(value) < 0.1:
        return f"{value:.4f}"
    else:
        return f"{value:.3f}"

# Example usage
if __name__ == "__main__":
    # Load the dataset using relative path
    file_path = "processed/pooled_stai_data_population_cleaned.csv"
    try:
        print(f"Attempting to load data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        
        '''
        columns:
        Index(['Participant ID', 'Dataset Source', 'Anxiety (Z)', 'Anxiety (Raw)',
            'Depressed Mood (Z)', 'Depressed Mood (Raw)', 'Gender', 'Location Type',
            'Age Group', 'Weekend Status', 'Digital Fragmentation',
            'Mobility Fragmentation', 'Digital Mobile Fragmentation',
            'Digital Home Fragmentation', 'Digital Home Mobility Delta',
            'Digital Duration', 'Mobile Duration', 'Digital Mobile Duration',
            'Digital Home Duration', 'Active Transport Duration',
            'Mechanized Transport Duration', 'Home Duration',
            'Mobility Episode Count', 'Intercept', 'unique_participant_id'],
            dtype='object')
        '''

        # Define variables
        dep_vars = ['Anxiety (Z)', 'Depressed Mood (Z)']
        main_ind_var = 'Digital Mobile Fragmentation'
        control_vars = ['Digital Mobile Duration', 'Mobility Episode Count', 'Mobile Duration','Age Group', 'Weekend Status', 'Gender', 'Location Type']
        
        # Add Intercept column required by statsmodels
        df["Intercept"] = 1
        
        # Specify interaction term - can be changed as needed
        interaction_terms = ["Gender"]
        
        print(f"Running multilevel models for {len(dep_vars)} dependent variables")
        print(f"Main independent variable: {main_ind_var}")
        print(f"Control variables: {control_vars}")
        if interaction_terms:
            print(f"Interaction terms: {main_ind_var} × {', '.join(interaction_terms)}")
        
        # Also use absolute path for output file
        output_file = os.path.join(script_dir, "results", "multilevel_fragmentation_results_with_interactions.csv")
        
        # Run the analyses
        results = run_multilevel_models(
            data=df,
            dependent_vars=dep_vars,
            main_indep_var=main_ind_var,
            control_vars=control_vars,
            interaction_terms=interaction_terms,
            output_file=output_file,
            create_visualizations=True
        )
        
        print("\nAnalysis complete!")
        print(df.columns)
        
    except Exception as e:
        print(f"Error during script execution: {e}")
        import traceback
        traceback.print_exc()
		
	