# statistical_tests.py
from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd

class StatisticalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.all_potential_controls = [
            'Gender_binary', 'Class', 'School_location',
            'total_duration_mobility_z', 'avg_duration_mobility_z', 'count_mobility_z',
            'is_weekend'
        ]
        self.control_variables = []
    
    def correlation_analysis(self, data, var1, var2, control_vars=None):
        if control_vars is None:
            corr, p_value = stats.pearsonr(data[var1], data[var2])
            return {
                'type': 'bivariate',
                'correlation': corr,
                'p_value': p_value,
                'n': len(data)
            }
        else:
            control_matrix = data[control_vars]
            partial_corr = pd.DataFrame(
                np.linalg.pinv(data[[var1, var2] + control_vars].corr().values),
                columns=data[[var1, var2] + control_vars].columns,
                index=data[[var1, var2] + control_vars].columns
            )
            r_partial = -partial_corr.iloc[0,1] / np.sqrt(partial_corr.iloc[0,0] * partial_corr.iloc[1,1])
            t_stat = r_partial * np.sqrt((len(data)-len(control_vars)-2)/(1-r_partial**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data)-len(control_vars)-2))
            
            return {
                'type': 'partial',
                'correlation': r_partial,
                'p_value': p_value,
                'control_vars': control_vars,
                'n': len(data)
            }
    
    def set_control_variables(self, data):
        self.control_variables = [var for var in self.all_potential_controls 
                                if var in data.columns]
    
    def regression_analysis(self, data, predictor, outcome, potential_controls=None):
        if not self.control_variables:
            self.set_control_variables(data)
            
        if potential_controls is None:
            potential_controls = self.control_variables.copy()
        
        X_base = sm.add_constant(data[predictor])
        base_model = sm.OLS(data[outcome], X_base).fit()
        base_results = {
            'type': 'base',
            'predictor': predictor,
            'outcome': outcome,
            'coefficient': base_model.params[predictor],
            'std_error': base_model.bse[predictor],
            'p_value': base_model.pvalues[predictor],
            'r_squared': base_model.rsquared,
            'adj_r_squared': base_model.rsquared_adj,
            'n': len(data)
        }
        
        current_controls = []
        best_model = base_model
        improvement_threshold = 0.01
        
        while potential_controls:
            best_improvement = 0
            best_control = None
            
            for control in potential_controls:
                X_test = sm.add_constant(data[[predictor] + current_controls + [control]])
                test_model = sm.OLS(data[outcome], X_test).fit()
                improvement = test_model.rsquared_adj - best_model.rsquared_adj
                
                if improvement > best_improvement and improvement > improvement_threshold:
                    best_improvement = improvement
                    best_control = control
            
            if best_control:
                current_controls.append(best_control)
                potential_controls.remove(best_control)
                X_best = sm.add_constant(data[[predictor] + current_controls])
                best_model = sm.OLS(data[outcome], X_best).fit()
            else:
                break
        
        if current_controls:
            controlled_results = {
                'type': 'controlled',
                'predictor': predictor,
                'outcome': outcome,
                'coefficient': best_model.params[predictor],
                'std_error': best_model.bse[predictor],
                'p_value': best_model.pvalues[predictor],
                'r_squared': best_model.rsquared,
                'adj_r_squared': best_model.rsquared_adj,
                'n': len(data),
                'control_vars': current_controls,
                'control_coefficients': {var: best_model.params[var] for var in current_controls},
                'control_p_values': {var: best_model.pvalues[var] for var in current_controls}
            }
            return [base_results, controlled_results]
        
        return [base_results]
    
    def group_comparison(self, data, group_var, outcome_var, threshold=None, control_vars=None):
        if group_var == 'Gender':
            group1 = data[data[group_var] == 'נקבה'][outcome_var]
            group2 = data[data[group_var] == 'זכר'][outcome_var]
        else:
            if threshold is not None:
                data = data.copy()
                data[group_var] = data[group_var] > threshold
                group1 = data[data[group_var]][outcome_var]
                group2 = data[~data[group_var]][outcome_var]
            else:
                unique_values = sorted(data[group_var].unique())
                group1 = data[data[group_var] == unique_values[0]][outcome_var]
                group2 = data[data[group_var] == unique_values[1]][outcome_var]
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        
        base_results = {
            'type': 'unadjusted',
            'test': 't_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std(),
            'group1_n': len(group1),
            'group2_n': len(group2)
        }
        
        if control_vars:
            data_copy = data.copy()
            if group_var == 'Gender':
                data_copy['Gender_binary'] = data_copy[group_var].map({'נקבה': 0, 'זכר': 1})
                formula = f"{outcome_var} ~ C(Gender_binary) + " + " + ".join(control_vars)
            else:
                formula = f"{outcome_var} ~ C({group_var}) + " + " + ".join(control_vars)
            
            model = sm.OLS.from_formula(formula, data=data_copy).fit()
            
            if group_var == 'Gender':
                param_name = "C(Gender_binary)[T.1]"
            else:
                param_name = f"C({group_var})[T.True]" if threshold is not None else f"C({group_var})[T.{unique_values[1]}]"
            
            controlled_results = {
                'type': 'adjusted',
                'test': 'ancova',
                'coefficient': model.params[param_name],
                'std_error': model.bse[param_name],
                'p_value': model.pvalues[param_name],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'control_vars': control_vars,
                'control_coefficients': {var: model.params[var] for var in control_vars},
                'control_p_values': {var: model.pvalues[var] for var in control_vars},
                'n': len(data)
            }
            return [base_results, controlled_results]
        
        return [base_results]