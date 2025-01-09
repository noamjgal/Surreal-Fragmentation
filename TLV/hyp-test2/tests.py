# analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

@dataclass
class AnalysisResult:
    """Standardized container for analysis results"""
    test_type: str  # 'regression' or 't_test'
    predictor: str
    outcome: str
    coefficient: float  # beta coefficient for regression, difference in means for t-test
    std_error: float
    p_value: float
    effect_size: float  # R-squared for regression, Cohen's d for t-test
    n: int
    control_vars: Optional[List[str]] = None
    control_results: Optional[Dict[str, Dict[str, float]]] = None

class StatisticalAnalyzer:
    """
    Consolidated statistical analysis class that performs:
    1. Linear regression (with optional controls)
    2. T-tests for group comparisons
    """
    
    def __init__(self, control_variables: Optional[List[str]] = None):
        self.control_variables = control_variables or []
    
    def run_regression(self, 
                    data: pd.DataFrame, 
                    predictor: str, 
                    outcome: str,
                    control_vars: Optional[List[str]] = None) -> AnalysisResult:
        """
        Run linear regression with optional control variables
        """
        # Check data types before proceeding
        try:
            y = pd.to_numeric(data[outcome])
            X_base = pd.to_numeric(data[predictor])
        except Exception as e:
            print(f"Non-numeric data found in {predictor} or {outcome}")
            print(f"Predictor type: {data[predictor].dtype}, Outcome type: {data[outcome].dtype}")
            raise ValueError(f"Data must be numeric for regression analysis")
        
        # Start with base model
        X = sm.add_constant(X_base)
        base_model = sm.OLS(y, X).fit()
        
        # If no control variables, return base model results
        if not control_vars:
            return AnalysisResult(
                test_type='regression',
                predictor=predictor,
                outcome=outcome,
                coefficient=base_model.params[predictor],
                std_error=base_model.bse[predictor],
                p_value=base_model.pvalues[predictor],
                effect_size=base_model.rsquared,
                n=len(data)
            )
        
        # Add control variables
        X = pd.DataFrame(X)  # Convert to DataFrame to preserve column names
        for var in control_vars:
            X[var] = pd.to_numeric(data[var])
        
        controlled_model = sm.OLS(y, X).fit()
        
        # Prepare control variable results
        control_results = {
            var: {
                'coefficient': controlled_model.params[var],
                'std_error': controlled_model.bse[var],
                'p_value': controlled_model.pvalues[var]
            }
            for var in control_vars
        }
        
        return AnalysisResult(
            test_type='regression',
            predictor=predictor,
            outcome=outcome,
            coefficient=controlled_model.params[predictor],
            std_error=controlled_model.bse[predictor],
            p_value=controlled_model.pvalues[predictor],
            effect_size=controlled_model.rsquared,
            n=len(data),
            control_vars=control_vars,
            control_results=control_results
        )
    
    def run_ttest(self,
                  data: pd.DataFrame,
                  group_var: str,
                  outcome: str,
                  group_threshold: Optional[float] = None) -> AnalysisResult:
        """
        Run t-test comparing two groups
        Groups can be defined by a binary variable or by splitting a continuous 
        variable at a threshold (defaults to median if not specified)
        """
        # Prepare groups
        if group_threshold is None and data[group_var].nunique() > 2:
            group_threshold = data[group_var].median()
        
        if group_threshold is not None:
            group1_mask = data[group_var] <= group_threshold
            group1 = data[group1_mask][outcome]
            group2 = data[~group1_mask][outcome]
        else:
            # For binary variables
            groups = data[group_var].unique()
            if len(groups) != 2:
                raise ValueError(f"Group variable {group_var} must have exactly 2 categories")
            group1 = data[data[group_var] == groups[0]][outcome]
            group2 = data[data[group_var] == groups[1]][outcome]
        
        # Run t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
        cohens_d = (group1.mean() - group2.mean()) / pooled_std
        
        return AnalysisResult(
            test_type='t_test',
            predictor=group_var,
            outcome=outcome,
            coefficient=group1.mean() - group2.mean(),  # mean difference
            std_error=np.sqrt(group1.var()/len(group1) + group2.var()/len(group2)),
            p_value=p_value,
            effect_size=cohens_d,
            n=len(group1) + len(group2)
        )

class AnalysisRunner:
    """
    Runner class to execute multiple analyses and format results
    """
    def __init__(self, analyzer: StatisticalAnalyzer):
        self.analyzer = analyzer
        
    def run_analyses(self,
                    data: pd.DataFrame,
                    predictors: List[str],
                    outcomes: List[str],
                    control_vars: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run all specified analyses and return results as a DataFrame
        """
        results = []
        
        # Run regressions for continuous predictors
        continuous_predictors = [
            pred for pred in predictors 
            if data[pred].nunique() > 2
        ]
        
        for predictor in continuous_predictors:
            for outcome in outcomes:
                result = self.analyzer.run_regression(
                    data, predictor, outcome, control_vars
                )
                results.append(self._result_to_dict(result))
        
        # Run t-tests for binary predictors
        binary_predictors = [
            pred for pred in predictors 
            if data[pred].nunique() == 2
        ]
        
        for predictor in binary_predictors:
            for outcome in outcomes:
                result = self.analyzer.run_ttest(
                    data, predictor, outcome
                )
                results.append(self._result_to_dict(result))
        
        return pd.DataFrame(results)
    
    def _result_to_dict(self, result: AnalysisResult) -> Dict:
        """Convert AnalysisResult to dictionary for DataFrame creation"""
        result_dict = {
            'test_type': result.test_type,
            'predictor': result.predictor,
            'outcome': result.outcome,
            'coefficient': result.coefficient,
            'std_error': result.std_error,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'n': result.n
        }
        
        if result.control_vars:
            result_dict['control_vars'] = ','.join(result.control_vars)
            
            # Flatten control variable results
            for var, stats in result.control_results.items():
                for stat_name, value in stats.items():
                    result_dict[f'{var}_{stat_name}'] = value
        
        return result_dict