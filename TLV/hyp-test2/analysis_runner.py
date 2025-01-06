import pandas as pd
from multilevel_analyzer import MultilevelAnalyzer

# analysis_runner.py
class AnalysisRunner:
    def __init__(self, config, analyzer, results_manager):
        self.config = config
        self.analyzer = analyzer
        self.results_manager = results_manager
        self.multilevel_analyzer = MultilevelAnalyzer(config)

    def run_mobility_analysis(self, data):
        results = []
        for frag_index in self.config.frag_indices:
            for mobility_metric in ['total_duration_mobility', 'avg_duration_mobility', 'count_mobility']:
                # Correlation analysis
                corr_result = self.analyzer.correlation_analysis(data, frag_index, mobility_metric)
                corr_row = {
                    'analysis_type': 'mobility',
                    'frag_index': frag_index,
                    'mobility_metric': mobility_metric,
                    'type': 'correlation',
                    'correlation': corr_result['correlation'],
                    'p_value': corr_result['p_value'],
                    'n': corr_result['n']
                }
                results.append(corr_row)
                
                # Regression analysis
                reg_results = self.analyzer.regression_analysis(data, frag_index, mobility_metric)
                for reg_result in reg_results:
                    reg_row = {
                        'analysis_type': 'mobility',
                        'frag_index': frag_index,
                        'mobility_metric': mobility_metric,
                        'type': reg_result['type'],
                        'predictor': frag_index,
                        'outcome': mobility_metric,
                        'coefficient': reg_result['coefficient'],
                        'std_error': reg_result['std_error'],
                        'p_value': reg_result['p_value'],
                        'r_squared': reg_result['r_squared'],
                        'adj_r_squared': reg_result['adj_r_squared'],
                        'n': reg_result['n']
                    }
                    
                    if 'control_vars' in reg_result:
                        reg_row['control_vars'] = str(reg_result['control_vars'])
                        reg_row['control_coefficients'] = str(reg_result['control_coefficients'])
                        reg_row['control_p_values'] = str(reg_result['control_p_values'])
                    
                    results.append(reg_row)
        
        return pd.DataFrame(results)
    
    def run_emotional_analysis(self, data):
        results = []
        for frag_index in self.config.frag_indices:
            for outcome in self.config.emotional_outcomes:
                # Correlation analysis
                corr_result = self.analyzer.correlation_analysis(data, frag_index, outcome)
                corr_row = {
                    'analysis_type': 'emotional',
                    'frag_index': frag_index,
                    'outcome': outcome,
                    'type': 'correlation',
                    'correlation': corr_result['correlation'],
                    'p_value': corr_result['p_value'],
                    'n': corr_result['n']
                }
                results.append(corr_row)
                
                # Regression analysis
                reg_results = self.analyzer.regression_analysis(data, frag_index, outcome)
                for reg_result in reg_results:
                    reg_row = {
                        'analysis_type': 'emotional',
                        'frag_index': frag_index,
                        'outcome': outcome,
                        'type': reg_result['type'],
                        'predictor': frag_index,
                        'coefficient': reg_result['coefficient'],
                        'std_error': reg_result['std_error'],
                        'p_value': reg_result['p_value'],
                        'r_squared': reg_result['r_squared'],
                        'adj_r_squared': reg_result['adj_r_squared'],
                        'n': reg_result['n']
                    }
                    
                    if 'control_vars' in reg_result:
                        reg_row['control_vars'] = str(reg_result['control_vars'])
                        reg_row['control_coefficients'] = str(reg_result['control_coefficients'])
                        reg_row['control_p_values'] = str(reg_result['control_p_values'])
                    
                    results.append(reg_row)
                
                # Group comparison - only median split
                group_results = self.analyzer.group_comparison(
                    data, frag_index, outcome,
                    threshold=data[frag_index].median()
                )
                
                for group_result in group_results:
                    group_row = {
                        'analysis_type': 'emotional',
                        'frag_index': frag_index,
                        'outcome': outcome,
                        'type': group_result['type'],
                        'test': group_result['test'],
                        'p_value': group_result['p_value'],
                        't_statistic': group_result.get('t_statistic'),
                        'effect_size': group_result.get('effect_size'),
                        'group1_mean': group_result.get('group1_mean'),
                        'group2_mean': group_result.get('group2_mean'),
                        'group1_std': group_result.get('group1_std'),
                        'group2_std': group_result.get('group2_std'),
                        'group1_n': group_result.get('group1_n'),
                        'group2_n': group_result.get('group2_n'),
                        'n': group_result.get('n')
                    }
                    
                    if 'control_vars' in group_result:
                        group_row['control_vars'] = str(group_result['control_vars'])
                        group_row['control_coefficients'] = str(group_result['control_coefficients'])
                        group_row['control_p_values'] = str(group_result['control_p_values'])
                    
                    results.append(group_row)
        
        return pd.DataFrame(results)
    
    def run_population_analysis(self, data):
        results = []
        for factor in self.config.population_factors:
            for frag_index in self.config.frag_indices:
                group_results = self.analyzer.group_comparison(
                    data, factor, frag_index,
                    control_vars=[var for var in self.analyzer.control_variables if var != factor]
                )
                
                for result in group_results:
                    row = {
                        'analysis_type': 'population',
                        'factor': factor,
                        'frag_index': frag_index,
                        'type': result['type'],
                        'test': result['test'],
                        'p_value': result['p_value']
                    }
                    
                    # Add t-test specific fields
                    if result['test'] == 't_test':
                        row.update({
                            't_statistic': result['t_statistic'],
                            'effect_size': result['effect_size'],
                            'group1_mean': result['group1_mean'],
                            'group2_mean': result['group2_mean'],
                            'group1_std': result['group1_std'],
                            'group2_std': result['group2_std'],
                            'group1_n': result['group1_n'],
                            'group2_n': result['group2_n']
                        })
                    # Add regression specific fields
                    elif result['test'] == 'ancova':
                        row.update({
                            'coefficient': result['coefficient'],
                            'std_error': result['std_error'],
                            'r_squared': result['r_squared'],
                            'adj_r_squared': result['adj_r_squared'],
                            'n': result['n']
                        })
                        
                        if 'control_vars' in result:
                            row['control_vars'] = str(result['control_vars'])
                            row['control_coefficients'] = str(result['control_coefficients'])
                            row['control_p_values'] = str(result['control_p_values'])
                    
                    results.append(row)
        
        return pd.DataFrame(results)

    def format_multilevel_results(self, model_results, frag_index, outcome):
        """Format multilevel analysis results"""
        formatted_results = []
        
        if isinstance(model_results, dict):  # For fallback results
            formatted_results.append({
                'analysis_type': 'usage_group',
                'model_type': 'multilevel_fallback',
                'frag_index': frag_index,
                'outcome': outcome,
                **model_results
            })
        else:  # For successful MLM
            formatted_results.append({
                'analysis_type': 'usage_group',
                'model_type': 'multilevel',
                'frag_index': frag_index,
                'outcome': outcome,
                'coefficient': model_results.params[frag_index + '_std'],
                'std_error': model_results.bse[frag_index + '_std'],
                'p_value': model_results.pvalues[frag_index + '_std'],
                'random_effects_var': model_results.cov_re.values[0][0]
            })
        
        return formatted_results

    def format_between_subject_results(self, between_results, frag_index, outcome):
        """Format between-subject analysis results"""
        formatted_results = []
        
        # Regression results
        model = between_results['regression']
        formatted_results.append({
            'analysis_type': 'usage_group',
            'model_type': 'between_subject',
            'frag_index': frag_index,
            'outcome': outcome,
            'coefficient': model.params[frag_index],
            'std_error': model.bse[frag_index],
            'p_value': model.pvalues[frag_index],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        })
        
        # Group effects if available
        if between_results['group_effects']:
            formatted_results.append({
                'analysis_type': 'usage_group',
                'model_type': 'group_effects',
                'frag_index': frag_index,
                'outcome': outcome,
                **between_results['group_effects']
            })
        
        return formatted_results

    def run_usage_group_analysis(self, data):
        """Enhanced analysis preserving all data"""
        results = []
        
        for frag_index in self.config.frag_indices:
            for outcome in self.config.emotional_outcomes:
                # Run comprehensive analysis
                analysis_results = self.multilevel_analyzer.run_analysis(
                    data, frag_index, outcome
                )
                
                if analysis_results:
                    # Process multilevel results if available
                    if analysis_results['multilevel']:
                        results.extend(self.format_multilevel_results(
                            analysis_results['multilevel'],
                            frag_index,
                            outcome
                        ))
                    
                    # Process between-subject results
                    if analysis_results['between_subject']:
                        results.extend(self.format_between_subject_results(
                            analysis_results['between_subject'],
                            frag_index,
                            outcome
                        ))
                    
                    # Process correlational results
                    if analysis_results['correlational']:
                        results.append({
                            'analysis_type': 'usage_group',
                            'model_type': 'correlational',
                            'frag_index': frag_index,
                            'outcome': outcome,
                            **analysis_results['correlational']
                        })
                    
                    # Add data structure information
                    results.append({
                        'analysis_type': 'usage_group',
                        'model_type': 'data_structure',
                        'frag_index': frag_index,
                        'outcome': outcome,
                        **analysis_results['data_structure']
                    })
        
        return pd.DataFrame(results)