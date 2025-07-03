"""
Value at Risk (VaR) Calculator
Professional implementation with multiple VaR methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy import stats
from datetime import datetime, timedelta

class VaRCalculator:
    """
    Professional VaR Calculator
    
    Methods:
    - Historical Simulation
    - Parametric (Normal)
    - Monte Carlo Simulation
    - Conditional VaR (CVaR/Expected Shortfall)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "confidence_levels": [0.95, 0.99],
            "lookback_days": 252,
            "monte_carlo_simulations": 10000,
            "correlation_adjustment": True,
            "volatility_scaling": True
        }
    
    def calculate_portfolio_var(self, 
                               positions: Dict[str, float], 
                               returns_data: pd.DataFrame,
                               method: str = "historical") -> Dict[str, Any]:
        """
        Calculate portfolio VaR using specified method
        """
        
        if method == "historical":
            return self._historical_var(positions, returns_data)
        elif method == "parametric":
            return self._parametric_var(positions, returns_data)
        elif method == "monte_carlo":
            return self._monte_carlo_var(positions, returns_data)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, positions: Dict[str, float], returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Historical Simulation VaR"""
        
        # Get portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        if len(portfolio_returns) == 0:
            return self._empty_var_result()
        
        results = {}
        
        for confidence_level in self.config["confidence_levels"]:
            alpha = 1 - confidence_level
            
            # Calculate VaR
            var_value = np.percentile(portfolio_returns, alpha * 100)
            
            # Calculate CVaR (Expected Shortfall)
            cvar_value = portfolio_returns[portfolio_returns <= var_value].mean()
            
            results[f"var_{int(confidence_level*100)}"] = abs(var_value)
            results[f"cvar_{int(confidence_level*100)}"] = abs(cvar_value)
        
        # Additional metrics
        results.update({
            "method": "historical",
            "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
            "max_loss": abs(portfolio_returns.min()),
            "avg_loss": abs(portfolio_returns[portfolio_returns < 0].mean()) if (portfolio_returns < 0).any() else 0,
            "observations": len(portfolio_returns)
        })
        
        return results
    
    def _parametric_var(self, positions: Dict[str, float], returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Parametric VaR (assumes normal distribution)"""
        
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        if len(portfolio_returns) == 0:
            return self._empty_var_result()
        
        # Calculate mean and standard deviation
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        results = {}
        
        for confidence_level in self.config["confidence_levels"]:
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # VaR calculation
            var_value = mean_return + z_score * std_return
            
            # CVaR calculation (for normal distribution)
            cvar_value = mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)
            
            results[f"var_{int(confidence_level*100)}"] = abs(var_value)
            results[f"cvar_{int(confidence_level*100)}"] = abs(cvar_value)
        
        results.update({
            "method": "parametric",
            "portfolio_volatility": std_return * np.sqrt(252),
            "mean_return": mean_return * 252,
            "skewness": stats.skew(portfolio_returns),
            "kurtosis": stats.kurtosis(portfolio_returns)
        })
        
        return results
    
    def _monte_carlo_var(self, positions: Dict[str, float], returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Monte Carlo Simulation VaR"""
        
        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()
        
        # Calculate mean returns and volatilities
        mean_returns = returns_data.mean()
        volatilities = returns_data.std()
        
        # Generate random scenarios
        simulated_returns = self._generate_monte_carlo_scenarios(
            mean_returns, volatilities, correlation_matrix
        )
        
        # Calculate portfolio returns for each scenario
        portfolio_scenarios = []
        for scenario in simulated_returns:
            portfolio_return = sum(positions[asset] * scenario[i] 
                                 for i, asset in enumerate(returns_data.columns) 
                                 if asset in positions)
            portfolio_scenarios.append(portfolio_return)
        
        portfolio_scenarios = np.array(portfolio_scenarios)
        
        results = {}
        
        for confidence_level in self.config["confidence_levels"]:
            alpha = 1 - confidence_level
            
            # Calculate VaR
            var_value = np.percentile(portfolio_scenarios, alpha * 100)
            
            # Calculate CVaR
            cvar_value = portfolio_scenarios[portfolio_scenarios <= var_value].mean()
            
            results[f"var_{int(confidence_level*100)}"] = abs(var_value)
            results[f"cvar_{int(confidence_level*100)}"] = abs(cvar_value)
        
        results.update({
            "method": "monte_carlo",
            "simulations": len(portfolio_scenarios),
            "portfolio_volatility": portfolio_scenarios.std() * np.sqrt(252),
            "scenarios_generated": self.config["monte_carlo_simulations"]
        })
        
        return results
    
    def _calculate_portfolio_returns(self, positions: Dict[str, float], returns_data: pd.DataFrame) -> pd.Series:
        """Calculate historical portfolio returns"""
        
        # Filter returns data for assets in portfolio
        portfolio_assets = [asset for asset in positions.keys() if asset in returns_data.columns]
        
        if not portfolio_assets:
            return pd.Series([])
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        
        for asset in portfolio_assets:
            weight = positions[asset]
            portfolio_returns += weight * returns_data[asset]
        
        return portfolio_returns.dropna()
    
    def _generate_monte_carlo_scenarios(self, mean_returns: pd.Series, 
                                       volatilities: pd.Series, 
                                       correlation_matrix: pd.DataFrame) -> np.ndarray:
        """Generate Monte Carlo scenarios"""
        
        n_assets = len(mean_returns)
        n_simulations = self.config["monte_carlo_simulations"]
        
        # Cholesky decomposition for correlation
        try:
            chol_matrix = np.linalg.cholesky(correlation_matrix.values)
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use identity
            chol_matrix = np.eye(n_assets)
        
        # Generate random normal variables
        random_normals = np.random.normal(0, 1, (n_simulations, n_assets))
        
        # Apply correlation structure
        correlated_normals = random_normals @ chol_matrix.T
        
        # Generate scenarios
        scenarios = []
        for i in range(n_simulations):
            scenario_returns = []
            for j, asset in enumerate(mean_returns.index):
                return_value = mean_returns[asset] + volatilities[asset] * correlated_normals[i, j]
                scenario_returns.append(return_value)
            scenarios.append(scenario_returns)
        
        return np.array(scenarios)
    
    def _empty_var_result(self) -> Dict[str, Any]:
        """Return empty VaR result when no data available"""
        results = {}
        for confidence_level in self.config["confidence_levels"]:
            results[f"var_{int(confidence_level*100)}"] = 0.0
            results[f"cvar_{int(confidence_level*100)}"] = 0.0
        
        results.update({
            "method": "none",
            "portfolio_volatility": 0.0,
            "observations": 0
        })
        
        return results
    
    def calculate_component_var(self, positions: Dict[str, float], 
                               returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        if len(portfolio_returns) == 0:
            return {}
        
        component_vars = {}
        
        # Calculate marginal VaR for each asset
        for asset in positions.keys():
            if asset not in returns_data.columns:
                continue
            
            # Calculate correlation with portfolio
            asset_returns = returns_data[asset].dropna()
            correlation = portfolio_returns.corr(asset_returns)
            
            # Asset volatility
            asset_vol = asset_returns.std()
            
            # Portfolio volatility
            portfolio_vol = portfolio_returns.std()
            
            # Marginal VaR
            if portfolio_vol > 0:
                marginal_var = correlation * asset_vol / portfolio_vol
                component_var = positions[asset] * marginal_var
                component_vars[asset] = component_var
        
        return component_vars
    
    def stress_test_var(self, positions: Dict[str, float], 
                       returns_data: pd.DataFrame,
                       stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform stress testing on VaR calculations"""
        
        stress_results = {}
        
        for i, scenario in enumerate(stress_scenarios):
            # Apply stress scenario to returns
            stressed_returns = returns_data.copy()
            
            for asset, shock in scenario.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock
            
            # Calculate VaR under stress
            stressed_var = self.calculate_portfolio_var(positions, stressed_returns, "historical")
            stress_results[f"scenario_{i+1}"] = stressed_var
        
        return stress_results
    
    def get_var_summary(self, positions: Dict[str, float], 
                       returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive VaR summary"""
        
        # Calculate VaR using all methods
        historical_var = self.calculate_portfolio_var(positions, returns_data, "historical")
        parametric_var = self.calculate_portfolio_var(positions, returns_data, "parametric")
        monte_carlo_var = self.calculate_portfolio_var(positions, returns_data, "monte_carlo")
        
        # Component VaR
        component_var = self.calculate_component_var(positions, returns_data)
        
        return {
            "historical_var": historical_var,
            "parametric_var": parametric_var,
            "monte_carlo_var": monte_carlo_var,
            "component_var": component_var,
            "summary": {
                "var_95_avg": np.mean([
                    historical_var.get("var_95", 0),
                    parametric_var.get("var_95", 0),
                    monte_carlo_var.get("var_95", 0)
                ]),
                "var_99_avg": np.mean([
                    historical_var.get("var_99", 0),
                    parametric_var.get("var_99", 0),
                    monte_carlo_var.get("var_99", 0)
                ])
            }
        } 