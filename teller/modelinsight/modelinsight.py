import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, partial_dependence
import json
import warnings
warnings.filterwarnings('ignore')


class ModelInsight:
    """
    A comprehensive ML model interpretation toolkit with rich visualizations.

    This class provides unified access to model diagnostics, feature importance (permutation and built-in),
    partial dependence plots, individual prediction explanations, confusion matrix, regression/classification metrics,
    residual analysis, feature statistics, and interactive D3.js dashboards (with dark mode support).

    Features
    --------
    - Feature importance (permutation, built-in, or both)
    - Partial dependence plots
    - Individual prediction explanations
    - Confusion matrix & classification metrics
    - Residual plots for regression
    - What-if analysis
    - Feature statistics
    - Interactive D3.js visualizations with dark mode

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Trained sklearn-compatible model (regressor or classifier).
    X : pandas.DataFrame
        Training features.
    y : pandas.Series or np.ndarray
        Training target.
    X_test : pandas.DataFrame, optional
        Test features.
    y_test : pandas.Series or np.ndarray, optional
        Test target.
    feature_names : list of str, optional
        List of feature names (auto-detected from DataFrame if not provided).
    target_name : str, default="target"
        Name of target variable.
    model_type : str, default="auto"
        'classification', 'regression', or 'auto' (auto-detects type).

    Examples
    --------
    # Example usage
    if __name__ == "__main__":
        from sklearn.datasets import load_diabetes, load_iris
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Example 1: Regression
        print("Creating regression example...")
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelInsight(
            model=model,
            X=X_train,
            y=y_train,
            X_test=X_test,
            y_test=y_test,
            target_name="diabetes_progression"
        )

        # Generate and save dashboard
        explainer.save_html("diabetes_dashboard.html", dark_mode=True)

        # Example 2: Classification
        print("\nCreating classification example...")
        X, y = load_iris(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelInsight(
            model=model,
            X=X_train,
            y=y_train,
            X_test=X_test,
            y_test=y_test,
            target_name="iris_species"
        )

        explainer.save_html("iris_dashboard.html",
                            include_sections=['importance', 'metrics', 'predictions', 'confusion_matrix'],
                            dark_mode=False)

        # Example 3: Using API methods
        print("\n=== API Usage Examples ===")

        # Get feature importance
        importance = explainer.compute_feature_importance(method='permutation')
        print("\nTop 5 Important Features:")
        print(importance.nlargest(5, 'importance')[['feature', 'importance']])

        # Get metrics
        metrics = explainer.get_metrics()
        print("\nModel Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Explain single prediction
        explanation = explainer.explain_prediction(0)
        print("\nExplanation for first instance:")
        print(f"  Prediction: {explanation['prediction']:.2f}")
        print(f"  Actual: {explanation['actual']:.2f}")
        print(f"  Features: {list(explanation['features'].keys())[:3]}...")

        # Get feature statistics
        stats = explainer.get_feature_statistics()
        print("\nFeature Statistics (first feature):")
        print(stats.iloc[0])

        print("\n✅ Dashboards created successfully!")
        print("📊 Open 'diabetes_dashboard.html' or 'iris_dashboard.html' in your browser")
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        model_type: str = "auto"
    ):
        """
        Initialize the ModelInsight.
        
        Args:
            model: Trained sklearn-compatible model
            X: Training features (DataFrame)
            y: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            feature_names: List of feature names (auto-detected from DataFrame)
            target_name: Name of target variable
            model_type: 'classification', 'regression', or 'auto'
        """
        self.model = model
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.X_test = X_test if X_test is None or isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        self.y_test = y_test if y_test is None or isinstance(y_test, pd.Series) else pd.Series(y_test)
        
        self.feature_names = feature_names or list(self.X.columns)
        self.target_name = target_name
        
        # Auto-detect model type
        if model_type == "auto":
            self.model_type = self._detect_model_type()
        else:
            self.model_type = model_type
            
        # Cache for computed values
        self._cache = {}
        
    def _detect_model_type(self) -> str:
        """Detect if model is classification or regression."""
        if hasattr(self.model, '_estimator_type'):
            return self.model._estimator_type
        elif hasattr(self.model, 'predict_proba'):
            return 'classifier'
        else:
            return 'regressor'
    
    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get model predictions."""
        if X is None:
            X = self.X_test if self.X_test is not None else self.X
        return self.model.predict(X)
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities for classification."""
        if self.model_type != 'classifier' or not hasattr(self.model, 'predict_proba'):
            return None
        if X is None:
            X = self.X_test if self.X_test is not None else self.X
        return self.model.predict_proba(X)
    
    def compute_feature_importance(self, method: str = "permutation", n_repeats: int = 10) -> pd.DataFrame:
        """
        Compute feature importance.
        
        Args:
            method: 'permutation', 'builtin', or 'both'
            n_repeats: Number of repeats for permutation importance
        """
        cache_key = f"importance_{method}_{n_repeats}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        importance_data = []
        
        # Built-in importance (tree-based models)
        if method in ['builtin', 'both'] and hasattr(self.model, 'feature_importances_'):
            for i, name in enumerate(self.feature_names):
                importance_data.append({
                    'feature': name,
                    'importance': self.model.feature_importances_[i],
                    'method': 'builtin'
                })
        
        # Permutation importance
        if method in ['permutation', 'both']:
            X_eval = self.X_test if self.X_test is not None else self.X
            y_eval = self.y_test if self.y_test is not None else self.y
            
            perm_importance = permutation_importance(
                self.model, X_eval, y_eval,
                n_repeats=n_repeats,
                random_state=42
            )
            
            for i, name in enumerate(self.feature_names):
                importance_data.append({
                    'feature': name,
                    'importance': perm_importance.importances_mean[i],
                    'std': perm_importance.importances_std[i],
                    'method': 'permutation'
                })
        
        df = pd.DataFrame(importance_data)
        self._cache[cache_key] = df
        return df
    
    def compute_partial_dependence(self, features: List[Union[int, str]], grid_resolution: int = 50) -> Dict:
        """Compute partial dependence for given features."""
        feature_indices = []
        for f in features:
            if isinstance(f, str):
                feature_indices.append(self.feature_names.index(f))
            else:
                feature_indices.append(f)
        
        pd_result = partial_dependence(
            self.model, self.X, feature_indices,
            grid_resolution=grid_resolution
        )
        
        result = {
            'grid_values': [grid.tolist() for grid in pd_result['grid_values']],
            'average': pd_result['average'].tolist(),
            'features': [self.feature_names[i] for i in feature_indices]
        }
        
        return result
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix for classification."""
        if self.model_type != 'classifier' or self.y_test is None:
            return None
        
        from sklearn.metrics import confusion_matrix
        y_pred = self.get_predictions(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        if self.X_test is None or self.y_test is None:
            X_eval, y_eval = self.X, self.y
        else:
            X_eval, y_eval = self.X_test, self.y_test
        
        y_pred = self.get_predictions(X_eval)
        
        if self.model_type == 'classifier':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            try:
                return {
                    'accuracy': accuracy_score(y_eval, y_pred),
                    'precision': precision_score(y_eval, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_eval, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_eval, y_pred, average='weighted', zero_division=0)
                }
            except:
                return {'accuracy': accuracy_score(y_eval, y_pred)}
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            return {
                'rmse': np.sqrt(mean_squared_error(y_eval, y_pred)),
                'mae': mean_absolute_error(y_eval, y_pred),
                'r2': r2_score(y_eval, y_pred)
            }
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """Get descriptive statistics for all features."""
        stats = self.X.describe().T
        stats['missing'] = self.X.isnull().sum()
        stats['missing_pct'] = (self.X.isnull().sum() / len(self.X)) * 100
        return stats
    
    def explain_prediction(self, idx: int) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            idx: Index of the instance to explain
        """
        X_instance = self.X.iloc[idx:idx+1]
        prediction = self.model.predict(X_instance)[0]
        
        explanation = {
            'index': idx,
            'prediction': float(prediction),
            'actual': float(self.y.iloc[idx]) if idx < len(self.y) else None,
            'features': {}
        }
        
        # Add feature values
        for feature in self.feature_names:
            explanation['features'][feature] = float(X_instance[feature].values[0])
        
        # Add probability if classification
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_instance)[0]
            explanation['probabilities'] = proba.tolist()
        
        return explanation
    
    def to_html(
        self,
        include_sections: Optional[List[str]] = None,
        dark_mode: bool = False,
        title: str = "Model Explainer Dashboard"
    ) -> str:
        """
        Generate interactive HTML dashboard with D3.js visualizations.
        
        Args:
            include_sections: List of sections to include. Options:
                ['importance', 'metrics', 'predictions', 'residuals', 'confusion_matrix', 'stats']
            dark_mode: Start in dark mode
            title: Dashboard title
        """
        if include_sections is None:
            include_sections = ['importance', 'metrics', 'predictions', 'stats']
        
        # Prepare data
        importance_df = self.compute_feature_importance(method='both')
        metrics = self.get_metrics()
        stats = self.get_feature_statistics()
        
        # Get predictions for visualization
        X_eval = self.X_test if self.X_test is not None else self.X
        y_eval = self.y_test if self.y_test is not None else self.y
        y_pred = self.get_predictions(X_eval)
        
        predictions_data = {
            'actual': y_eval.tolist()[:100],  # Limit to 100 points for performance
            'predicted': y_pred.tolist()[:100]
        }
        
        # Prepare importance data
        importance_perm = importance_df[importance_df['method'] == 'permutation'].sort_values('importance', ascending=False).head(15)
        importance_data = importance_perm[['feature', 'importance']].to_dict('records')
        
        # Confusion matrix for classification
        confusion_matrix_data = None
        if self.model_type == 'classifier' and 'confusion_matrix' in include_sections:
            cm = self.get_confusion_matrix()
            if cm is not None:
                confusion_matrix_data = cm.tolist()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-card: #ffffff;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --accent-color: #0d6efd;
            --accent-hover: #0b5ed7;
            --success: #198754;
            --danger: #dc3545;
            --warning: #ffc107;
            --shadow: rgba(0, 0, 0, 0.1);
        }}
        
        body.dark-mode {{
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-card: #242424;
            --text-primary: #e9ecef;
            --text-secondary: #adb5bd;
            --border-color: #404040;
            --accent-color: #4dabf7;
            --accent-hover: #339af0;
            --success: #51cf66;
            --danger: #ff6b6b;
            --warning: #ffd43b;
            --shadow: rgba(0, 0, 0, 0.3);
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            transition: background 0.3s, color 0.3s;
            line-height: 1.6;
        }}
        
        .header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 4px var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            font-size: 1.75rem;
            font-weight: 600;
        }}
        
        .theme-toggle {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        
        .theme-toggle:hover {{
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px var(--shadow);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow);
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-color);
        }}
        
        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px var(--shadow);
        }}
        
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
        }}
        
        .axis text {{
            fill: var(--text-primary);
            font-size: 12px;
        }}
        
        .axis line, .axis path {{
            stroke: var(--border-color);
        }}
        
        .grid line {{
            stroke: var(--border-color);
            stroke-opacity: 0.3;
        }}
        
        .tooltip {{
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            box-shadow: 0 4px 12px var(--shadow);
            font-size: 0.85rem;
            z-index: 1000;
        }}
        
        .bar {{
            transition: opacity 0.2s;
        }}
        
        .bar:hover {{
            opacity: 0.8;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 1rem;
            }}
            
            h1 {{
                font-size: 1.25rem;
            }}
            
            .container {{
                padding: 1rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body class="{'dark-mode' if dark_mode else ''}">
    <div class="header">
        <div class="header-content">
            <h1>{title}</h1>
            <button class="theme-toggle" onclick="toggleTheme()">
                🌓 Toggle Theme
            </button>
        </div>
    </div>
    
    <div class="container">
        {''.join([self._generate_section(section, metrics, importance_data, predictions_data, confusion_matrix_data, stats) 
                  for section in include_sections])}
    </div>
    
    <div class="tooltip"></div>
    
    <script>
        const data = {{
            importance: {json.dumps(importance_data)},
            predictions: {json.dumps(predictions_data)},
            confusionMatrix: {json.dumps(confusion_matrix_data)},
            modelType: "{self.model_type}"
        }};
        
        function toggleTheme() {{
            document.body.classList.toggle('dark-mode');
        }}
        
        // Initialize charts
        {self._generate_chart_scripts(include_sections)}
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_section(self, section: str, metrics: Dict, importance_data: List, 
                          predictions_data: Dict, confusion_matrix_data: Optional[List], 
                          stats: pd.DataFrame) -> str:
        """Generate HTML for a specific section."""
        
        if section == 'metrics':
            metric_cards = []
            for key, value in metrics.items():
                metric_cards.append(f"""
                    <div class="metric-card">
                        <div class="metric-label">{key.upper()}</div>
                        <div class="metric-value">{value:.4f}</div>
                    </div>
                """)
            return f'<div class="metrics-grid">{"".join(metric_cards)}</div>'
        
        elif section == 'importance':
            return '''
                <div class="chart-container">
                    <h2 class="chart-title">Feature Importance</h2>
                    <div id="importance-chart"></div>
                </div>
            '''
        
        elif section == 'predictions':
            title = "Predictions vs Actual" if self.model_type == 'regressor' else "Prediction Distribution"
            return f'''
                <div class="chart-container">
                    <h2 class="chart-title">{title}</h2>
                    <div id="predictions-chart"></div>
                </div>
            '''
        
        elif section == 'confusion_matrix' and confusion_matrix_data:
            return '''
                <div class="chart-container">
                    <h2 class="chart-title">Confusion Matrix</h2>
                    <div id="confusion-matrix-chart"></div>
                </div>
            '''
        
        elif section == 'stats':
            stats_html = stats.head(10).to_html(classes='stats-table', float_format=lambda x: f'{x:.3f}')
            return f'''
                <div class="chart-container">
                    <h2 class="chart-title">Feature Statistics (Top 10)</h2>
                    <div style="overflow-x: auto;">{stats_html}</div>
                </div>
            '''
        
        return ''
    
    def _generate_chart_scripts(self, sections: List[str]) -> str:
        """Generate D3.js chart initialization scripts."""
        scripts = []
        
        if 'importance' in sections:
            scripts.append("""
                // Feature Importance Chart
                function createImportanceChart() {
                    const container = d3.select('#importance-chart');
                    const width = container.node().getBoundingClientRect().width;
                    const height = 400;
                    const margin = {top: 20, right: 30, bottom: 40, left: 150};
                    
                    const svg = container.append('svg')
                        .attr('width', width)
                        .attr('height', height);
                    
                    const chartWidth = width - margin.left - margin.right;
                    const chartHeight = height - margin.top - margin.bottom;
                    
                    const g = svg.append('g')
                        .attr('transform', `translate(${margin.left},${margin.top})`);
                    
                    const x = d3.scaleLinear()
                        .domain([0, d3.max(data.importance, d => d.importance)])
                        .range([0, chartWidth]);
                    
                    const y = d3.scaleBand()
                        .domain(data.importance.map(d => d.feature))
                        .range([0, chartHeight])
                        .padding(0.2);
                    
                    // Grid
                    g.append('g')
                        .attr('class', 'grid')
                        .call(d3.axisBottom(x).tickSize(chartHeight).tickFormat(''));
                    
                    // Bars with animation
                    g.selectAll('.bar')
                        .data(data.importance)
                        .enter()
                        .append('rect')
                        .attr('class', 'bar')
                        .attr('x', 0)
                        .attr('y', d => y(d.feature))
                        .attr('height', y.bandwidth())
                        .attr('fill', '#4dabf7')
                        .attr('width', 0)
                        .on('mouseover', function(event, d) {
                            const tooltip = d3.select('.tooltip');
                            tooltip.style('opacity', 1)
                                .html(`<strong>${d.feature}</strong><br/>Importance: ${d.importance.toFixed(4)}`);
                        })
                        .on('mousemove', function(event) {
                            d3.select('.tooltip')
                                .style('left', (event.pageX + 10) + 'px')
                                .style('top', (event.pageY - 10) + 'px');
                        })
                        .on('mouseout', function() {
                            d3.select('.tooltip').style('opacity', 0);
                        })
                        .transition()
                        .duration(1000)
                        .ease(d3.easeCubicOut)
                        .attr('width', d => x(d.importance));
                    
                    // Axes
                    g.append('g')
                        .attr('class', 'axis')
                        .attr('transform', `translate(0,${chartHeight})`)
                        .call(d3.axisBottom(x));
                    
                    g.append('g')
                        .attr('class', 'axis')
                        .call(d3.axisLeft(y));
                }
                createImportanceChart();
            """)
        
        if 'predictions' in sections:
            scripts.append("""
                // Predictions Chart
                function createPredictionsChart() {
                    const container = d3.select('#predictions-chart');
                    const width = container.node().getBoundingClientRect().width;
                    const height = 400;
                    const margin = {top: 20, right: 30, bottom: 50, left: 60};
                    
                    const svg = container.append('svg')
                        .attr('width', width)
                        .attr('height', height);
                    
                    const chartWidth = width - margin.left - margin.right;
                    const chartHeight = height - margin.top - margin.bottom;
                    
                    const g = svg.append('g')
                        .attr('transform', `translate(${margin.left},${margin.top})`);
                    
                    const dataPoints = data.predictions.actual.map((a, i) => ({
                        actual: a,
                        predicted: data.predictions.predicted[i],
                        index: i
                    }));
                    
                    const x = d3.scaleLinear()
                        .domain([0, dataPoints.length - 1])
                        .range([0, chartWidth]);
                    
                    const yExtent = d3.extent([...data.predictions.actual, ...data.predictions.predicted]);
                    const y = d3.scaleLinear()
                        .domain(yExtent)
                        .range([chartHeight, 0]);
                    
                    // Grid
                    g.append('g')
                        .attr('class', 'grid')
                        .call(d3.axisLeft(y).tickSize(-chartWidth).tickFormat(''));
                    
                    // Line for actual values
                    const actualLine = d3.line()
                        .x((d, i) => x(i))
                        .y(d => y(d));
                    
                    const actualPath = g.append('path')
                        .datum(data.predictions.actual)
                        .attr('fill', 'none')
                        .attr('stroke', '#51cf66')
                        .attr('stroke-width', 2)
                        .attr('d', actualLine);
                    
                    // Line for predicted values
                    const predPath = g.append('path')
                        .datum(data.predictions.predicted)
                        .attr('fill', 'none')
                        .attr('stroke', '#4dabf7')
                        .attr('stroke-width', 2)
                        .attr('d', actualLine);
                    
                    // Animate lines
                    const totalLength = actualPath.node().getTotalLength();
                    actualPath
                        .attr('stroke-dasharray', totalLength)
                        .attr('stroke-dashoffset', totalLength)
                        .transition()
                        .duration(1500)
                        .ease(d3.easeCubicInOut)
                        .attr('stroke-dashoffset', 0);
                    
                    predPath
                        .attr('stroke-dasharray', totalLength)
                        .attr('stroke-dashoffset', totalLength)
                        .transition()
                        .duration(1500)
                        .ease(d3.easeCubicInOut)
                        .attr('stroke-dashoffset', 0);
                    
                    // Axes
                    g.append('g')
                        .attr('class', 'axis')
                        .attr('transform', `translate(0,${chartHeight})`)
                        .call(d3.axisBottom(x));
                    
                    g.append('g')
                        .attr('class', 'axis')
                        .call(d3.axisLeft(y));
                    
                    // Legend
                    const legend = svg.append('g')
                        .attr('transform', `translate(${width - 150}, 20)`);
                    
                    legend.append('line')
                        .attr('x1', 0).attr('x2', 30)
                        .attr('y1', 0).attr('y2', 0)
                        .attr('stroke', '#51cf66')
                        .attr('stroke-width', 2);
                    
                    legend.append('text')
                        .attr('x', 35).attr('y', 5)
                        .style('fill', 'var(--text-primary)')
                        .text('Actual');
                    
                    legend.append('line')
                        .attr('x1', 0).attr('x2', 30)
                        .attr('y1', 20).attr('y2', 20)
                        .attr('stroke', '#4dabf7')
                        .attr('stroke-width', 2);
                    
                    legend.append('text')
                        .attr('x', 35).attr('y', 25)
                        .style('fill', 'var(--text-primary)')
                        .text('Predicted');
                }
                createPredictionsChart();
            """)
        
        return '\n'.join(scripts)
    
    def save_html(self, filepath: str, **kwargs):
        """Save dashboard to HTML file."""
        html = self.to_html(**kwargs)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Dashboard saved to {filepath}")
