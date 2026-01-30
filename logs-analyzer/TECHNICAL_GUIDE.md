# Technical Guide: Logs Analyzer

The Logs Analyzer is a sophisticated Streamlit-based web application designed for comprehensive log analysis across multiple formats. It provides interactive dashboards with advanced analytics including clusterization, temporal analysis, and multi-format log processing capabilities.

## Technical Overview

The analyzer is built as a **Multi-Module Web Application** with specialized analyzers for different log formats. Each module provides tailored analysis capabilities while maintaining a consistent user interface and advanced machine learning features.

### 1. Architecture

The application follows a **Modular Streamlit Architecture** with the following components:

- **Main Application** (`main.py`): Navigation hub with sidebar-based module selection
- **TXT Analyzer** (`logsanalyser-txt.py`): Advanced text log analysis with ML clustering
- **JSON Analyzer** (`logsanalyser-json.py`): Sentry event analysis with severity scoring
- **CSV Analyzer** (`logsanalyser.py`): Structured data analysis with automatic field extraction
- **Style Module** (`style.py`): Consistent UI theming and styling

### 2. Core Technologies

- **Streamlit**: Interactive web framework for data applications
- **Plotly**: Advanced interactive visualizations
- **Scikit-learn**: Machine learning for clustering and analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **JSON/CSV Processing**: Multi-format data handling

## Module Capabilities

### TXT Log Analyzer (`logsanalyser-txt.py`)

**Purpose**: Advanced analysis of unstructured text logs with intelligent error detection.

**Key Features**:
- **Multi-level Error Detection**: Automatic classification (ERROR, WARNING, INFO, DEBUG)
- **Smart Parsing**: Regex-based log pattern recognition with timestamp extraction
- **Advanced Clusterization**: KMeans, DBSCAN, and LDA topic modeling
- **Temporal Analysis**: Hourly/daily event distribution and trend analysis
- **Interactive Filtering**: Real-time log exploration with search and pagination

**Technical Implementation**:
```python
# Error level classification
level = "INFO"
message_upper = message.upper()
if any(word in message_upper for word in ["ERROR", "EXCEPTION", "FAILED", "CRITICAL", "FATAL"]):
    level = "ERROR"
elif any(word in message_upper for word in ["WARNING", "WARN", "DEPRECATED"]):
    level = "WARNING"
```

### JSON Event Analyzer (`logsanalyser-json.py`)

**Purpose**: Specialized analysis of Sentry JSON events with context-aware severity scoring.

**Key Features**:
- **Severity Scoring**: Multi-factor severity calculation (error type, breadcrumbs, user interactions)
- **Context Analysis**: Device, OS, platform, and user journey analysis
- **Breadcrumb Tracking**: User interaction path reconstruction
- **Component Mapping**: Error correlation with application components and screens

**Technical Implementation**:
```python
def calculate_severity_score(data, exception_info, breadcrumbs):
    """Calculates severity score based on multiple factors"""
    score = 0
    # Factor 1: Exception type analysis
    error_type = exception_info.get('type', '').lower()
    critical_errors = ['nullpointer', 'outofmemory', 'stackoverflow', 'fatal', 'crash']
    if any(critical in error_type for critical in critical_errors):
        score += 3
    # Factor 2: Breadcrumb analysis
    score += min(len(breadcrumbs) // 5, 3)
    # Factor 3: User interaction detection
    user_interactions = sum(1 for bc in breadcrumbs if 'click' in str(bc.get('message', '')).lower())
    score += min(user_interactions, 2)
    return score
```

### CSV Data Analyzer (`logsanalyser.py`)

**Purpose**: Structured data analysis with automatic JSON field extraction and advanced clustering.

**Key Features**:
- **Automatic Field Discovery**: Recursive JSON structure analysis and field extraction
- **Smart CSV Processing**: Robust CSV parsing with multiple fallback methods
- **Persona Analysis**: User behavior clustering and profiling
- **Advanced Clusterization**: Optimal cluster number detection with elbow method
- **Statistical Analysis**: Correlation matrices, distribution analysis, and hypothesis testing

**Technical Implementation**:
```python
def extract_all_fields(data, parent_key='', separator='.'):
    """Recursively extracts all fields from JSON data"""
    fields = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                fields.extend(extract_all_fields(value, new_key, separator))
            else:
                fields.append(new_key)
    return list(set(fields))
```

## Advanced Analytics

### 1. Machine Learning Clusterization

**KMeans Clustering**:
- **Optimal Cluster Detection**: Elbow method with WCSS (Within-Cluster Sum of Squares)
- **Silhouette Analysis**: Cluster quality assessment
- **Feature Engineering**: Automatic categorical encoding and normalization

**DBSCAN Clustering**:
- **Density-based Clustering**: Automatic outlier detection
- **Parameter Optimization**: Configurable eps and min_samples parameters
- **Noise Handling**: Explicit outlier identification

**LDA Topic Modeling**:
- **Text-based Clustering**: Topic discovery in log messages
- **Perplexity Optimization**: Automatic topic number detection
- **Word Importance**: Feature extraction and visualization

### 2. Temporal Analysis

**Time-based Patterns**:
- **Hourly Distribution**: Peak usage and error time identification
- **Daily Trends**: Weekly patterns and anomaly detection
- **Event Correlation**: Temporal relationship analysis

**Advanced Metrics**:
```python
# Temporal analysis implementation
temporal_analysis = {}
hourly_dist = df.groupby('hour_of_day').size()
temporal_analysis['peak_hour'] = hourly_dist.idxmax()
temporal_analysis['peak_hour_count'] = hourly_dist.max()
```

### 3. Interactive Visualizations

**Plotly Integration**:
- **Real-time Charts**: Interactive scatter plots, bar charts, and pie charts
- **Multi-dimensional Views**: 2D projections with PCA and t-SNE
- **Drill-down Capabilities**: Click-to-explore functionality

**Dashboard Features**:
- **Live Filtering**: Real-time data filtering and updates
- **Export Capabilities**: CSV export with cluster labels
- **Responsive Design**: Mobile and desktop optimized

## Operational Workflow

### Application Launch

Start the Streamlit application:

```bash
streamlit run main.py
```

Access the application at: `http://localhost:8501`

### Analysis Process

1. **Module Selection**: Choose analysis type from sidebar navigation
2. **File Upload**: Upload log files in supported formats
3. **Automatic Processing**: Smart parsing and field extraction
4. **Interactive Analysis**: Explore data with filters and visualizations
5. **Advanced Analytics**: Apply clustering and statistical analysis
6. **Export Results**: Download analyzed data with insights

### Data Processing Pipeline

1. **File Upload**: Accepts TXT, JSON, and CSV formats
2. **Format Detection**: Automatic format identification and processing
3. **Data Extraction**: Field extraction and normalization
4. **Quality Assessment**: Data validation and cleaning
5. **Analysis Execution**: Apply selected analytical methods
6. **Result Presentation**: Interactive dashboard with insights

## Understanding the Output

### Dashboard Metrics

- **Event Counts**: Total events, filtered events, critical events
- **Severity Distribution**: Error level breakdown with percentages
- **Temporal Patterns**: Time-based event distribution
- **Cluster Analysis**: Group characteristics and member counts

### Analysis Results

- **Cluster Profiles**: Detailed cluster characteristics and statistics
- **Topic Analysis**: LDA topic discovery with word importance
- **Correlation Insights**: Statistical relationships between variables
- **Anomaly Detection**: Outlier identification and analysis

### Export Capabilities

- **CSV Export**: Complete dataset with cluster labels
- **JSON Export**: Structured analysis results
- **Visualization Export**: Chart images and interactive plots

## Usage Best Practices

### 1. Data Preparation

- **File Format**: Ensure logs are in supported formats (TXT, JSON, CSV)
- **Encoding**: Use UTF-8 encoding for text files
- **Structure**: Maintain consistent JSON structure for optimal field extraction

### 2. Analysis Strategy

- **Start Simple**: Begin with basic filtering before advanced analytics
- **Iterative Approach**: Apply filters incrementally for focused analysis
- **Cross-validation**: Use multiple clustering methods for robust results

### 3. Performance Optimization

- **Large Files**: Use pagination and filtering for large datasets
- **Memory Management**: Process data in chunks for optimal performance
- **Visualization**: Limit data points in visualizations for responsiveness

### 4. Interpretation Guidelines

- **Context Awareness**: Consider application context when interpreting clusters
- **Statistical Significance**: Validate findings with statistical tests
- **Business Impact**: Focus on actionable insights and business value

## Technical Requirements

### System Requirements

- **Python**: 3.12 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: Sufficient space for uploaded files and temporary processing
- **Browser**: Modern browser with JavaScript support for Streamlit interface

### Deployment Options

- **Local Development**: Direct Streamlit execution
- **Docker**: Containerized deployment with docker-compose
- **Cloud**: Deployable to any Streamlit-compatible platform

This comprehensive technical guide provides the foundation for understanding, using, and extending the Logs Analyzer application for effective log analysis and insights discovery.
