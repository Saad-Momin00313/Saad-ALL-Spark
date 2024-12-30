from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
from datetime import datetime, timedelta
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import plotly.utils
import os
from dotenv import load_dotenv
import uuid
import logging
import numpy as np
from fastapi.openapi.utils import get_openapi
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

if GEMINI_API_KEY == "your_api_key_here":
    raise ValueError("Please replace the placeholder with your actual Gemini API key")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

app = FastAPI(
    title="Data Visualization API",
    description="API service for intelligent data visualization using Gemini AI",
    version="1.0.0"
)

# Rate limiting configuration
RATE_LIMIT_DURATION = timedelta(minutes=1)
MAX_REQUESTS_PER_MINUTE = 60
rate_limit_store = {}

class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass

async def check_rate_limit(request: Request):
    """Check if the request exceeds rate limits."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    # Clean up old entries
    rate_limit_store.clear()
    
    # Get or create client's request history
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    
    # Remove old requests
    client_requests = rate_limit_store[client_ip]
    client_requests = [t for t in client_requests if current_time - t < RATE_LIMIT_DURATION]
    
    # Check rate limit
    if len(client_requests) >= MAX_REQUESTS_PER_MINUTE:
        raise RateLimitError("Rate limit exceeded. Please try again later.")
    
    # Add current request
    client_requests.append(current_time)
    rate_limit_store[client_ip] = client_requests

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware to handle rate limiting."""
    try:
        await check_rate_limit(request)
        response = await call_next(request)
        return response
    except RateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"detail": str(e)}
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Update CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# In-memory storage for data and suggestions
data_store = {}

# Visualization prompt configuration
VISUALIZATION_PROMPTS = {
    "chart_types": {
        "bar": "comparing categories with exact values and proportions",
        "pie": "showing percentage breakdowns with precise percentages",
        "histogram": "showing value distributions with mean, median, and mode",
        "scatter": "showing relationships with correlation coefficients",
        "box": "showing value ranges with quartiles and specific outliers",
        "line": "showing trends with growth rates and key changes",
        "heatmap": "showing correlation matrices with specific correlation values",
        "violin": "showing distribution shapes with exact statistics",
        "sunburst": "showing hierarchical relationships with proportions",
        "treemap": "showing nested categories with size comparisons"
    },
    "insight_requirements": [
        "Describe the exact patterns shown in the visualization",
        "Explain what these patterns mean in the context of the data",
        "Highlight key findings and notable outliers",
        "Suggest potential business implications or actions",
        "Compare values and identify trends",
        "Explain relationships between variables (if applicable)",
        "Point out any unusual distributions or clusters",
        "Quantify the insights with specific numbers and percentages"
    ],
    "histogram_requirements": [
        "Only suggest histograms for meaningful numeric columns (e.g., age, salary, balance)",
        "Avoid using index or ID columns like 'RowNumber'",
        "Specify appropriate bin sizes based on the data range",
        "Include insights about the distribution shape, central tendency, and outliers"
    ],
    "insight_components": [
        "Distribution characteristics (mean, median, mode, skewness)",
        "Key percentiles and ranges",
        "Notable patterns or clusters",
        "Business implications of the distribution",
        "Comparison to expected patterns",
        "Actionable recommendations based on the distribution"
    ],
    "actionable_insights": [
        "Using exact numbers and percentages from the data",
        "Identifying specific patterns with supporting values",
        "Highlighting outliers with their exact values",
        "Describing relationships with correlation coefficients",
        "Comparing categories with precise differences",
        "Suggesting specific actions based on the findings",
        "Explaining the business impact of each insight",
        "Providing context for why each pattern matters"
    ]
}

def build_visualization_prompt(data_summary: Dict[str, Any], prompts: Dict[str, Any] = VISUALIZATION_PROMPTS) -> str:
    """Build the visualization prompt dynamically."""
    prompt_parts = [
        "Analyze this dataset and suggest meaningful visualizations that reveal clear insights.",
        f"Dataset Summary:\n{json.dumps(data_summary, indent=2)}\n",
        
        "For histograms:",
        *[f"- {req}" for req in prompts["histogram_requirements"]],
        
        "\nProvide a list of visualizations in the following JSON format. For each visualization, provide DETAILED insights that:",
        *[f"{i+1}. {req}" for i, req in enumerate(prompts["insight_requirements"])],
        
        "\nFormat:",
        "[\n    {",
        '        "id": "chart_1",',
        '        "type": "chart_type",',
        '        "title": "Clear, descriptive title that includes key metrics or variables",',
        '        "description": "Detailed insights that include:',
        *[f"            - {comp}" for comp in prompts["insight_components"]],
        '        ",',
        '        "x_axis": "column_name",  # For histograms, choose meaningful numeric columns',
        '        "y_axis": "column_name",  # Optional for histograms',
        '        "additional_params": {',
        '            "nbins": "auto",  # Let the visualization function determine optimal bins',
        '            "color": "column_for_grouping",',
        '            "trendline": "ols",  # for scatter plots',
        '            "points": "outliers"  # for box plots',
        '        }',
        '    }\n]',
        
        "\nUse these chart types:",
        *[f"{i+1}. {chart_type} - for {desc}" for i, (chart_type, desc) in enumerate(prompts["chart_types"].items())],
        
        "\nMake the insights extremely specific and actionable by:",
        *[f"{i+1}. {insight}" for i, insight in enumerate(prompts["actionable_insights"])],
        
        "\nEnsure all column names are exactly as they appear in the dataset.",
        "Focus on generating insights that are specific to this dataset rather than generic observations."
    ]
    
    return "\n".join(prompt_parts)

class ChartSuggestion(BaseModel):
    id: str = Field(..., description="Unique identifier for the chart suggestion")
    type: str = Field(..., description="Type of visualization (e.g., scatter, bar, box)")
    title: str = Field(..., description="Descriptive title for the visualization")
    description: str = Field(..., description="Detailed explanation of the visualization and its insights")
    x_axis: Optional[str] = Field(None, description="Column name for x-axis")
    y_axis: Optional[str] = Field(None, description="Column name for y-axis")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the visualization")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chart_1",
                "type": "scatter",
                "title": "Revenue vs Time",
                "description": "Scatter plot showing revenue trends over time",
                "x_axis": "date",
                "y_axis": "revenue",
                "additional_params": {
                    "color": "category",
                    "trendline": "ols"
                }
            }
        }

class ChartSuggestionResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    suggestions: List[ChartSuggestion] = Field(..., description="List of visualization suggestions")

class VisualizationRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier from upload response")
    chart_id: str = Field(..., description="Chart identifier from suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "chart_id": "chart_1"
            }
        }

class VisualizationResponse(BaseModel):
    chart_id: str = Field(..., description="Chart identifier")
    type: str = Field(..., description="Type of visualization")
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Chart description")
    plot_data: Dict[str, Any] = Field(..., description="Plotly figure data")

class DataSummary(BaseModel):
    filename: str
    upload_time: str
    row_count: int
    column_count: int
    columns: Dict[str, str]  # column name -> data type
    missing_values: Dict[str, int]
    unique_values: Dict[str, int]
    numeric_stats: Optional[Dict[str, Dict[str, float]]]
    categorical_stats: Optional[Dict[str, Dict[str, Any]]]
    correlations: Optional[Dict[str, Dict[str, float]]]

class SessionManager:
    def __init__(self, max_sessions: int = 100, session_timeout_minutes: int = 60):
        self.max_sessions = max_sessions
        self.session_timeout = pd.Timedelta(minutes=session_timeout_minutes)
        self.sessions = {}
        self.last_cleanup = pd.Timestamp.now()
        self.cleanup_interval = pd.Timedelta(minutes=5)
    
    def add_session(self, session_id: str, data: Dict[str, Any]):
        """Add a new session."""
        self._cleanup_if_needed()
        
        # Remove oldest session if limit reached
        if len(self.sessions) >= self.max_sessions:
            oldest_session = min(self.sessions.items(), key=lambda x: x[1]["last_accessed"])
            del self.sessions[oldest_session[0]]
        
        self.sessions[session_id] = {
            "data": data,
            "created": pd.Timestamp.now(),
            "last_accessed": pd.Timestamp.now()
        }
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data and update last accessed time."""
        self._cleanup_if_needed()
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if pd.Timestamp.now() - session["last_accessed"] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        session["last_accessed"] = pd.Timestamp.now()
        return session["data"]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def _cleanup_if_needed(self):
        """Clean up expired sessions if cleanup interval has passed."""
        now = pd.Timestamp.now()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if now - session["last_accessed"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]

# Initialize session manager
session_manager = SessionManager()

def get_chart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get chart suggestions from Gemini AI based on data analysis."""
    try:
        # Get column information
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get meaningful columns for visualizations
        id_patterns = ['id', 'index', 'row', 'number', 'customerid', 'userid']
        meaningful_numeric_columns = [
            col for col in numeric_columns 
            if not any(pattern in col.lower() for pattern in id_patterns)
        ]
        
        meaningful_categorical_columns = [
            col for col in categorical_columns 
            if not any(pattern in col.lower() for pattern in id_patterns)
            and df[col].nunique() < len(df) * 0.5  # Exclude columns with too many unique values
        ]
        
        # Define default suggestions with meaningful columns
        default_suggestions = [
            {
                "id": "chart_1",
                "type": "bar",
                "title": f"Distribution of {meaningful_categorical_columns[0].replace('_', ' ').title()}" if meaningful_categorical_columns else "Category Distribution",
                "description": f"Shows the distribution of {meaningful_categorical_columns[0].replace('_', ' ').title() if meaningful_categorical_columns else 'categories'} across the dataset. Each bar represents the count of items in each category.",
                "x_axis": meaningful_categorical_columns[0] if meaningful_categorical_columns else categorical_columns[0],
                "y_axis": None,
                "additional_params": {
                    "barmode": "group"
                }
            },
            {
                "id": "chart_2",
                "type": "histogram",
                "title": f"Distribution of {meaningful_numeric_columns[0].replace('_', ' ').title()}" if meaningful_numeric_columns else "Numeric Distribution",
                "description": f"Shows the distribution of {meaningful_numeric_columns[0].replace('_', ' ').title() if meaningful_numeric_columns else 'values'} with mean and median lines, revealing the central tendency and spread of the data.",
                "x_axis": meaningful_numeric_columns[0] if meaningful_numeric_columns else numeric_columns[0],
                "y_axis": None,
                "additional_params": {
                    "nbins": "auto"
                }
            },
            {
                "id": "chart_3",
                "type": "pie",
                "title": f"Percentage Breakdown of {meaningful_categorical_columns[0].replace('_', ' ').title()}" if meaningful_categorical_columns else "Category Breakdown",
                "description": f"Shows the percentage distribution of {meaningful_categorical_columns[0].replace('_', ' ').title() if meaningful_categorical_columns else 'categories'}. Each slice represents a category's proportion of the total.",
                "x_axis": meaningful_categorical_columns[0] if meaningful_categorical_columns else categorical_columns[0],
                "y_axis": None,
                "additional_params": {}
            }
        ]
        
        # Calculate basic statistics
        stats = df.describe()
        correlations = df[numeric_columns].corr() if numeric_columns else pd.DataFrame()
        
        # Get unique value counts for categorical columns
        categorical_stats = {
            col: {
                'value_counts': df[col].value_counts().to_dict(),
                'unique_count': df[col].nunique()
            }
            for col in categorical_columns
        }
        
        # Prepare data summary for Gemini
        data_summary = {
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "sample_rows": df.head().to_dict(),
            "statistics": stats.to_dict(),
            "categorical_stats": categorical_stats,
            "strong_correlations": [
                {"col1": col1, "col2": col2, "correlation": correlations.loc[col1, col2]}
                for col1 in correlations.columns
                for col2 in correlations.columns
                if col1 < col2 and abs(correlations.loc[col1, col2]) > 0.5
            ] if not correlations.empty else []
        }
        
        try:
            # Try to get suggestions from Gemini AI
            prompt = build_visualization_prompt(data_summary)
            
            logger.info("Requesting chart suggestions from Gemini AI")
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            # Clean up the response text
            response_text = response_text.strip()
            if not response_text:
                raise ValueError("Empty response from AI")
            
            suggestions = json.loads(response_text)
            if not suggestions or not isinstance(suggestions, list):
                raise ValueError("Invalid suggestions format")
            
            logger.info(f"Received {len(suggestions)} chart suggestions")
            
            # Validate suggestions
            validated_suggestions = []
            for i, suggestion in enumerate(suggestions):
                try:
                    # Add ID if missing
                    suggestion["id"] = suggestion.get("id", f"chart_{i+1}")
                    
                    # Validate required fields
                    if not all(k in suggestion for k in ["type", "title", "description"]):
                        continue
                    
                    # Normalize chart type
                    suggestion["type"] = normalize_chart_type(suggestion["type"])
                    
                    # Validate columns exist
                    if suggestion.get("x_axis") and suggestion["x_axis"] not in df.columns:
                        continue
                    if suggestion.get("y_axis") and suggestion["y_axis"] not in df.columns:
                        continue
                    
                    # Validate additional parameters
                    if "additional_params" in suggestion:
                        params = suggestion["additional_params"]
                        for key in ["color", "size", "facet_col", "facet_row", "animation_frame"]:
                            if params.get(key) and params[key] not in df.columns:
                                params.pop(key)
                        
                        # Handle values parameter for heatmaps
                        if params.get("values"):
                            if isinstance(params["values"], list):
                                params["values"] = [col for col in params["values"] if col in df.columns]
                            elif isinstance(params["values"], str) and params["values"] not in df.columns:
                                params.pop("values")
                    
                    validated_suggestions.append(suggestion)
                except Exception as e:
                    logger.warning(f"Error validating suggestion {i}: {str(e)}")
                    continue
            
            if validated_suggestions:
                return validated_suggestions
            else:
                logger.warning("No valid suggestions from AI, using defaults")
                return default_suggestions
            
        except Exception as e:
            logger.warning(f"Error getting AI suggestions: {str(e)}, using defaults")
            return default_suggestions
        
    except Exception as e:
        logger.error(f"Error in get_chart_suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def normalize_chart_type(chart_type: str) -> str:
    """Normalize chart type names to standard format."""
    type_mapping = {
        'scatter_plot': 'scatter',
        'violin_plot': 'violin',
        'box_plot': 'box',
        'scatter_matrix': 'scatter_matrix',
        'density_heatmap': 'density_heatmap',
        'heatmap': 'heatmap'
    }
    return type_mapping.get(chart_type.lower().strip(), chart_type.lower().strip())

def generate_visualization(df: pd.DataFrame, chart_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a specific visualization based on configuration."""
    try:
        import plotly.graph_objects as go
        
        chart_type = normalize_chart_type(chart_config["type"])
        title = chart_config["title"]
        additional_params = chart_config.get("additional_params", {})
        
        if chart_type == "heatmap":
            # Handle correlation heatmap
            if additional_params.get("values"):
                # Use specified columns
                columns = additional_params["values"]
                if isinstance(columns, str):
                    columns = [columns]
                numeric_df = df[columns]
            else:
                # Use all numeric columns
                numeric_df = df.select_dtypes(include=['int64', 'float64'])
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Features",
                yaxis_title="Features",
                width=800,
                height=800
            )
        
        # Handle combined chart types
        elif "+" in chart_type:
            chart_types = [t.strip() for t in chart_type.split("+")]
            
            if set(chart_types) == {"violin", "box"}:
                # Create combined violin and box plot
                y_axis = additional_params.get("y_axis") or chart_config.get("y_axis")
                x_axis = additional_params.get("x_axis") or chart_config.get("x_axis")
                
                fig = go.Figure()
                
                # Add violin plot
                fig.add_trace(go.Violin(
                    x=df[x_axis] if x_axis else None,
                    y=df[y_axis],
                    name="Distribution",
                    side="positive",
                    meanline={"visible": True}
                ))
                
                # Add box plot
                fig.add_trace(go.Box(
                    x=df[x_axis] if x_axis else None,
                    y=df[y_axis],
                    name="Box Plot",
                    boxpoints="outliers"
                ))
                
                # Handle faceting if requested
                if additional_params.get("facet_row"):
                    facet_data = []
                    for facet_val in df[additional_params["facet_row"]].unique():
                        subset = df[df[additional_params["facet_row"]] == facet_val]
                        facet_data.append({
                            "type": "violin",
                            "y": subset[y_axis],
                            "name": f"{facet_val} - Distribution",
                            "legendgroup": str(facet_val),
                            "scalegroup": str(facet_val),
                            "side": "positive",
                            "meanline": {"visible": True}
                        })
                        facet_data.append({
                            "type": "box",
                            "y": subset[y_axis],
                            "name": f"{facet_val} - Box",
                            "legendgroup": str(facet_val),
                            "boxpoints": "outliers"
                        })
                    
                    fig = go.Figure(data=facet_data)
                
                fig.update_layout(
                    violingap=0.1,
                    violinmode='overlay',
                    title=title,
                    xaxis_title=x_axis if x_axis else "",
                    yaxis_title=y_axis if y_axis else ""
                )
            
            else:
                raise ValueError(f"Unsupported combined chart type: {chart_type}")
        
        elif chart_type in ["sunburst", "treemap"]:
            # Create path for hierarchical charts
            path = []
            if additional_params.get("facet_col"):
                path.append(additional_params["facet_col"])
            if additional_params.get("facet_row"):
                path.append(additional_params["facet_row"])
            if additional_params.get("color"):
                path.append(additional_params["color"])
            
            # If no path is specified, use categorical columns
            if not path:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) >= 2:
                    path = categorical_cols[:2].tolist()
                elif len(categorical_cols) == 1:
                    path = [categorical_cols[0]]
                else:
                    raise ValueError(f"No suitable categorical columns found for {chart_type} chart")
            
            # Add count column
            df_copy = df.copy()
            df_copy['Count'] = 1
            
            # If a value column is specified, use it instead of Count
            values = additional_params.get("values", "Count")
            if values != "Count" and values in df.columns:
                values_col = values
            else:
                values_col = "Count"
            
            if chart_type == "sunburst":
                fig = px.sunburst(
                    df_copy,
                    path=path,
                    values=values_col,
                    title=title,
                    color=additional_params.get("color")
                )
                fig.update_traces(
                    textinfo="label+value+percent parent",
                    insidetextorientation="radial"
                )
            else:  # treemap
                fig = px.treemap(
                    df_copy,
                    path=path,
                    values=values_col,
                    title=title,
                    color=additional_params.get("color")
                )
                fig.update_traces(
                    textinfo="label+value+percent parent"
                )
        
        else:
            # Handle other chart types using existing code...
            return generate_basic_visualization(df, chart_config, additional_params)
        
        # Add common layout improvements
        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.95),
            margin=dict(t=80, l=50, r=50, b=50),
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise

def generate_basic_visualization(df: pd.DataFrame, chart_config: Dict[str, Any], additional_params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate basic visualizations using plotly express."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    title = chart_config["title"]
    chart_type = normalize_chart_type(chart_config["type"])
    
    # Common parameters for regular charts
    common_params = {
        "title": title,
        "color": additional_params.get("color"),
        "custom_data": additional_params.get("custom_data"),
        "hover_data": additional_params.get("hover_data")
    }
    
    if chart_type == "pie":
        # Handle pie charts
        x_axis = chart_config.get("x_axis")
        if not x_axis:
            raise ValueError("x_axis is required for pie chart")
        
        try:
            # Calculate value counts for the categorical column
            counts = df[x_axis].value_counts()
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=counts.index.astype(str),
                    values=counts.values,
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                )
            ])
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    y=0.95
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update traces for better appearance
            fig.update_traces(
                rotation=90,
                textposition='inside',
                insidetextorientation='radial'
            )
            
            return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            raise ValueError(f"Could not create pie chart: {str(e)}")
    
    elif chart_type == "histogram":
        # Handle histogram
        x_axis = chart_config.get("x_axis") or additional_params.get("x_axis")
        if not x_axis:
            raise ValueError("x_axis is required for histogram")
        
        # Convert to numeric if needed
        try:
            values = pd.to_numeric(df[x_axis])
        except:
            values = df[x_axis]
        
        # Calculate optimal number of bins if not specified
        if additional_params.get("nbins") == "auto":
            # Use Freedman-Diaconis rule for bin size
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            bin_size = 2 * iqr / (len(values) ** (1/3))
            if bin_size > 0:
                nbins = int((values.max() - values.min()) / bin_size)
                nbins = min(max(nbins, 10), 50)  # Keep bins between 10 and 50
            else:
                nbins = 30
        else:
            nbins = additional_params.get("nbins", 30)
        
        # Create histogram using graph_objects for more control
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=nbins,
            name=x_axis.replace('_', ' ').title(),
            showlegend=True
        ))
        
        # Calculate statistics
        mean_value = values.mean()
        median_value = values.median()
        std_dev = values.std()
        
        # Update layout with more detailed title
        fig.update_layout(
            title=f"Distribution of {x_axis.replace('_', ' ').title()}",
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title="Count",
            bargap=0.1
        )
        
        # Add mean and median lines with annotations
        fig.add_vline(
            x=mean_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_value:.2f}",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=median_value,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {median_value:.2f}",
            annotation_position="bottom"
        )
        
        # Add standard deviation range
        fig.add_vrect(
            x0=mean_value - std_dev,
            x1=mean_value + std_dev,
            fillcolor="rgba(0,100,80,0.1)",
            layer="below",
            line_width=0,
            annotation_text="68% of data",
            annotation_position="top right"
        )
    
    elif chart_type == "bar":
        # Handle bar charts
        x_axis = chart_config.get("x_axis")
        y_axis = chart_config.get("y_axis")
        
        if not x_axis:
            raise ValueError("x_axis is required for bar chart")
        
        # Handle count-based bar charts
        if not y_axis:
            if additional_params.get("color"):
                # Create grouped/stacked bar chart with counts
                df_count = df.groupby([x_axis, additional_params["color"]]).size().reset_index(name='Count')
                fig = px.bar(
                    df_count,
                    x=x_axis,
                    y='Count',
                    color=additional_params["color"],
                    barmode=additional_params.get("barmode", "group"),
                    title=title,
                    text='Count'
                )
            else:
                # Create simple count-based bar chart
                counts = df[x_axis].value_counts().reset_index()
                counts.columns = [x_axis, 'Count']
                fig = px.bar(
                    counts,
                    x=x_axis,
                    y='Count',
                    title=title,
                    text='Count'
                )
        else:
            # Create regular bar chart with specified y-axis
            params = {
                "color": additional_params.get("color"),
                "barmode": additional_params.get("barmode", "group")
            }
            fig = px.bar(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                text=y_axis,
                **params
            )
        
        # Update text position
        fig.update_traces(textposition='auto')
    
    elif chart_type == "line":
        # Handle line charts
        x_axis = chart_config.get("x_axis")
        y_axis = chart_config.get("y_axis")
        
        if not x_axis or not y_axis:
            raise ValueError("Both x_axis and y_axis are required for line chart")
        
        # Sort data by x-axis for better line connection
        df_sorted = df.sort_values(by=x_axis)
        
        # If there are too many points, resample the data
        if len(df_sorted) > 50:
            # Group by x-axis and calculate mean of y-axis
            df_grouped = df_sorted.groupby(x_axis)[y_axis].mean().reset_index()
            df_plot = df_grouped
        else:
            df_plot = df_sorted
        
        # Create line chart
        fig = px.line(
            df_plot,
            x=x_axis,
            y=y_axis,
            title=title
        )
        
        # Update layout for better readability
        fig.update_traces(
            mode='lines+markers',  # Show both lines and markers
            marker=dict(
                size=8,  # Larger markers
                symbol='circle'
            ),
            line=dict(
                width=2  # Thicker line
            )
        )
        
        # Improve axis labels
        fig.update_layout(
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title=y_axis.replace('_', ' ').title(),
            showlegend=True
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate=f"{x_axis}: %{{x}}<br>{y_axis}: %{{y:.2f}}<extra></extra>"
        )
        
        # Add gridlines for better readability
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            )
        )
    
    elif chart_type == "scatter":
        # Handle scatter plots
        x_axis = chart_config.get("x_axis")
        y_axis = chart_config.get("y_axis")
        
        if not x_axis or not y_axis:
            raise ValueError("Both x_axis and y_axis are required for scatter plot")
        
        # Create a copy of the dataframe for modifications
        df_plot = df.copy()
        
        # Handle text columns by calculating length
        if df[x_axis].dtype == 'object':
            df_plot[f'{x_axis}_length'] = df[x_axis].str.len()
            x_axis = f'{x_axis}_length'
            # Update axis title
            x_axis_title = "Text Length (characters)"
        else:
            x_axis_title = x_axis.replace('_', ' ').title()
        
        if df[y_axis].dtype == 'object':
            df_plot[f'{y_axis}_length'] = df[y_axis].str.len()
            y_axis = f'{y_axis}_length'
            # Update axis title
            y_axis_title = "Text Length (characters)"
        else:
            y_axis_title = y_axis.replace('_', ' ').title()
        
        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x=x_axis,
            y=y_axis,
            title=title,
            trendline=additional_params.get("trendline"),
            labels={
                x_axis: x_axis_title,
                y_axis: y_axis_title
            }
        )
        
        # Add hover template with original text if available
        if x_axis.endswith('_length'):
            original_col = x_axis[:-7]  # Remove '_length'
            hover_template = f"{x_axis_title}: %{{x}}<br>{y_axis_title}: %{{y}}<br>Text: %{{customdata}}"
            fig.update_traces(
                customdata=df[original_col],
                hovertemplate=hover_template
            )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            showlegend=True
        )
        
        # Add gridlines
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            )
        )
    
    elif chart_type == "box":
        # Handle box plots
        x_axis = chart_config.get("x_axis")
        y_axis = chart_config.get("y_axis")
        
        # If only one variable is provided, use it as y-axis
        if x_axis and not y_axis:
            y_axis = x_axis
            x_axis = None
        elif not y_axis and not x_axis:
            raise ValueError("At least one axis (x_axis or y_axis) is required for box plot")
        
        # Create box plot
        fig = px.box(
            df,
            x=x_axis,
            y=y_axis,
            title=title,
            points=additional_params.get("points", "outliers")
        )
        
        # Add mean line
        if pd.api.types.is_numeric_dtype(df[y_axis]):
            mean_value = df[y_axis].mean()
            fig.add_hline(
                y=mean_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_value:.2f}",
                annotation_position="right"
            )
    
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Add common layout improvements
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.95),
        margin=dict(t=80, l=50, r=50, b=50),
        template="plotly_white",
        height=600
    )
    
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the uploaded data."""
    try:
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        # For numeric columns, fill missing values with median
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill missing values with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Convert date columns to datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the uploaded data."""
    try:
        # Check if dataframe is empty
        if df.empty:
            return False, "The uploaded file is empty"
        
        # Check minimum number of rows
        if len(df) < 2:
            return False, "The dataset must contain at least 2 rows"
        
        # Check minimum number of columns
        if len(df.columns) < 2:
            return False, "The dataset must contain at least 2 columns"
        
        # Check for all-null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            return False, f"The following columns contain only null values: {', '.join(null_columns)}"
        
        # Check for constant columns
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            return False, f"The following columns contain only one unique value: {', '.join(constant_columns)}"
        
        # Check for excessive missing values (e.g., more than 50%)
        missing_ratios = df.isnull().mean()
        problematic_columns = missing_ratios[missing_ratios > 0.5].index.tolist()
        if problematic_columns:
            return False, f"The following columns have more than 50% missing values: {', '.join(problematic_columns)}"
        
        return True, "Data validation successful"
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False, str(e)

@app.post("/api/upload", response_model=ChartSuggestionResponse)
async def upload_file(file: UploadFile):
    """Upload CSV file and get chart suggestions."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Read CSV file
        df = pd.read_csv(file.file)
        
        # Validate data
        is_valid, message = validate_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Get chart suggestions
        suggestions = get_chart_suggestions(df)
        
        # Create session data
        session_data = {
            "data": df,
            "suggestions": suggestions,
            "original_filename": file.filename,
            "upload_time": pd.Timestamp.now(),
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        # Store session
        session_manager.add_session(session_id, session_data)
        
        return ChartSuggestionResponse(
            session_id=session_id,
            suggestions=[ChartSuggestion(**s) for s in suggestions]
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/visualize", response_model=VisualizationResponse)
async def generate_chart(request: VisualizationRequest):
    """Generate a specific visualization."""
    session_data = session_manager.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find the requested chart configuration
    chart_config = None
    for suggestion in session_data["suggestions"]:
        if suggestion["id"] == request.chart_id:
            chart_config = suggestion
            break
    
    if not chart_config:
        raise HTTPException(status_code=404, detail="Chart configuration not found")
    
    try:
        # Generate the visualization
        plot_data = generate_visualization(session_data["data"], chart_config)
        
        return VisualizationResponse(
            chart_id=request.chart_id,
            type=chart_config["type"],
            title=chart_config["title"],
            description=chart_config["description"],
            plot_data=plot_data
        )
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/data_summary/{session_id}", response_model=DataSummary)
async def get_data_summary(session_id: str):
    """Get summary statistics for the uploaded data."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = session_data["data"]
        
        # Get column types
        column_types = df.dtypes.astype(str).to_dict()
        
        # Get missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Get unique values
        unique_values = df.nunique().to_dict()
        
        # Get numeric statistics
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        numeric_stats = None
        if not numeric_df.empty:
            numeric_stats = numeric_df.describe().to_dict()
        
        # Get categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        categorical_stats = None
        if not categorical_df.empty:
            categorical_stats = {
                col: {
                    'top_values': df[col].value_counts().head(5).to_dict(),
                    'unique_count': df[col].nunique()
                }
                for col in categorical_df.columns
            }
        
        # Get correlations for numeric columns
        correlations = None
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            correlations = {
                col1: {
                    col2: corr_matrix.loc[col1, col2]
                    for col2 in corr_matrix.columns
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.5
                }
                for col1 in corr_matrix.columns
            }
            # Remove empty correlation entries
            correlations = {k: v for k, v in correlations.items() if v}
        
        return DataSummary(
            filename=session_data["original_filename"],
            upload_time=session_data["upload_time"].isoformat(),
            row_count=session_data["row_count"],
            column_count=session_data["column_count"],
            columns=column_types,
            missing_values=missing_values,
            unique_values=unique_values,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            correlations=correlations
        )
    
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data."""
    if session_manager.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/sessions/cleanup")
async def cleanup_sessions():
    """Manually trigger session cleanup."""
    try:
        initial_count = len(session_manager.sessions)
        session_manager._cleanup_if_needed()
        final_count = len(session_manager.sessions)
        
        return {
            "message": "Session cleanup completed",
            "sessions_removed": initial_count - final_count,
            "remaining_sessions": final_count
        }
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rate_limit")
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for the client."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    if client_ip not in rate_limit_store:
        return {
            "requests_remaining": MAX_REQUESTS_PER_MINUTE,
            "reset_in_seconds": 0
        }
    
    # Get valid requests in the current window
    valid_requests = [
        t for t in rate_limit_store[client_ip]
        if current_time - t < RATE_LIMIT_DURATION
    ]
    
    # Calculate remaining requests and reset time
    requests_remaining = max(0, MAX_REQUESTS_PER_MINUTE - len(valid_requests))
    if valid_requests:
        reset_time = valid_requests[0] + RATE_LIMIT_DURATION
        reset_in_seconds = max(0, (reset_time - current_time).total_seconds())
    else:
        reset_in_seconds = 0
    
    return {
        "requests_remaining": requests_remaining,
        "reset_in_seconds": reset_in_seconds
    }

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Data Visualization API",
        version="1.0.0",
        description="""
        An intelligent data visualization API that uses Gemini AI to analyze CSV data and suggest appropriate visualizations.
        
        Features:
        - CSV file upload and analysis
        - AI-powered visualization suggestions
        - Interactive chart generation
        - Data summary and statistics
        - Session management
        
        Usage:
        1. Upload a CSV file using POST /api/upload
        2. Get visualization suggestions in the response
        3. Generate specific visualizations using POST /api/visualize
        4. Get data summary using GET /api/data_summary/{session_id}
        5. Clean up using DELETE /api/session/{session_id}
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "RateLimit": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
            "description": "Rate limited to 60 requests per minute"
        }
    }
    
    # Add response examples
    openapi_schema["paths"]["/api/upload"]["post"]["responses"]["200"]["content"]["application/json"]["example"] = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "suggestions": [
            {
                "id": "chart_1",
                "type": "scatter",
                "title": "Example Scatter Plot",
                "description": "Shows correlation between variables",
                "x_axis": "column1",
                "y_axis": "column2"
            }
        ]
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi 