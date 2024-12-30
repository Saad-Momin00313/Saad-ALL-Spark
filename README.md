# Data Visualization Tool 🎨

An intelligent data visualization tool that uses Gemini AI to analyze CSV data and suggest appropriate visualizations.

## Features 🚀

- **AI-Powered Visualization Suggestions**: Automatically analyzes your data and suggests the most insightful visualizations
- **Interactive Charts**: Generate interactive Plotly charts with detailed insights
- **Multiple Chart Types**: Supports various chart types including:
  - Bar charts
  - Pie charts
  - Histograms
  - Scatter plots
  - Box plots
  - Line charts
  - Heatmaps
  - Violin plots
  - Sunburst charts
  - Treemaps

## Setup 🛠️

1. Clone the repository:

```bash
git clone https://github.com/Saad-Momin00313/Saad-ALL-Spark.git
cd "Data Visualization Tool"
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file
   - Add your Gemini AI API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage 💡

1. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

2. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

3. Open your browser and go to:
   - API Documentation: http://localhost:8000/docs
   - Streamlit Interface: http://localhost:8501

## API Endpoints 🔌

- `POST /api/upload`: Upload CSV file and get visualization suggestions
- `POST /api/visualize`: Generate specific visualization
- `GET /api/data_summary/{session_id}`: Get data summary statistics
- `DELETE /api/session/{session_id}`: Clean up session data

## Demo 🎥

1. Upload your CSV file through the Streamlit interface
2. Get AI-powered visualization suggestions
3. Select a visualization to generate
4. View the interactive chart with insights

## Requirements 📋

See `requirements.txt` for full list of dependencies.

## License 📄

MIT License
