import streamlit as st
import requests
import pandas as pd
import plotly.io as pio
import json

# API settings
API_URL = "http://localhost:8000"

def main():
    st.set_page_config(page_title="Data Visualization App", layout="wide")
    
    st.title("📊 Intelligent Data Visualization")
    st.write("Upload your CSV file and get AI-powered visualization recommendations")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = None
    if 'selected_chart' not in st.session_state:
        st.session_state.selected_chart = None
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display sample of uploaded data
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
        
        # Get chart suggestions if not already done
        if st.session_state.session_id is None:
            with st.spinner("Analyzing data and generating chart suggestions..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Upload file and get suggestions
                files = {"file": uploaded_file}
                response = requests.post(f"{API_URL}/api/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.session_id = result["session_id"]
                    st.session_state.suggestions = result["suggestions"]
                else:
                    st.error(f"Error: {response.json()['detail']}")
                    return
        
        # Display chart suggestions
        if st.session_state.suggestions:
            st.subheader("🎨 Available Visualizations")
            
            # Create a selection box for charts
            chart_options = {f"{s['title']} ({s['type']})": s['id'] for s in st.session_state.suggestions}
            selected_option = st.selectbox(
                "Select a visualization to generate:",
                options=list(chart_options.keys())
            )
            
            # Get the selected chart ID
            selected_chart_id = chart_options[selected_option]
            
            # Find the selected chart details
            selected_chart = next(s for s in st.session_state.suggestions if s['id'] == selected_chart_id)
            
            # Display chart details
            with st.expander("Chart Details", expanded=True):
                st.markdown(f"**Type:** {selected_chart['type'].capitalize()}")
                st.markdown(f"**Description:** {selected_chart['description']}")
                st.markdown(f"**Data Used:**")
                st.markdown(f"- X-axis: {selected_chart['x_axis']}")
                st.markdown(f"- Y-axis: {selected_chart['y_axis']}")
                if selected_chart.get('additional_params'):
                    st.markdown("**Additional Parameters:**")
                    for param, value in selected_chart['additional_params'].items():
                        if value:
                            st.markdown(f"- {param}: {value}")
            
            # Generate visualization button
            if st.button("Generate Visualization"):
                with st.spinner("Generating visualization..."):
                    # Request visualization
                    response = requests.post(
                        f"{API_URL}/api/visualize",
                        json={
                            "session_id": st.session_state.session_id,
                            "chart_id": selected_chart_id
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display the visualization
                        st.subheader("📈 Generated Visualization")
                        fig = pio.from_json(json.dumps(result["plot_data"]))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display any insights or observations
                        st.markdown("### 📝 Insights")
                        st.write(result["description"])
                    else:
                        st.error(f"Error generating visualization: {response.json()['detail']}")

if __name__ == "__main__":
    main() 