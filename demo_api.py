import requests
import json

def demo_visualization_api():
    """Demo script showing how to use the Data Visualization API."""
    
    # API endpoints
    BASE_URL = "http://localhost:8000"
    UPLOAD_URL = f"{BASE_URL}/api/upload"
    VISUALIZE_URL = f"{BASE_URL}/api/visualize"
    
    # Example: Upload a CSV file
    with open('sample_data.csv', 'rb') as f:
        files = {'file': ('sample_data.csv', f, 'text/csv')}
        response = requests.post(UPLOAD_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            session_id = result['session_id']
            suggestions = result['suggestions']
            
            print("\n=== Visualization Suggestions ===")
            for suggestion in suggestions:
                print(f"\nChart ID: {suggestion['id']}")
                print(f"Type: {suggestion['type']}")
                print(f"Title: {suggestion['title']}")
                print(f"Description: {suggestion['description']}")
            
            # Example: Generate a specific visualization
            if suggestions:
                chart_id = suggestions[0]['id']  # Use first suggestion
                viz_request = {
                    "session_id": session_id,
                    "chart_id": chart_id
                }
                
                viz_response = requests.post(VISUALIZE_URL, json=viz_request)
                if viz_response.status_code == 200:
                    viz_result = viz_response.json()
                    print("\n=== Generated Visualization ===")
                    print(f"Chart ID: {viz_result['chart_id']}")
                    print(f"Type: {viz_result['type']}")
                    print(f"Title: {viz_result['title']}")
                    print("Plot data available in 'plot_data' field")
                else:
                    print(f"Error generating visualization: {viz_response.text}")
        else:
            print(f"Error uploading file: {response.text}")

if __name__ == "__main__":
    demo_visualization_api() 