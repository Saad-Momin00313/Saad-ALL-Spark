import pytest
from fastapi.testclient import TestClient
from main import app
import pandas as pd
import io
import time
import json

client = TestClient(app)

def create_sample_csv():
    """Create a sample CSV file for testing with meaningful data."""
    df = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'sales': [1200, 1500, 2000, 2300, 2800, 3100, 3500, 3800, 3400, 3200, 2900, 3600],
        'customers': [120, 150, 180, 200, 250, 280, 310, 340, 290, 270, 260, 320],
        'avg_transaction': [100, 100, 111, 115, 112, 111, 113, 112, 117, 119, 112, 113],
        'category': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3', 'Q4', 'Q4', 'Q4']
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_complete_visualization_workflow():
    """Test the complete visualization workflow with actual results."""
    # 1. Upload CSV
    csv_content = create_sample_csv()
    files = {
        'file': ('test.csv', csv_content, 'text/csv')
    }
    upload_response = client.post("/api/upload", files=files)
    assert upload_response.status_code == 200
    job_id = upload_response.json()["job_id"]
    
    # 2. Wait for processing to complete (with timeout)
    max_attempts = 10
    attempt = 0
    results = None
    
    while attempt < max_attempts:
        results_response = client.get(f"/api/visualizations/{job_id}")
        assert results_response.status_code == 200
        results = results_response.json()
        
        if results["status"] == "completed":
            break
        elif results["status"] == "failed":
            pytest.fail(f"Job failed with error: {results.get('error')}")
        
        time.sleep(2)  # Wait 2 seconds before next attempt
        attempt += 1
    
    # 3. Verify and display results
    assert results["status"] == "completed", "Job did not complete in time"
    assert results["visualizations"] is not None, "No visualizations were generated"
    
    # 4. Print visualization details
    print("\nVisualization Results:")
    print("=" * 50)
    for idx, viz in enumerate(results["visualizations"], 1):
        print(f"\nVisualization {idx}:")
        print(f"Type: {viz['type']}")
        print(f"Title: {viz['title']}")
        print(f"Description: {viz['description']}")
        print("-" * 30)
    
    # 5. Save a sample visualization to file for inspection
    if results["visualizations"]:
        with open("sample_visualization.json", "w") as f:
            json.dump(results["visualizations"][0], f, indent=2)
        print("\nSaved first visualization to 'sample_visualization.json'")

def test_upload_invalid_file():
    """Test uploading an invalid file type."""
    files = {
        'file': ('test.txt', 'invalid content', 'text/plain')
    }
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]

def test_get_nonexistent_job():
    """Test getting results for a non-existent job."""
    response = client.get("/api/visualizations/nonexistent-job")
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]

if __name__ == "__main__":
    print("Running visualization API tests...")
    pytest.main([__file__, "-v"]) 