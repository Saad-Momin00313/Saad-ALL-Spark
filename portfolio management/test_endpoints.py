import requests
import json
import time
from datetime import datetime
from typing import List, Dict

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # Increased timeout
MAX_RETRIES = 3
RETRY_DELAY = 2

def wait_for_server():
    """Wait for server to be ready"""
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/portfolio", timeout=TIMEOUT)
            if response.status_code == 200:
                return True
        except:
            time.sleep(RETRY_DELAY)
    return False

def validate_response(response: requests.Response, endpoint: str) -> None:
    """Validate API response"""
    if response.status_code != 200:
        raise Exception(f"Error in response from {endpoint}: {response.text}")

def test_portfolio_endpoints(results: List) -> None:
    """Test portfolio-related endpoints"""
    try:
        # Test get portfolio
        print("\nGET /portfolio")
        response = requests.get(f"{BASE_URL}/portfolio", timeout=TIMEOUT)
        validate_response(response, "/portfolio")
        results.append("\nGET /portfolio")
        results.append(json.dumps(response.json(), indent=2))

        # Add test assets
        test_assets = [
            {
                "symbol": "AAPL",
                "type": "stock",
                "shares": 10,
                "price": 150.0
            },
            {
                "symbol": "MSFT",
                "type": "stock",
                "shares": 15,
                "price": 200.0
            },
            {
                "symbol": "GOOGL",
                "type": "stock",
                "shares": 5,
                "price": 180.0
            },
            {
                "symbol": "AMZN",
                "type": "stock",
                "shares": 8,
                "price": 120.0
            },
            {
                "symbol": "BTC-USD",
                "type": "crypto",
                "shares": 0.5,
                "price": 40000.0
            },
            {
                "symbol": "VFINX",
                "type": "mutual_fund",
                "shares": 20,
                "price": 300.0
            }
        ]

        for asset in test_assets:
            print(f"\nPOST /portfolio/assets - {asset['symbol']}")
            response = requests.post(
                f"{BASE_URL}/portfolio/assets",
                json=asset,
                timeout=TIMEOUT
            )
            validate_response(response, f"/portfolio/assets - {asset['symbol']}")
            results.append(f"\nPOST /portfolio/assets - {asset['symbol']}")
            results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing portfolio endpoints: {str(e)}")
        results.append(f"\nError testing portfolio endpoints: {str(e)}")

def test_market_endpoints(results: List) -> None:
    """Test market-related endpoints"""
    try:
        endpoints = [
            "/market/insights",
            "/market/recommendations",
            "/market/sentiment",
            "/market/news"
        ]
        
        for endpoint in endpoints:
            print(f"\nGET {endpoint}")
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            validate_response(response, endpoint)
            results.append(f"\nGET {endpoint}")
            results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing market endpoints: {str(e)}")
        results.append(f"\nError testing market endpoints: {str(e)}")

def test_analysis_endpoints(results: List) -> None:
    """Test analysis endpoints"""
    try:
        endpoints = [
            "/portfolio/analysis",
            "/portfolio/history",
            "/portfolio/optimize",
            "/portfolio/rebalance",
            "/portfolio/correlation",
            "/portfolio/risk_assessment",
            "/portfolio/custom_recommendations"
        ]
        
        for endpoint in endpoints:
            print(f"\nGET {endpoint}")
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            validate_response(response, endpoint)
            results.append(f"\nGET {endpoint}")
            results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing analysis endpoints: {str(e)}")
        results.append(f"\nError testing analysis endpoints: {str(e)}")

def test_new_endpoints(results: List) -> None:
    """Test the newly added endpoints"""
    try:
        # Test performance attribution
        print("\nGET /portfolio/attribution")
        response = requests.get(f"{BASE_URL}/portfolio/attribution", timeout=TIMEOUT)
        validate_response(response, "/portfolio/attribution")
        results.append("\nGET /portfolio/attribution")
        results.append(json.dumps(response.json(), indent=2))

        # Test portfolio backtesting
        print("\nGET /portfolio/backtest")
        params = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        response = requests.get(f"{BASE_URL}/portfolio/backtest", params=params, timeout=TIMEOUT)
        validate_response(response, "/portfolio/backtest")
        results.append("\nGET /portfolio/backtest")
        results.append(json.dumps(response.json(), indent=2))

        # Test support/resistance levels
        print("\nGET /technical/AAPL/support-resistance")
        response = requests.get(f"{BASE_URL}/technical/AAPL/support-resistance", timeout=TIMEOUT)
        validate_response(response, "/technical/AAPL/support-resistance")
        results.append("\nGET /technical/AAPL/support-resistance")
        results.append(json.dumps(response.json(), indent=2))

        # Test trading signals
        print("\nGET /technical/AAPL/signals")
        response = requests.get(f"{BASE_URL}/technical/AAPL/signals", timeout=TIMEOUT)
        validate_response(response, "/technical/AAPL/signals")
        results.append("\nGET /technical/AAPL/signals")
        results.append(json.dumps(response.json(), indent=2))

        # Test volume analysis
        print("\nGET /technical/AAPL/volume")
        response = requests.get(f"{BASE_URL}/technical/AAPL/volume", timeout=TIMEOUT)
        validate_response(response, "/technical/AAPL/volume")
        results.append("\nGET /technical/AAPL/volume")
        results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing new endpoints: {str(e)}")
        results.append(f"\nError testing new endpoints: {str(e)}")

def test_chat_endpoints(results: List) -> None:
    """Test chat-related endpoints"""
    try:
        test_messages = [
            {
                "endpoint": "/chat",
                "message": "How is my portfolio performing?"
            },
            {
                "endpoint": "/chat/portfolio",
                "message": "What are the main risks in my portfolio?"
            },
            {
                "endpoint": "/chat/market",
                "message": "What are the current market conditions?"
            }
        ]
        
        for test in test_messages:
            print(f"\nPOST {test['endpoint']}")
            response = requests.post(
                f"{BASE_URL}{test['endpoint']}", 
                json={"message": test['message']},
                timeout=TIMEOUT
            )
            validate_response(response, test['endpoint'])
            results.append(f"\nPOST {test['endpoint']}")
            results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing chat endpoints: {str(e)}")
        results.append(f"\nError testing chat endpoints: {str(e)}")

def test_technical_endpoints(results: List) -> None:
    """Test technical analysis endpoints"""
    try:
        # Test basic technical analysis
        print("\nGET /technical/AAPL")
        response = requests.get(f"{BASE_URL}/technical/AAPL", timeout=TIMEOUT)
        validate_response(response, "/technical/AAPL")
        results.append("\nGET /technical/AAPL")
        results.append(json.dumps(response.json(), indent=2))

        # Test price chart
        print("\nGET /technical/AAPL/chart")
        response = requests.get(f"{BASE_URL}/technical/AAPL/chart?timeframe=1y", timeout=TIMEOUT)
        validate_response(response, "/technical/AAPL/chart")
        results.append("\nGET /technical/AAPL/chart")
        results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing technical endpoints: {str(e)}")
        results.append(f"\nError testing technical endpoints: {str(e)}")

def test_asset_management_endpoints(results: List) -> None:
    """Test asset management endpoints"""
    try:
        # First add an asset to get its ID
        test_asset = {
            "symbol": "NVDA",
            "type": "stock",
            "shares": 5,
            "price": 400.0
        }
        
        print("\nPOST /portfolio/assets")
        response = requests.post(
            f"{BASE_URL}/portfolio/assets",
            json=test_asset,
            timeout=TIMEOUT
        )
        validate_response(response, "/portfolio/assets")
        results.append("\nPOST /portfolio/assets")
        results.append(json.dumps(response.json(), indent=2))
        
        asset_id = response.json()["asset_id"]
        
        # Test GET single asset
        print(f"\nGET /portfolio/assets/{asset_id}")
        response = requests.get(
            f"{BASE_URL}/portfolio/assets/{asset_id}",
            timeout=TIMEOUT
        )
        validate_response(response, f"/portfolio/assets/{asset_id}")
        results.append(f"\nGET /portfolio/assets/{asset_id}")
        results.append(json.dumps(response.json(), indent=2))
        
        # Test UPDATE asset
        update_data = {
            "shares": 10,
            "price": 450.0
        }
        print(f"\nPUT /portfolio/assets/{asset_id}")
        response = requests.put(
            f"{BASE_URL}/portfolio/assets/{asset_id}",
            json=update_data,
            timeout=TIMEOUT
        )
        validate_response(response, f"/portfolio/assets/{asset_id}")
        results.append(f"\nPUT /portfolio/assets/{asset_id}")
        results.append(json.dumps(response.json(), indent=2))
        
        # Test DELETE asset
        print(f"\nDELETE /portfolio/assets/{asset_id}")
        response = requests.delete(
            f"{BASE_URL}/portfolio/assets/{asset_id}",
            timeout=TIMEOUT
        )
        validate_response(response, f"/portfolio/assets/{asset_id}")
        results.append(f"\nDELETE /portfolio/assets/{asset_id}")
        results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing asset management endpoints: {str(e)}")
        results.append(f"\nError testing asset management endpoints: {str(e)}")

def test_additional_analysis_endpoints(results: List) -> None:
    """Test additional analysis endpoints"""
    try:
        # Test market conditions
        print("\nGET /market/conditions")
        response = requests.get(f"{BASE_URL}/market/conditions", timeout=TIMEOUT)
        validate_response(response, "/market/conditions")
        results.append("\nGET /market/conditions")
        results.append(json.dumps(response.json(), indent=2))
        
        # Test sector analysis
        print("\nGET /portfolio/sector")
        response = requests.get(f"{BASE_URL}/portfolio/sector", timeout=TIMEOUT)
        validate_response(response, "/portfolio/sector")
        results.append("\nGET /portfolio/sector")
        results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing additional analysis endpoints: {str(e)}")
        results.append(f"\nError testing additional analysis endpoints: {str(e)}")

def test_strategy_endpoints(results: List) -> None:
    """Test strategy-related endpoints"""
    try:
        print("\nGET /portfolio/strategies")
        response = requests.get(f"{BASE_URL}/portfolio/strategies", timeout=TIMEOUT)
        validate_response(response, "/portfolio/strategies")
        results.append("\nGET /portfolio/strategies")
        results.append(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"Error testing strategy endpoints: {str(e)}")
        results.append(f"\nError testing strategy endpoints: {str(e)}")

def main():
    results = []
    results.append(f"Test Run: {datetime.now().isoformat()}\n\n")
    
    # Wait for server to be ready
    if not wait_for_server():
        print("Error: Server not ready")
        return
    
    try:
        # Test portfolio endpoints
        test_portfolio_endpoints(results)
        
        # Test asset management endpoints
        test_asset_management_endpoints(results)
        
        # Test market endpoints
        test_market_endpoints(results)
        
        # Test analysis endpoints
        test_analysis_endpoints(results)
        
        # Test additional analysis endpoints
        test_additional_analysis_endpoints(results)
        
        # Test strategy endpoints
        test_strategy_endpoints(results)
        
        # Test chat endpoints
        test_chat_endpoints(results)
        
        # Test technical endpoints
        test_technical_endpoints(results)
        
        # Test new endpoints
        test_new_endpoints(results)
        
        # Write results to file
        with open("endpoint_test_results.txt", "w") as f:
            f.write("\n".join(results))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        with open("endpoint_test_results.txt", "w") as f:
            f.write(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 