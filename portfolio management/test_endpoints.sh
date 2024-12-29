#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Portfolio Management Endpoints${NC}"

# Test portfolio overview
echo -e "\n${GREEN}Getting portfolio overview:${NC}"
curl -X GET "http://localhost:8000/portfolio"

# Add sample assets
echo -e "\n${GREEN}Adding sample assets:${NC}"
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "type": "stock",
    "shares": 10,
    "price": 150.0
  }'

curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "type": "crypto",
    "shares": 0.5,
    "price": 40000.0
  }'

# Get portfolio analysis
echo -e "\n${GREEN}Getting portfolio analysis:${NC}"
curl -X GET "http://localhost:8000/portfolio/analysis"

# Get performance attribution
echo -e "\n${GREEN}Getting performance attribution:${NC}"
curl -X GET "http://localhost:8000/portfolio/attribution"

echo -e "\n${BLUE}Testing Market Analysis Endpoints${NC}"

# Get market insights
echo -e "\n${GREEN}Getting market insights:${NC}"
curl -X GET "http://localhost:8000/market/insights"

# Get investment recommendations
echo -e "\n${GREEN}Getting investment recommendations:${NC}"
curl -X GET "http://localhost:8000/market/recommendations"

# Get market sentiment
echo -e "\n${GREEN}Getting market sentiment:${NC}"
curl -X GET "http://localhost:8000/market/sentiment"

# Get market news
echo -e "\n${GREEN}Getting market news:${NC}"
curl -X GET "http://localhost:8000/market/news"

echo -e "\n${BLUE}Testing Technical Analysis Endpoints${NC}"

# Get support/resistance levels for AAPL
echo -e "\n${GREEN}Getting support/resistance levels for AAPL:${NC}"
curl -X GET "http://localhost:8000/technical/AAPL/support-resistance"

# Get trading signals for AAPL
echo -e "\n${GREEN}Getting trading signals for AAPL:${NC}"
curl -X GET "http://localhost:8000/technical/AAPL/signals"

# Get volume analysis for AAPL
echo -e "\n${GREEN}Getting volume analysis for AAPL:${NC}"
curl -X GET "http://localhost:8000/technical/AAPL/volume"

# Test portfolio optimization
echo -e "\n${GREEN}Getting portfolio optimization suggestions:${NC}"
curl -X GET "http://localhost:8000/portfolio/optimize"

# Test risk assessment
echo -e "\n${GREEN}Getting risk assessment:${NC}"
curl -X GET "http://localhost:8000/portfolio/risk_assessment"

# Test correlation analysis
echo -e "\n${GREEN}Getting correlation analysis:${NC}"
curl -X GET "http://localhost:8000/portfolio/correlation"

echo -e "\n${BLUE}All tests completed${NC}" 