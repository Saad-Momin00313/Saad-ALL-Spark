#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Portfolio Management with Multiple Assets${NC}"

# Add diverse set of assets
echo -e "\n${GREEN}Adding multiple assets to portfolio:${NC}"

# Add AAPL stock
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "type": "stock",
    "shares": 10,
    "price": 150.0
  }'

# Add MSFT stock
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "MSFT",
    "type": "stock",
    "shares": 5,
    "price": 300.0
  }'

# Add GOOGL stock
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "GOOGL",
    "type": "stock",
    "shares": 3,
    "price": 2500.0
  }'

# Add BTC-USD cryptocurrency
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "type": "crypto",
    "shares": 0.5,
    "price": 40000.0
  }'

# Add ETH-USD cryptocurrency
curl -X POST "http://localhost:8000/portfolio/assets" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETH-USD",
    "type": "crypto",
    "shares": 2.0,
    "price": 2000.0
  }'

echo -e "\n${GREEN}Getting updated portfolio overview:${NC}"
curl -X GET "http://localhost:8000/portfolio"

echo -e "\n${GREEN}Getting detailed portfolio analysis:${NC}"
curl -X GET "http://localhost:8000/portfolio/analysis"

echo -e "\n${GREEN}Getting performance attribution:${NC}"
curl -X GET "http://localhost:8000/portfolio/attribution"

echo -e "\n${BLUE}Testing Technical Analysis for Multiple Symbols${NC}"

# Test technical analysis for AAPL
echo -e "\n${GREEN}Technical Analysis for AAPL:${NC}"
curl -X GET "http://localhost:8000/technical/AAPL/support-resistance"
curl -X GET "http://localhost:8000/technical/AAPL/signals"
curl -X GET "http://localhost:8000/technical/AAPL/volume"

# Test technical analysis for MSFT
echo -e "\n${GREEN}Technical Analysis for MSFT:${NC}"
curl -X GET "http://localhost:8000/technical/MSFT/support-resistance"
curl -X GET "http://localhost:8000/technical/MSFT/signals"
curl -X GET "http://localhost:8000/technical/MSFT/volume"

echo -e "\n${GREEN}Getting portfolio optimization with multiple assets:${NC}"
curl -X GET "http://localhost:8000/portfolio/optimize"

echo -e "\n${GREEN}Getting correlation analysis for diverse portfolio:${NC}"
curl -X GET "http://localhost:8000/portfolio/correlation"

echo -e "\n${BLUE}All specific tests completed${NC}" 