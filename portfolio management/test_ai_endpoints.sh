#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing AI-Powered Endpoints${NC}"

# Test market insights
echo -e "\n${GREEN}Getting market insights:${NC}"
curl -X GET "http://localhost:8000/market/insights" | python -m json.tool

# Test market sentiment
echo -e "\n${GREEN}Getting market sentiment:${NC}"
curl -X GET "http://localhost:8000/market/sentiment" | python -m json.tool

# Test market news
echo -e "\n${GREEN}Getting market news:${NC}"
curl -X GET "http://localhost:8000/market/news" | python -m json.tool

# Test risk assessment
echo -e "\n${GREEN}Getting AI-powered risk assessment:${NC}"
curl -X GET "http://localhost:8000/portfolio/risk_assessment" | python -m json.tool

echo -e "\n${BLUE}All AI endpoint tests completed${NC}" 