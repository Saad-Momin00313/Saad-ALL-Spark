#!/bin/bash

echo "Testing Market Analysis Endpoints"
echo "================================"

# Test market insights
echo -e "\nGetting market insights:"
curl -s http://localhost:8000/market/insights | json_pp

# Test market recommendations
echo -e "\nGetting market recommendations:"
curl -s http://localhost:8000/market/recommendations | json_pp

# Test market sentiment
echo -e "\nGetting market sentiment:"
curl -s http://localhost:8000/market/sentiment | json_pp

# Test market news
echo -e "\nGetting market news:"
curl -s http://localhost:8000/market/news | json_pp

# Test investment strategies
echo -e "\nGetting investment strategies:"
curl -s http://localhost:8000/portfolio/strategies | json_pp

# Test custom recommendations
echo -e "\nGetting custom recommendations:"
curl -s http://localhost:8000/portfolio/custom_recommendations | json_pp

echo -e "\nMarket endpoint tests completed" 