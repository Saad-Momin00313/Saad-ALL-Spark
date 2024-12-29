#!/bin/bash

echo "Testing Technical Analysis Endpoints"
echo "=================================="

SYMBOL="AAPL"

# Test basic technical analysis
echo -e "\nGetting technical analysis for $SYMBOL:"
curl -s "http://localhost:8000/technical/$SYMBOL" | json_pp

# Test price chart
echo -e "\nGetting price chart for $SYMBOL:"
curl -s "http://localhost:8000/technical/$SYMBOL/chart?timeframe=1y" | json_pp

# Test support/resistance levels
echo -e "\nGetting support/resistance levels for $SYMBOL:"
curl -s "http://localhost:8000/technical/$SYMBOL/support-resistance" | json_pp

# Test trading signals
echo -e "\nGetting trading signals for $SYMBOL:"
curl -s "http://localhost:8000/technical/$SYMBOL/signals" | json_pp

# Test volume analysis
echo -e "\nGetting volume analysis for $SYMBOL:"
curl -s "http://localhost:8000/technical/$SYMBOL/volume" | json_pp

echo -e "\nTechnical analysis endpoint tests completed" 