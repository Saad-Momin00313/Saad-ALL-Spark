#!/bin/bash

echo "Running All Endpoint Tests"
echo "========================"

# Make all test scripts executable
chmod +x test_portfolio_endpoints.sh
chmod +x test_market_endpoints.sh
chmod +x test_technical_endpoints.sh
chmod +x test_chat_endpoints.sh

# Run all test scripts
echo -e "\nRunning portfolio endpoint tests..."
./test_portfolio_endpoints.sh

echo -e "\nRunning market endpoint tests..."
./test_market_endpoints.sh

echo -e "\nRunning technical endpoint tests..."
./test_technical_endpoints.sh

echo -e "\nRunning chat endpoint tests..."
./test_chat_endpoints.sh

echo -e "\nAll endpoint tests completed" 