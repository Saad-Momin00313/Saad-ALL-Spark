#!/bin/bash

echo "Testing Portfolio Management Endpoints"
echo "===================================="

# Test get empty portfolio
echo -e "\nGetting initial empty portfolio:"
curl -s http://localhost:8000/portfolio | json_pp

# Test adding assets one by one and verify portfolio after each addition
echo -e "\nAdding AAPL stock and verifying:"
curl -s -X POST http://localhost:8000/portfolio/assets \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","type":"stock","shares":10,"price":150.0}' | json_pp
echo -e "\nPortfolio after adding AAPL:"
curl -s http://localhost:8000/portfolio | json_pp

echo -e "\nAdding MSFT stock and verifying:"
curl -s -X POST http://localhost:8000/portfolio/assets \
  -H "Content-Type: application/json" \
  -d '{"symbol":"MSFT","type":"stock","shares":15,"price":200.0}' | json_pp
echo -e "\nPortfolio after adding MSFT:"
curl -s http://localhost:8000/portfolio | json_pp

echo -e "\nAdding GOOGL stock and verifying:"
curl -s -X POST http://localhost:8000/portfolio/assets \
  -H "Content-Type: application/json" \
  -d '{"symbol":"GOOGL","type":"stock","shares":5,"price":180.0}' | json_pp
echo -e "\nPortfolio after adding GOOGL:"
curl -s http://localhost:8000/portfolio | json_pp

echo -e "\nAdding BTC-USD crypto and verifying:"
curl -s -X POST http://localhost:8000/portfolio/assets \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USD","type":"crypto","shares":0.5,"price":40000.0}' | json_pp
echo -e "\nPortfolio after adding BTC-USD:"
curl -s http://localhost:8000/portfolio | json_pp

echo -e "\nAdding VFINX mutual fund and verifying:"
curl -s -X POST http://localhost:8000/portfolio/assets \
  -H "Content-Type: application/json" \
  -d '{"symbol":"VFINX","type":"mutual_fund","shares":20,"price":300.0}' | json_pp
echo -e "\nPortfolio after adding VFINX:"
curl -s http://localhost:8000/portfolio | json_pp

# Test portfolio analysis with populated portfolio
echo -e "\nGetting portfolio analysis:"
curl -s http://localhost:8000/portfolio/analysis | json_pp

# Test portfolio history with populated portfolio
echo -e "\nGetting portfolio history:"
curl -s http://localhost:8000/portfolio/history | json_pp

# Test portfolio optimization with populated portfolio
echo -e "\nGetting portfolio optimization:"
curl -s http://localhost:8000/portfolio/optimize | json_pp

# Test portfolio rebalancing with populated portfolio
echo -e "\nGetting rebalancing suggestions:"
curl -s http://localhost:8000/portfolio/rebalance | json_pp

# Test correlation analysis with populated portfolio
echo -e "\nGetting correlation analysis:"
curl -s http://localhost:8000/portfolio/correlation | json_pp

# Test performance attribution with populated portfolio
echo -e "\nGetting performance attribution:"
curl -s http://localhost:8000/portfolio/attribution | json_pp

# Test portfolio backtesting with populated portfolio
echo -e "\nTesting portfolio backtesting:"
curl -s "http://localhost:8000/portfolio/backtest?start_date=2023-01-01&end_date=2023-12-31" | json_pp

# Display portfolio summary
echo -e "\nFinal portfolio summary:"
curl -s http://localhost:8000/portfolio | json_pp

echo -e "\nPortfolio endpoint tests completed" 