#!/bin/bash

echo "Testing Chat Endpoints"
echo "===================="

# Test general chat
echo -e "\nTesting general chat:"
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How is my portfolio performing?"}' | json_pp

# Test portfolio chat
echo -e "\nTesting portfolio chat:"
curl -s -X POST http://localhost:8000/chat/portfolio \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the main risks in my portfolio?"}' | json_pp

# Test market chat
echo -e "\nTesting market chat:"
curl -s -X POST http://localhost:8000/chat/market \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the current market conditions?"}' | json_pp

echo -e "\nChat endpoint tests completed" 