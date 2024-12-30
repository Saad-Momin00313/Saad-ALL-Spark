from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
import json
import yfinance as yf

from src.asset_tracker import AssetTracker
from src.financial_analyzer import FinancialAnalyzer
from src.ai_insights import AIInvestmentAdvisor
from src.chatbot import FinanceChatbot
from config.config import Config

app = FastAPI(title="Portfolio Management API")
config = Config()

# Initialize components
asset_tracker = AssetTracker()  # Initialize without database for now
financial_analyzer = FinancialAnalyzer()
ai_advisor = AIInvestmentAdvisor(config.GEMINI_API_KEY)
chatbot = FinanceChatbot(config.GEMINI_API_KEY)

@app.get("/portfolio")
async def get_portfolio() -> Dict:
    """Get portfolio overview"""
    try:
        assets = asset_tracker.get_all_assets()
        return {
            "assets": assets,
            "total_value": sum(asset["current_value"] for asset in assets.values()) if assets else 0,
            "performance_metrics": financial_analyzer.get_performance_metrics(assets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/assets")
async def add_asset(asset: Dict) -> Dict:
    """Add a new asset to the portfolio"""
    try:
        return asset_tracker.add_asset(asset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/analysis")
async def get_portfolio_analysis() -> Dict:
    """Get portfolio analysis"""
    try:
        assets = asset_tracker.get_all_assets()
        return financial_analyzer.analyze_portfolio_risk(assets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/insights")
async def get_market_insights() -> Dict:
    """Get AI-generated market insights"""
    try:
        return ai_advisor.get_market_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/recommendations")
async def get_market_recommendations() -> Dict:
    """Get AI-generated investment recommendations"""
    try:
        return ai_advisor.get_recommendations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/history")
async def get_portfolio_history() -> Dict:
    """Get portfolio history"""
    try:
        assets = asset_tracker.get_all_assets()
        return financial_analyzer.get_portfolio_history(assets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: Dict) -> Dict:
    """Chat with the AI assistant"""
    try:
        response = chatbot.generate_response(message["message"])
        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/portfolio")
async def portfolio_chat(message: Dict) -> Dict:
    """Chat about portfolio analysis"""
    try:
        portfolio = asset_tracker.get_all_assets()
        response = chatbot.analyze_portfolio(portfolio, message["message"])
        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/market")
async def market_chat(message: Dict) -> Dict:
    """Chat about market conditions"""
    try:
        response = chatbot.analyze_market(message["message"])
        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/sentiment")
async def get_market_sentiment() -> Dict:
    """Get AI-generated market sentiment analysis"""
    try:
        return ai_advisor.analyze_market_sentiment()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/news")
async def get_market_news() -> Dict:
    """Get AI-analyzed market news and impact assessment"""
    try:
        return ai_advisor.analyze_market_news()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/strategies")
async def get_investment_strategies() -> Dict:
    """Get AI-generated investment strategies"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return ai_advisor.generate_investment_strategies(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/risk_assessment")
async def get_risk_assessment() -> Dict:
    """Get AI-driven portfolio risk assessment"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return ai_advisor.assess_portfolio_risk(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/custom_recommendations")
async def get_custom_recommendations() -> Dict:
    """Get personalized portfolio recommendations"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return ai_advisor.get_custom_recommendations(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/optimize")
async def optimize_portfolio() -> Dict:
    """Optimize portfolio using Modern Portfolio Theory"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.optimize_portfolio(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/rebalance")
async def get_rebalancing_suggestions() -> Dict:
    """Get portfolio rebalancing suggestions"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.get_rebalancing_suggestions(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/correlation")
async def get_correlation_analysis() -> Dict:
    """Get portfolio correlation analysis"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.analyze_correlations(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}")
async def get_technical_analysis(symbol: str) -> Dict:
    """Get technical analysis for a symbol"""
    try:
        return financial_analyzer.analyze_technical_indicators(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}/chart")
async def get_price_chart(symbol: str, timeframe: str = "1y") -> Dict:
    """Get price chart data with technical analysis"""
    try:
        return financial_analyzer.get_price_chart_data(symbol, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}/support-resistance")
async def get_support_resistance(symbol: str) -> Dict:
    """Get support and resistance levels"""
    try:
        return financial_analyzer.calculate_support_resistance(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}/signals")
async def get_trading_signals(symbol: str) -> Dict:
    """Get trading signals based on technical analysis"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        return financial_analyzer._generate_trading_signals(hist)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}/volume")
async def get_volume_analysis(symbol: str) -> Dict:
    """Get volume analysis"""
    try:
        return financial_analyzer.analyze_volume(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/attribution")
async def get_performance_attribution() -> Dict:
    """Get portfolio performance attribution analysis"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.analyze_performance_attribution(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/backtest")
async def backtest_portfolio(start_date: str, end_date: str) -> Dict:
    """Backtest portfolio performance"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.backtest_portfolio(portfolio, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/assets/{asset_id}")
async def get_asset(asset_id: str) -> Dict:
    """Get a specific asset by ID"""
    try:
        return asset_tracker.get_asset(asset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/portfolio/assets/{asset_id}")
async def update_asset(asset_id: str, updates: Dict) -> Dict:
    """Update an existing asset"""
    try:
        return asset_tracker.update_asset(asset_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/portfolio/assets/{asset_id}")
async def delete_asset(asset_id: str) -> Dict:
    """Delete an asset from the portfolio"""
    try:
        return asset_tracker.delete_asset(asset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/conditions")
async def get_market_conditions() -> Dict:
    """Get overall market conditions"""
    try:
        return financial_analyzer.analyze_market_conditions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/sector")
async def get_sector_analysis() -> Dict:
    """Get sector-specific analysis"""
    try:
        portfolio = asset_tracker.get_all_assets()
        return financial_analyzer.analyze_sectors(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 