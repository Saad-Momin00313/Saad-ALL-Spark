# Portfolio Analysis API

A comprehensive API for portfolio management, market analysis, and investment insights powered by AI.

## Features

- Portfolio Management

  - Add and manage portfolio assets
  - Track portfolio performance
  - Generate performance attribution reports
  - Portfolio optimization suggestions

- Market Analysis

  - Real-time market insights
  - Investment recommendations
  - Market sentiment analysis
  - News aggregation

- Technical Analysis
  - Support and resistance levels
  - Trading signals
  - Volume analysis
  - Technical indicators

## Prerequisites

- Docker Desktop
  - [Mac Installation](https://docs.docker.com/desktop/install/mac-install/)
  - [Windows Installation](https://docs.docker.com/desktop/install/windows-install/)
  - [Linux Installation](https://docs.docker.com/engine/install/)

## Quick Start

1. **Clone the Repository**

   ```bash
   git clone [repository-url]
   cd port
   ```

2. **Environment Setup**

   ```bash
   # Copy example environment file
   cp .env.example .env

   # Edit .env file and add your credentials:
   # GEMINI_API_KEY=your_api_key_here
   # LOG_LEVEL=INFO
   # ENVIRONMENT=production
   ```

3. **Build and Start**

   ```bash
   # Build and start the application
   docker-compose up -d

   # Verify the application is running
   curl http://localhost:8000/health
   ```

## API Documentation

Once the application is running, you can access:

- Interactive API documentation: `http://localhost:8000/docs`
- OpenAPI specification: `http://localhost:8000/openapi.json`

### Key Endpoints

- Portfolio Management

  - `GET /portfolio` - Get portfolio overview
  - `POST /portfolio/assets` - Add new asset
  - `GET /portfolio/analysis` - Get portfolio analysis
  - `GET /portfolio/attribution` - Get performance attribution

- Market Analysis

  - `GET /market/insights` - Get market insights
  - `GET /market/recommendations` - Get investment recommendations
  - `GET /market/sentiment` - Get market sentiment
  - `GET /market/news` - Get market news

- Technical Analysis
  - `GET /technical/{symbol}/support-resistance` - Get support/resistance levels
  - `GET /technical/{symbol}/signals` - Get trading signals
  - `GET /technical/{symbol}/volume` - Get volume analysis

## Common Commands

```bash
# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Restart the application
docker-compose restart

# Update to latest version
git pull
docker-compose build
docker-compose up -d
```

## Troubleshooting

1. **Container Issues**

   ```bash
   # Check container status
   docker-compose ps

   # View detailed logs
   docker-compose logs -f

   # Rebuild from scratch
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

2. **API Key Issues**

   - Ensure your Gemini API key is correctly set in the `.env` file
   - Verify the API key has the necessary permissions
   - Check the logs for any authentication errors

3. **Port Conflicts**
   - If port 8000 is already in use, modify the port mapping in `docker-compose.yml`
   - Default port can be changed by modifying the `ports` section:
     ```yaml
     ports:
       - "8001:8000" # Change 8001 to your desired port
     ```

## Development

For development purposes:

```bash
# Run with hot reload
docker-compose -f docker-compose.dev.yml up

# Run tests
docker-compose exec api python -m pytest

# Check logs in real-time
docker-compose logs -f
```

## Security Notes

- Never commit your `.env` file or API keys
- Keep Docker and all dependencies updated
- Use HTTPS in production
- Follow security best practices for API key management

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
