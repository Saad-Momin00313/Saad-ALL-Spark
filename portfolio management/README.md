# Portfolio Management System

## Quick Start Guide

### Prerequisites

- Docker
- Docker Compose
- Git
- Gemini API Key (Get it from https://makersuite.google.com/app/apikey)

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Saad-Momin00313/Saad-ALL-Spark.git
cd Saad-ALL-Spark/portfolio\ management
```

2. Environment Setup:

#### Getting Your API Key

1. Go to https://makersuite.google.com/app/apikey


#### Setting Up Environment



**Using export:**

```bash
# Set the API key as an environment variable
export GEMINI_API_KEY=your_actual_api_key_here

# Run the application
docker-compose up --build
```



Other environment variables (optional):

```env
# Optional - Default values shown
ENVIRONMENT=development
LOG_LEVEL=INFO
MARKET_DATA_CACHE_DURATION=3600
DEFAULT_MARKET_INDEX=^GSPC
```

3. Running the Application:

#### For Development:

```bash
docker-compose up --build
```

#### For Production:

```bash
docker-compose -f docker-compose.prod.yml up --build
```

### API Endpoints

The API will be available at:

- Development: `http://localhost:8000`
- Production: `https://localhost:443`

### For Frontend Team

#### API Documentation

The API endpoints are available at:

- Swagger UI: `http://localhost:8000/docs` (Development)
- ReDoc: `http://localhost:8000/redoc` (Development)

#### Key Endpoints:

- `/api/v1/portfolio/`: Portfolio management endpoints
- `/api/v1/market/`: Market data endpoints
- `/api/v1/analysis/`: Financial analysis endpoints
- `/api/v1/chat/`: Chatbot interaction endpoints

### Testing

While test files are not included in the repository, you can run tests locally:

1. Set up a local development environment
2. Run the test suite using the provided shell scripts

### Troubleshooting

1. If you encounter permission issues:

```bash
chmod +x *.sh
```

2. If ports are already in use:

```bash
# Check running containers
docker ps
# Stop existing containers
docker-compose down
```

3. To view logs:

```bash
docker-compose logs -f
```

### Container Management

1. To stop the containers:

```bash
docker-compose down
```

2. To rebuild after changes:

```bash
docker-compose up --build
```

3. To view running containers:

```bash
docker ps
```

### Security Best Practices

1. API Key Security:

   - Never commit your API key to version control
   - Don't share your API key with others
   - Each developer should get their own API key
   - Rotate API keys periodically
   - Use environment variables for sensitive data

2. Production Deployment:
   - Use secure secrets management
   - Consider using cloud key management services
   - Implement proper authentication
   - Use HTTPS in production
   - Regular security audits

### Contact

For any issues or questions, please contact the development team.
