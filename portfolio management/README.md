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
2. Sign in with your Google account
3. Create a new API key
4. Keep this key secure and never commit it to version control

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

#### CORS Configuration

The API supports CORS for frontend integration with the following configurations:

- Allowed origins: All (customizable in production)
- Allowed methods: GET, POST, PUT, DELETE, OPTIONS
- Allowed headers: All standard headers
- Credentials: Supported

#### API Integration Guide

1. **Authentication**:

   - Use Bearer token authentication
   - Include token in Authorization header
   - Example: `Authorization: Bearer your_token_here`

2. **Response Format**:
   All endpoints return JSON with consistent structure:

   ```json
   {
     "status": "success|error",
     "data": {...},
     "message": "Optional message"
   }
   ```

3. **Error Handling**:

   - HTTP status codes are properly used
   - Detailed error messages provided
   - Validation errors include field-specific details

4. **Rate Limiting**:
   - 100 requests per minute per IP (configurable)
   - Rate limit headers included in response

#### Key Endpoints:

- `/api/v1/portfolio/`: Portfolio management endpoints
- `/api/v1/market/`: Market data endpoints
- `/api/v1/analysis/`: Financial analysis endpoints
- `/api/v1/chat/`: Chatbot interaction endpoints

### Production Readiness Checklist

1. **Security**:

   - ✅ SSL/TLS encryption
   - ✅ API key security
   - ✅ Rate limiting
   - ✅ Input validation
   - ✅ CORS configuration

2. **Performance**:

   - ✅ Database indexing
   - ✅ Caching implemented
   - ✅ Response compression
   - ✅ Load balancing ready

3. **Monitoring**:

   - ✅ Health check endpoints
   - ✅ Logging system
   - ✅ Error tracking
   - ✅ Performance metrics

4. **Scalability**:

   - ✅ Containerized
   - ✅ Stateless design
   - ✅ Horizontal scaling ready
   - ✅ Cache strategy

5. **Deployment**:
   - ✅ CI/CD ready
   - ✅ Environment configurations
   - ✅ Backup strategy
   - ✅ Rollback procedures

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


### Contact

For any issues or questions, please contact the development team.
