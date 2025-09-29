# CVML Deployment Guide

This guide covers different deployment options for the CVML Cardio Health Check Kit Analyzer.

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)
- Git

### Local Development Setup

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd CVML
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Start the Application**
   ```bash
   ./start_all.sh
   ```

3. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and Start Services**
   ```bash
   docker-compose up --build
   ```

2. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

### Individual Docker Containers

1. **Backend Container**
   ```bash
   cd backend
   docker build -t cvml-backend .
   docker run -p 8000:8000 cvml-backend
   ```

2. **Frontend Container**
   ```bash
   cd frontend
   docker build -t cvml-frontend .
   docker run -p 3000:3000 cvml-frontend
   ```

## Production Deployment

### Backend Deployment Options

#### Option 1: Railway
1. Connect your GitHub repository to Railway
2. Set environment variables:
   - `PYTHONPATH=/app`
   - `MODEL_PATH=/app/models`
3. Deploy automatically on push

#### Option 2: Heroku
1. Create `Procfile` in backend directory:
   ```
   web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```
2. Deploy using Heroku CLI:
   ```bash
   heroku create cvml-backend
   git push heroku main
   ```

#### Option 3: AWS EC2
1. Launch EC2 instance (Ubuntu 20.04)
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx
   ```
3. Setup application and configure nginx

### Frontend Deployment Options

#### Option 1: Vercel (Recommended)
1. Connect GitHub repository to Vercel
2. Set environment variables:
   - `NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.com`
3. Deploy automatically

#### Option 2: Netlify
1. Connect GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `.next`

#### Option 3: AWS S3 + CloudFront
1. Build the application: `npm run build`
2. Upload to S3 bucket
3. Configure CloudFront distribution

## Model Training and Deployment

### 1. Prepare Training Data

```bash
# Organize your data
mkdir -p data/kit_detection/{train,val,test}/{images,labels}
mkdir -p data/classification/{positive,negative,invalid}

# Add your images to appropriate directories
```

### 2. Train Object Detection Model

```bash
cd backend/models
python train_detection.py
```

### 3. Train Classification Model

```bash
cd backend/models
python train_classification.py
```

### 4. Deploy Models

```bash
# Copy trained models to production
cp models/kit_detection/weights/best.pt /path/to/production/models/
cp models/classification/best_model.h5 /path/to/production/models/
```

## Environment Configuration

### Backend Environment Variables

```bash
# .env file for backend
PYTHONPATH=/app
MODEL_PATH=/app/models
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
```

### Frontend Environment Variables

```bash
# .env.local file for frontend
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.com
NEXT_PUBLIC_APP_NAME=CVML Cardio Health Check Kit Analyzer
```

## Monitoring and Logging

### Health Checks

- Backend: `GET /health`
- Frontend: Built-in Next.js health check

### Logging

- Backend logs: Check application logs
- Frontend logs: Browser console and server logs

### Performance Monitoring

- Monitor API response times
- Track model inference times
- Monitor memory usage

## Security Considerations

### API Security
- Implement rate limiting
- Add authentication if needed
- Use HTTPS in production
- Validate input images

### Model Security
- Secure model files
- Implement model versioning
- Monitor for adversarial attacks

## Scaling Considerations

### Backend Scaling
- Use multiple worker processes
- Implement load balancing
- Consider model caching
- Use GPU acceleration for inference

### Frontend Scaling
- Use CDN for static assets
- Implement caching strategies
- Optimize bundle size

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model file paths
   - Verify model file integrity
   - Check available memory

2. **API Connection Issues**
   - Verify CORS configuration
   - Check network connectivity
   - Validate API endpoints

3. **Image Processing Errors**
   - Check image format support
   - Verify image size limits
   - Check OpenCV installation

### Debug Mode

```bash
# Backend debug mode
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --log-level debug

# Frontend debug mode
cd frontend
npm run dev -- --debug
```

## Backup and Recovery

### Model Backup
```bash
# Backup trained models
tar -czf models-backup.tar.gz models/
```

### Database Backup (if applicable)
```bash
# Backup any databases
pg_dump your_database > backup.sql
```

## Updates and Maintenance

### Updating Models
1. Train new models
2. Test thoroughly
3. Deploy with zero-downtime
4. Monitor performance

### Application Updates
1. Test in staging environment
2. Deploy during low-traffic periods
3. Monitor for issues
4. Rollback if necessary

## Support

For issues and questions:
- Check the logs first
- Review this documentation
- Create an issue in the repository
- Contact the development team
