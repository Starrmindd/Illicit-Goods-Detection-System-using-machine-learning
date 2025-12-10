# AI-Powered X-ray Contraband Detection System

An end-to-end AI system for detecting contraband and illicit goods in X-ray scans using computer vision and deep learning.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   X-ray Images  │───▶│  Flask REST API │───▶│  Operator UI    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Training │    │  AI Detection   │    │   Audit Logs   │
│    Pipeline     │    │     Engine      │    │   & Metrics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

- **Real-time Detection**: <500ms inference latency on GPU
- **Multi-class Support**: Firearms, knives, explosives, drugs, electronics
- **High Accuracy**: Target mAP ≥ 0.90 for critical classes
- **Secure API**: JWT authentication, HTTPS, audit logging
- **Operator Dashboard**: Real-time monitoring and threat assessment
- **Retraining Pipeline**: Continuous model improvement
- **Production Ready**: Docker + Kubernetes deployment

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd xray-detection-system

# Start with Docker Compose
docker-compose up -d

# Access dashboard
open http://localhost:3000
```

## Components

- `/data` - Dataset management and preprocessing
- `/training` - Model training and evaluation
- `/inference` - Model optimization and ONNX export
- `/api` - Flask REST API service
- `/dashboard` - React operator interface
- `/ops` - Deployment configurations

## Performance Targets

- **Latency**: ≤500ms per image (GPU), ≤100ms optimized
- **Throughput**: 32 images/batch with linear scaling
- **Accuracy**: mAP ≥0.90 for high-value threat classes

## Security & Compliance

- HTTPS with JWT authentication
- Role-based access control (RBAC)
- Comprehensive audit logging
- Data encryption at rest
- Configurable retention policies

## ENJOY ;)