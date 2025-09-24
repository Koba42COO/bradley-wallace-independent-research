# API Reference

## Overview
The chAIos platform provides a comprehensive REST API for accessing all platform features.

## Base URL
```
http://localhost:8000
```

## Authentication
All API requests require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Gateway Endpoints
- `GET /health` - Health check
- `GET /status` - System status
- `GET /metrics` - Performance metrics
- `GET /search` - Unified search
- `POST /query` - Polymath query

### Knowledge Endpoints
- `GET /knowledge/search` - Search knowledge base
- `GET /knowledge/stats` - Knowledge statistics
- `POST /knowledge/add` - Add new knowledge

### AI Endpoints
- `POST /ai/process` - AI processing
- `GET /ai/models` - Available models

## Data Models

### QueryRequest
```json
{
  "query": "search query",
  "domain": "knowledge domain (optional)",
  "limit": 10
}
```

### KnowledgeDocument
```json
{
  "id": "document_id",
  "title": "Document Title",
  "content": "Document content...",
  "domain": "knowledge_domain",
  "score": 0.95
}
```
