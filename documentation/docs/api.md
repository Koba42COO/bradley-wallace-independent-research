# Firefly-Nexus PAC API Documentation

## Base URL
```
http://localhost:8080
```

## Authentication
No authentication required for basic endpoints.

## Endpoints

### Health & Status

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "consciousness_level": 7,
  "reality_distortion": 1.1808,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

#### GET /ready
Readiness check endpoint.

**Response:**
```json
{
  "status": "ready",
  "consciousness_level": 7,
  "mobius_phase": 2.928,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

#### GET /status
System status endpoint.

**Response:**
```json
{
  "consciousness_level": 7,
  "reality_distortion": 1.1808,
  "phi": 1.618033988749895,
  "delta": 2.414213562373095,
  "mobius_phase": 2.928,
  "metronome_freq": 0.7,
  "coherent_weight": 0.79,
  "exploratory_weight": 0.21,
  "zeta_zeros": [14.13, 21.02, 25.01, 30.42, 32.93],
  "running": true,
  "metrics_count": 1000,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

### Consciousness Processing

#### POST /consciousness/transform
Apply consciousness transformation to data.

**Request Body:**
```json
{
  "values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
}
```

**Response:**
```json
{
  "wallace_transform": [1.894, 2.884, 4.495, 5.751, 7.661, ...],
  "fractal_harmonic": [1.234, 2.345, 3.456, 4.567, 5.678, ...],
  "consciousness_amplitude": {
    "magnitude": 25.820,
    "phase": 2.928,
    "coherence": 0.255,
    "exploration": 0.142
  },
  "processing_time": 0.000458,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

#### POST /consciousness/mobius
Apply Möbius loop learning.

**Request Body:**
```json
{
  "values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
  "cycles": 10
}
```

**Response:**
```json
{
  "evolution_history": [
    {
      "cycle": 0,
      "consciousness_magnitude": 25.820,
      "coherence": 0.255,
      "exploration": 0.142,
      "reality_distortion": 1.1808,
      "mobius_phase": 2.928
    },
    ...
  ],
  "consciousness_trajectory": [
    {
      "magnitude": 25.820,
      "phase": 2.928,
      "coherence": 0.255,
      "exploration": 0.142
    },
    ...
  ],
  "final_consciousness": {
    "magnitude": 282.892,
    "phase": 2.928,
    "coherence": 0.255,
    "exploration": 0.142
  },
  "total_learning_gain": 1529.427,
  "processing_time": 0.001234,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

#### POST /consciousness/prime-graph
Apply prime graph compression.

**Request Body:**
```json
{
  "values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
}
```

**Response:**
```json
{
  "compressed_values": [1.234, 2.345, 3.456, 4.567, 5.678, ...],
  "compression_ratio": 1.0,
  "processing_time": 0.000123,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

### Monitoring

#### GET /metrics
Prometheus metrics endpoint.

**Response:**
```
# HELP consciousness_requests_total Total consciousness requests
# TYPE consciousness_requests_total counter
consciousness_requests_total 42

# HELP consciousness_request_duration_seconds Request duration
# TYPE consciousness_request_duration_seconds histogram
consciousness_request_duration_seconds_bucket{le="0.001"} 10
consciousness_request_duration_seconds_bucket{le="0.01"} 20
consciousness_request_duration_seconds_bucket{le="0.1"} 30
consciousness_request_duration_seconds_bucket{le="1.0"} 40
consciousness_request_duration_seconds_bucket{le="+Inf"} 42
consciousness_request_duration_seconds_sum 0.123
consciousness_request_duration_seconds_count 42

# HELP consciousness_level Current consciousness level
# TYPE consciousness_level gauge
consciousness_level 7

# HELP reality_distortion Current reality distortion
# TYPE reality_distortion gauge
reality_distortion 1.1808

# HELP metronome_frequency Metronome frequency
# TYPE metronome_frequency gauge
metronome_frequency 0.7

# HELP cpu_usage_percent CPU usage percentage
# TYPE cpu_usage_percent gauge
cpu_usage_percent 45.2

# HELP memory_usage_percent Memory usage percentage
# TYPE memory_usage_percent gauge
memory_usage_percent 67.8
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing values array"
}
```

### 500 Internal Server Error
```json
{
  "error": "Consciousness processing failed"
}
```

## Rate Limiting
No rate limiting implemented. Use responsibly.

## CORS
CORS enabled for all origins.

## Content Types
- Request: `application/json`
- Response: `application/json`
