/**
 * chAIos Server Configuration
 * ===========================
 * Centralized configuration for the Express server
 */

const os = require('os');
const path = require('path');

module.exports = {
  // Server Configuration
  server: {
    port: process.env.PORT || 4200,
    host: process.env.HOST || '0.0.0.0',
    environment: process.env.NODE_ENV || 'development'
  },

  // Backend API Configuration
  api: {
    baseUrl: process.env.API_BASE_URL || 'http://localhost:8000',
    timeout: parseInt(process.env.API_TIMEOUT) || 30000,
    retryAttempts: parseInt(process.env.API_RETRY_ATTEMPTS) || 3,
    retryDelay: parseInt(process.env.API_RETRY_DELAY) || 1000
  },

  // SSL Configuration
  ssl: {
    enabled: process.env.SSL_ENABLED === 'true',
    keyPath: process.env.SSL_KEY_PATH || './certs/key.pem',
    certPath: process.env.SSL_CERT_PATH || './certs/cert.pem'
  },

  // Performance Configuration
  performance: {
    clustering: {
      enabled: process.env.ENABLE_CLUSTERING === 'true',
      maxWorkers: parseInt(process.env.MAX_WORKERS) || os.cpus().length
    },
    compression: {
      enabled: process.env.ENABLE_COMPRESSION !== 'false',
      level: parseInt(process.env.COMPRESSION_LEVEL) || 6,
      threshold: parseInt(process.env.COMPRESSION_THRESHOLD) || 1024
    },
    caching: {
      enabled: process.env.ENABLE_CACHING !== 'false',
      staticMaxAge: process.env.STATIC_CACHE_MAX_AGE || '1y',
      htmlMaxAge: process.env.HTML_CACHE_MAX_AGE || '1h'
    }
  },

  // Security Configuration
  security: {
    cors: {
      origin: process.env.CORS_ORIGIN ? 
        process.env.CORS_ORIGIN.split(',') : 
        ['http://localhost:4200', 'http://localhost:8100'],
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin']
    },
    rateLimit: {
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW) || 15 * 60 * 1000, // 15 minutes
      max: parseInt(process.env.RATE_LIMIT_MAX) || 1000,
      message: {
        error: 'Too many requests from this IP, please try again later.',
        code: 'RATE_LIMIT_EXCEEDED'
      }
    },
    helmet: {
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'", "https://unpkg.com", "https://fonts.googleapis.com"],
          scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
          fontSrc: ["'self'", "https://fonts.gstatic.com", "data:"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "ws:", "wss:", process.env.API_BASE_URL || 'http://localhost:8000']
        }
      },
      crossOriginEmbedderPolicy: false
    }
  },

  // Feature Flags
  features: {
    apiProxy: process.env.ENABLE_API_PROXY !== 'false',
    websockets: process.env.ENABLE_WEBSOCKETS !== 'false',
    healthCheck: process.env.ENABLE_HEALTH_CHECK !== 'false',
    metrics: process.env.ENABLE_METRICS !== 'false',
    hotReload: process.env.HOT_RELOAD !== 'false' && process.env.NODE_ENV === 'development'
  },

  // Paths Configuration
  paths: {
    dist: path.join(__dirname, 'dist', 'app'),
    assets: path.join(__dirname, 'src', 'assets'),
    certs: path.join(__dirname, 'certs'),
    logs: path.join(__dirname, 'logs')
  },

  // Logging Configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.LOG_FORMAT || 'combined',
    enableConsole: process.env.ENABLE_CONSOLE_LOG !== 'false',
    enableFile: process.env.ENABLE_FILE_LOG === 'true',
    maxSize: process.env.LOG_MAX_SIZE || '10m',
    maxFiles: process.env.LOG_MAX_FILES || '5'
  },

  // WebSocket Configuration
  websocket: {
    enabled: process.env.ENABLE_WEBSOCKETS !== 'false',
    heartbeatInterval: parseInt(process.env.WS_HEARTBEAT_INTERVAL) || 30000,
    maxConnections: parseInt(process.env.WS_MAX_CONNECTIONS) || 1000,
    messageMaxSize: parseInt(process.env.WS_MESSAGE_MAX_SIZE) || 1024 * 1024 // 1MB
  },

  // Health Check Configuration
  healthCheck: {
    enabled: process.env.ENABLE_HEALTH_CHECK !== 'false',
    interval: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 30000,
    timeout: parseInt(process.env.HEALTH_CHECK_TIMEOUT) || 5000,
    endpoints: [
      {
        name: 'api',
        url: `${process.env.API_BASE_URL || 'http://localhost:8000'}/health`,
        timeout: 5000
      },
      {
        name: 'database',
        url: `${process.env.API_BASE_URL || 'http://localhost:8000'}/health/database`,
        timeout: 3000
      }
    ]
  },

  // Development Configuration
  development: {
    hotReload: process.env.HOT_RELOAD !== 'false',
    debugMode: process.env.DEBUG_MODE === 'true',
    mockApi: process.env.MOCK_API === 'true',
    verboseLogging: process.env.VERBOSE_LOGGING === 'true'
  },

  // Production Configuration
  production: {
    enableCompression: true,
    enableCaching: true,
    enableClustering: true,
    enableSSL: process.env.SSL_ENABLED === 'true',
    enableMetrics: true,
    enableHealthChecks: true
  }
};
