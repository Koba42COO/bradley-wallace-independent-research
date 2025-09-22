#!/usr/bin/env node
/**
 * chAIos Express Server
 * ====================
 * Node.js Express server for serving Angular frontend and API orchestration
 * Integrates with FastAPI backend and provides development/production serving
 */

const express = require('express');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const { createProxyMiddleware } = require('http-proxy-middleware');
const fs = require('fs');
const https = require('https');
const http = require('http');
const WebSocket = require('ws');
const cluster = require('cluster');
const os = require('os');

// Configuration
const CONFIG = {
  // Server Configuration
  PORT: process.env.PORT || 4200,
  HOST: process.env.HOST || '0.0.0.0',
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // Backend API Configuration
  API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:8000',
  API_TIMEOUT: process.env.API_TIMEOUT || 30000,
  
  // SSL Configuration
  SSL_ENABLED: process.env.SSL_ENABLED === 'true',
  SSL_KEY_PATH: process.env.SSL_KEY_PATH || './certs/key.pem',
  SSL_CERT_PATH: process.env.SSL_CERT_PATH || './certs/cert.pem',
  
  // Performance Configuration
  ENABLE_CLUSTERING: process.env.ENABLE_CLUSTERING === 'true',
  MAX_WORKERS: process.env.MAX_WORKERS || os.cpus().length,
  ENABLE_COMPRESSION: process.env.ENABLE_COMPRESSION !== 'false',
  ENABLE_CACHING: process.env.ENABLE_CACHING !== 'false',
  
  // Security Configuration
  CORS_ORIGIN: process.env.CORS_ORIGIN || ['http://localhost:4200', 'http://localhost:8100'],
  RATE_LIMIT_WINDOW: process.env.RATE_LIMIT_WINDOW || 15 * 60 * 1000, // 15 minutes
  RATE_LIMIT_MAX: process.env.RATE_LIMIT_MAX || 1000,
  
  // Paths
  DIST_PATH: path.join(__dirname, 'dist', 'app'),
  ASSETS_PATH: path.join(__dirname, 'src', 'assets'),
  
  // Feature Flags
  ENABLE_API_PROXY: process.env.ENABLE_API_PROXY !== 'false',
  ENABLE_WEBSOCKETS: process.env.ENABLE_WEBSOCKETS !== 'false',
  ENABLE_HEALTH_CHECK: process.env.ENABLE_HEALTH_CHECK !== 'false',
  ENABLE_METRICS: process.env.ENABLE_METRICS !== 'false'
};

// Clustering for production
if (CONFIG.ENABLE_CLUSTERING && cluster.isMaster && CONFIG.NODE_ENV === 'production') {
  console.log(`🚀 chAIos Master Process ${process.pid} starting...`);
  console.log(`📊 Spawning ${CONFIG.MAX_WORKERS} worker processes`);
  
  // Fork workers
  for (let i = 0; i < CONFIG.MAX_WORKERS; i++) {
    cluster.fork();
  }
  
  cluster.on('exit', (worker, code, signal) => {
    console.log(`💀 Worker ${worker.process.pid} died. Spawning replacement...`);
    cluster.fork();
  });
  
  return;
}

// Express Application Setup
const app = express();

// Trust proxy for load balancers
app.set('trust proxy', 1);

// Security Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://unpkg.com", "https://fonts.googleapis.com"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      fontSrc: ["'self'", "https://fonts.gstatic.com", "data:"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:", CONFIG.API_BASE_URL]
    }
  },
  crossOriginEmbedderPolicy: false
}));

// CORS Configuration
app.use(cors({
  origin: CONFIG.CORS_ORIGIN,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin']
}));

// Compression
if (CONFIG.ENABLE_COMPRESSION) {
  app.use(compression({
    filter: (req, res) => {
      if (req.headers['x-no-compression']) return false;
      return compression.filter(req, res);
    },
    level: 6,
    threshold: 1024
  }));
}

// Request Logging
app.use((req, res, next) => {
  const start = Date.now();
  const timestamp = new Date().toISOString();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const status = res.statusCode;
    const method = req.method;
    const url = req.url;
    const userAgent = req.get('User-Agent') || 'Unknown';
    
    // Color coding for status
    const statusColor = status >= 500 ? '\x1b[31m' : status >= 400 ? '\x1b[33m' : status >= 300 ? '\x1b[36m' : '\x1b[32m';
    const resetColor = '\x1b[0m';
    
    console.log(`[${timestamp}] ${method} ${url} ${statusColor}${status}${resetColor} ${duration}ms - ${userAgent}`);
  });
  
  next();
});

// Body Parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate Limiting
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({
  windowMs: CONFIG.RATE_LIMIT_WINDOW,
  max: CONFIG.RATE_LIMIT_MAX,
  message: {
    error: 'Too many requests from this IP, please try again later.',
    code: 'RATE_LIMIT_EXCEEDED'
  },
  standardHeaders: true,
  legacyHeaders: false
});
app.use('/api/', limiter);

// Health Check Endpoint
if (CONFIG.ENABLE_HEALTH_CHECK) {
  app.get('/health', (req, res) => {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      pid: process.pid,
      version: require('./package.json').version,
      environment: CONFIG.NODE_ENV,
      services: {
        frontend: 'operational',
        api_proxy: CONFIG.ENABLE_API_PROXY ? 'operational' : 'disabled',
        websockets: CONFIG.ENABLE_WEBSOCKETS ? 'operational' : 'disabled'
      }
    };
    
    res.status(200).json(health);
  });
}

// Metrics Endpoint
if (CONFIG.ENABLE_METRICS) {
  app.get('/metrics', (req, res) => {
    const metrics = {
      timestamp: new Date().toISOString(),
      system: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage(),
        pid: process.pid,
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version
      },
      server: {
        port: CONFIG.PORT,
        environment: CONFIG.NODE_ENV,
        clustering: CONFIG.ENABLE_CLUSTERING,
        compression: CONFIG.ENABLE_COMPRESSION,
        ssl: CONFIG.SSL_ENABLED
      }
    };
    
    res.status(200).json(metrics);
  });
}

// API Proxy Middleware
if (CONFIG.ENABLE_API_PROXY) {
  const apiProxy = createProxyMiddleware({
    target: CONFIG.API_BASE_URL,
    changeOrigin: true,
    timeout: CONFIG.API_TIMEOUT,
    pathRewrite: {
      '^/api': '' // Remove /api prefix when forwarding to backend
    },
    onProxyReq: (proxyReq, req, res) => {
      // Add custom headers
      proxyReq.setHeader('X-Forwarded-For', req.ip);
      proxyReq.setHeader('X-Forwarded-Proto', req.protocol);
      proxyReq.setHeader('X-Frontend-Server', 'chAIos-Express');
      
      console.log(`🔄 Proxying: ${req.method} ${req.url} -> ${CONFIG.API_BASE_URL}${req.url.replace('/api', '')}`);
    },
    onProxyRes: (proxyRes, req, res) => {
      // Add CORS headers to proxied responses
      proxyRes.headers['Access-Control-Allow-Origin'] = '*';
      proxyRes.headers['Access-Control-Allow-Credentials'] = 'true';
    },
    onError: (err, req, res) => {
      console.error('❌ Proxy Error:', err.message);
      res.status(500).json({
        error: 'Backend service unavailable',
        message: 'The API server is currently unavailable. Please try again later.',
        timestamp: new Date().toISOString()
      });
    }
  });
  
  app.use('/api', apiProxy);
}

// Static File Serving
if (CONFIG.ENABLE_CACHING) {
  // Cache static assets
  app.use('/assets', express.static(path.join(CONFIG.DIST_PATH, 'assets'), {
    maxAge: '1y',
    etag: true,
    lastModified: true,
    setHeaders: (res, path) => {
      if (path.endsWith('.js') || path.endsWith('.css')) {
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      }
    }
  }));
  
  // Cache other static files
  app.use(express.static(CONFIG.DIST_PATH, {
    maxAge: '1h',
    etag: true,
    lastModified: true,
    index: false // Don't serve index.html here
  }));
} else {
  app.use(express.static(CONFIG.DIST_PATH, { index: false }));
}

// Development Hot Reload Support
if (CONFIG.NODE_ENV === 'development') {
  app.get('/dev-reload', (req, res) => {
    res.json({ reload: true, timestamp: Date.now() });
  });
}

// Angular SPA Routing - Catch all handler
app.get('*', (req, res) => {
  const indexPath = path.join(CONFIG.DIST_PATH, 'index.html');
  
  if (fs.existsSync(indexPath)) {
    // Inject environment variables into index.html
    let html = fs.readFileSync(indexPath, 'utf8');
    
    // Replace environment placeholders
    html = html.replace('{{API_BASE_URL}}', CONFIG.API_BASE_URL);
    html = html.replace('{{NODE_ENV}}', CONFIG.NODE_ENV);
    html = html.replace('{{VERSION}}', require('./package.json').version);
    
    res.setHeader('Content-Type', 'text/html');
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.send(html);
  } else {
    res.status(404).json({
      error: 'Frontend not built',
      message: 'Please run "npm run build" to build the frontend first.',
      timestamp: new Date().toISOString()
    });
  }
});

// Error Handling Middleware
app.use((err, req, res, next) => {
  console.error('❌ Server Error:', err);
  
  const isDevelopment = CONFIG.NODE_ENV === 'development';
  
  res.status(err.status || 500).json({
    error: err.message || 'Internal Server Error',
    ...(isDevelopment && { stack: err.stack }),
    timestamp: new Date().toISOString(),
    path: req.path,
    method: req.method
  });
});

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `The requested resource ${req.path} was not found.`,
    timestamp: new Date().toISOString()
  });
});

// Server Creation and Startup
function createServer() {
  if (CONFIG.SSL_ENABLED && fs.existsSync(CONFIG.SSL_KEY_PATH) && fs.existsSync(CONFIG.SSL_CERT_PATH)) {
    const options = {
      key: fs.readFileSync(CONFIG.SSL_KEY_PATH),
      cert: fs.readFileSync(CONFIG.SSL_CERT_PATH)
    };
    return https.createServer(options, app);
  } else {
    return http.createServer(app);
  }
}

const server = createServer();

// WebSocket Support
if (CONFIG.ENABLE_WEBSOCKETS) {
  const wss = new WebSocket.Server({ server });
  
  wss.on('connection', (ws, req) => {
    const clientIP = req.socket.remoteAddress;
    console.log(`🔌 WebSocket connected: ${clientIP}`);
    
    ws.on('message', (message) => {
      try {
        const data = JSON.parse(message);
        console.log('📨 WebSocket message:', data);
        
        // Echo back for now - implement your WebSocket logic here
        ws.send(JSON.stringify({
          type: 'echo',
          data: data,
          timestamp: new Date().toISOString()
        }));
      } catch (error) {
        console.error('❌ WebSocket message error:', error);
        ws.send(JSON.stringify({
          type: 'error',
          message: 'Invalid message format',
          timestamp: new Date().toISOString()
        }));
      }
    });
    
    ws.on('close', () => {
      console.log(`🔌 WebSocket disconnected: ${clientIP}`);
    });
    
    ws.on('error', (error) => {
      console.error('❌ WebSocket error:', error);
    });
    
    // Send welcome message
    ws.send(JSON.stringify({
      type: 'welcome',
      message: 'Connected to chAIos WebSocket server',
      timestamp: new Date().toISOString()
    }));
  });
}

// Graceful Shutdown
process.on('SIGTERM', () => {
  console.log('🛑 SIGTERM received. Starting graceful shutdown...');
  server.close(() => {
    console.log('✅ Server closed successfully');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('\n🛑 SIGINT received. Starting graceful shutdown...');
  server.close(() => {
    console.log('✅ Server closed successfully');
    process.exit(0);
  });
});

// Start Server
server.listen(CONFIG.PORT, CONFIG.HOST, () => {
  const protocol = CONFIG.SSL_ENABLED ? 'https' : 'http';
  const workerInfo = CONFIG.ENABLE_CLUSTERING ? ` (Worker ${process.pid})` : '';
  
  console.log('');
  console.log('🤖 ===================================');
  console.log('🚀 chAIos Express Server Started');
  console.log('📐 Chiral Harmonic Intelligence');
  console.log('===================================');
  console.log(`🌐 Server: ${protocol}://${CONFIG.HOST}:${CONFIG.PORT}${workerInfo}`);
  console.log(`📁 Serving: ${CONFIG.DIST_PATH}`);
  console.log(`🔄 API Proxy: ${CONFIG.ENABLE_API_PROXY ? CONFIG.API_BASE_URL : 'Disabled'}`);
  console.log(`🔌 WebSockets: ${CONFIG.ENABLE_WEBSOCKETS ? 'Enabled' : 'Disabled'}`);
  console.log(`🛡️  Security: ${CONFIG.SSL_ENABLED ? 'HTTPS' : 'HTTP'}`);
  console.log(`🗜️  Compression: ${CONFIG.ENABLE_COMPRESSION ? 'Enabled' : 'Disabled'}`);
  console.log(`⚡ Environment: ${CONFIG.NODE_ENV}`);
  console.log(`📊 Clustering: ${CONFIG.ENABLE_CLUSTERING ? 'Enabled' : 'Disabled'}`);
  console.log('===================================');
  console.log('');
  
  if (CONFIG.NODE_ENV === 'development') {
    console.log('🔍 Development endpoints:');
    console.log(`   Health: ${protocol}://${CONFIG.HOST}:${CONFIG.PORT}/health`);
    console.log(`   Metrics: ${protocol}://${CONFIG.HOST}:${CONFIG.PORT}/metrics`);
    console.log(`   Hot Reload: ${protocol}://${CONFIG.HOST}:${CONFIG.PORT}/dev-reload`);
    console.log('');
  }
});

module.exports = app;
