import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { createServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import mongoose from 'mongoose';
import redis from 'redis';
import winston from 'winston';

// Import routes
import healthRoutes from './routes/health';
import optimizationRoutes from './routes/optimization';
import dashboardRoutes from './routes/dashboard';
import systemRoutes from './routes/system';

// Environment configuration
const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/cudnt';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// Initialize Express app
const app = express();
const server = createServer(app);

// Initialize Socket.IO
const io = new SocketServer(server, {
  cors: {
    origin: ["http://localhost:4200", "http://localhost:8100"],
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: ["http://localhost:4200", "http://localhost:8100"],
  credentials: true
}));
app.use(compression());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// Logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Database connections
async function connectDatabases() {
  try {
    // MongoDB connection
    await mongoose.connect(MONGODB_URI);
    logger.info('Connected to MongoDB');

    // Redis connection
    const redisClient = redis.createClient({ url: REDIS_URL });
    await redisClient.connect();
    logger.info('Connected to Redis');

    return { mongo: mongoose.connection, redis: redisClient };
  } catch (error) {
    logger.error('Database connection failed:', error);
    throw error;
  }
}

// Socket.IO connection handling
io.on('connection', (socket) => {
  logger.info(`Client connected: ${socket.id}`);

  socket.on('disconnect', () => {
    logger.info(`Client disconnected: ${socket.id}`);
  });

  // Real-time optimization updates
  socket.on('subscribe_optimization', (optimizationId) => {
    socket.join(`optimization_${optimizationId}`);
  });

  socket.on('unsubscribe_optimization', (optimizationId) => {
    socket.leave(`optimization_${optimizationId}`);
  });
});

// Routes
app.use('/api/health', healthRoutes);
app.use('/api/optimize', optimizationRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/status', systemRoutes);

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req: express.Request, res: express.Response) => {
  res.status(404).json({
    success: false,
    message: 'API endpoint not found'
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

// Start server
async function startServer() {
  try {
    // Connect to databases
    const dbConnections = await connectDatabases();

    // Start HTTP server with Socket.IO
    server.listen(PORT, () => {
      logger.info(`🚀 CUDNT Backend Server running on port ${PORT}`);
      logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`🔗 WebSocket endpoint: ws://localhost:${PORT}`);
      logger.info(`📱 Frontend URL: http://localhost:4200`);
      logger.info(`📱 Mobile App URL: http://localhost:8100`);
    });

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Export for testing
export { app, server, io, logger };

// Start server if this file is run directly
if (require.main === module) {
  startServer();
}
