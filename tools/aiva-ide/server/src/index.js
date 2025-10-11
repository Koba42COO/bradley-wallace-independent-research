const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs-extra');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:5173"],
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Constants
const PORT = process.env.PORT || 3001;
const PROJECT_ROOT = path.join(__dirname, '../../../'); // Points to dev directory

// Store active rooms for collaboration
const activeRooms = new Map();

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('join-room', (roomId) => {
    socket.join(roomId);
    console.log(`User ${socket.id} joined room ${roomId}`);

    // Track room membership
    if (!activeRooms.has(roomId)) {
      activeRooms.set(roomId, new Set());
    }
    activeRooms.get(roomId).add(socket.id);

    // Notify others in room
    socket.to(roomId).emit('user-joined', { userId: socket.id });
  });

  socket.on('leave-room', (roomId) => {
    socket.leave(roomId);
    console.log(`User ${socket.id} left room ${roomId}`);

    // Remove from tracking
    if (activeRooms.has(roomId)) {
      activeRooms.get(roomId).delete(socket.id);
      if (activeRooms.get(roomId).size === 0) {
        activeRooms.delete(roomId);
      }
    }

    socket.to(roomId).emit('user-left', { userId: socket.id });
  });

  socket.on('code-change', (data) => {
    // Broadcast code changes to all users in the same room
    socket.to(data.roomId).emit('code-change', {
      ...data,
      userId: socket.id,
      timestamp: new Date().toISOString()
    });
  });

  socket.on('cursor-move', (data) => {
    // Broadcast cursor movements for real-time collaboration
    socket.to(data.roomId).emit('cursor-move', {
      ...data,
      userId: socket.id
    });
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);

    // Clean up room memberships
    for (const [roomId, users] of activeRooms.entries()) {
      if (users.has(socket.id)) {
        users.delete(socket.id);
        socket.to(roomId).emit('user-left', { userId: socket.id });

        if (users.size === 0) {
          activeRooms.delete(roomId);
        }
      }
    }
  });
});

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'aiva-ide-server',
    version: '1.0.0',
    websocket_clients: io.engine.clientsCount,
    active_rooms: activeRooms.size,
    timestamp: new Date().toISOString()
  });
});

// GPT Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model = 'gpt-4' } = req.body;

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: 'OpenAI API key not configured',
        message: 'Please set OPENAI_API_KEY environment variable'
      });
    }

    const { OpenAI } = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const completion = await openai.chat.completions.create({
      model: model,
      messages: messages,
      temperature: 0.7,
      max_tokens: 2000
    });

    res.json({
      message: completion.choices[0].message.content,
      usage: completion.usage,
      model: completion.model
    });

  } catch (error) {
    console.error('GPT Chat error:', error);
    res.status(500).json({
      error: 'Failed to process chat request',
      details: error.message
    });
  }
});

// Code completion endpoint
app.post('/api/complete', async (req, res) => {
  try {
    const { code, language, context } = req.body;

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: 'OpenAI API key not configured'
      });
    }

    const { OpenAI } = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const prompt = `Complete the following ${language} code:\n\n${context ? `Context: ${context}\n\n` : ''}Code:\n${code}\n\nCompletion:`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.3,
      max_tokens: 1000,
      stop: ['\n\n', '```']
    });

    res.json({
      completion: completion.choices[0].message.content.trim(),
      usage: completion.usage
    });

  } catch (error) {
    console.error('Code completion error:', error);
    res.status(500).json({
      error: 'Failed to complete code',
      details: error.message
    });
  }
});

// File system operations

// Get files/directories
app.get('/api/files', async (req, res) => {
  try {
    const { path: requestedPath = '' } = req.query;
    const fullPath = path.join(PROJECT_ROOT, requestedPath);

    // Security check - ensure we're within project root
    if (!fullPath.startsWith(PROJECT_ROOT)) {
      return res.status(403).json({ error: 'Access denied: outside project root' });
    }

    const stats = await fs.stat(fullPath);
    const isDirectory = stats.isDirectory();

    if (isDirectory) {
      const items = await fs.readdir(fullPath);
      const files = [];

      for (const item of items) {
        try {
          const itemPath = path.join(fullPath, item);
          const itemStats = await fs.stat(itemPath);

          files.push({
            name: item,
            path: path.relative(PROJECT_ROOT, itemPath),
            type: itemStats.isDirectory() ? 'directory' : 'file',
            size: itemStats.size,
            modified: itemStats.mtime.toISOString()
          });
        } catch (error) {
          // Skip inaccessible files
          continue;
        }
      }

      res.json({ files, path: requestedPath });
    } else {
      res.status(400).json({ error: 'Path is not a directory' });
    }

  } catch (error) {
    console.error('Files listing error:', error);
    res.status(500).json({
      error: 'Failed to list files',
      details: error.message
    });
  }
});

// Read file
app.get('/api/files/*', async (req, res) => {
  try {
    const filePath = req.params[0];
    const fullPath = path.join(PROJECT_ROOT, filePath);

    // Security check
    if (!fullPath.startsWith(PROJECT_ROOT)) {
      return res.status(403).json({ error: 'Access denied: outside project root' });
    }

    const stats = await fs.stat(fullPath);
    if (!stats.isFile()) {
      return res.status(400).json({ error: 'Path is not a file' });
    }

    // Check file size (limit to 10MB)
    if (stats.size > 10 * 1024 * 1024) {
      return res.status(413).json({ error: 'File too large (>10MB)' });
    }

    const content = await fs.readFile(fullPath, 'utf8');

    res.json({
      path: filePath,
      content: content,
      size: stats.size,
      modified: stats.mtime.toISOString()
    });

  } catch (error) {
    console.error('File read error:', error);
    res.status(500).json({
      error: 'Failed to read file',
      details: error.message
    });
  }
});

// Write file
app.post('/api/files/*', async (req, res) => {
  try {
    const filePath = req.params[0];
    const { content } = req.body;

    if (!content) {
      return res.status(400).json({ error: 'Content is required' });
    }

    const fullPath = path.join(PROJECT_ROOT, filePath);

    // Security check
    if (!fullPath.startsWith(PROJECT_ROOT)) {
      return res.status(403).json({ error: 'Access denied: outside project root' });
    }

    // Ensure directory exists
    await fs.ensureDir(path.dirname(fullPath));

    // Write file
    await fs.writeFile(fullPath, content, 'utf8');

    // Get updated stats
    const stats = await fs.stat(fullPath);

    res.json({
      success: true,
      path: filePath,
      size: stats.size,
      modified: stats.mtime.toISOString()
    });

  } catch (error) {
    console.error('File write error:', error);
    res.status(500).json({
      error: 'Failed to write file',
      details: error.message
    });
  }
});

// Create file
app.post('/api/files', async (req, res) => {
  try {
    const { path: filePath, content = '' } = req.body;

    if (!filePath) {
      return res.status(400).json({ error: 'Path is required' });
    }

    const fullPath = path.join(PROJECT_ROOT, filePath);

    // Security check
    if (!fullPath.startsWith(PROJECT_ROOT)) {
      return res.status(403).json({ error: 'Access denied: outside project root' });
    }

    // Check if file already exists
    const exists = await fs.pathExists(fullPath);
    if (exists) {
      return res.status(409).json({ error: 'File already exists' });
    }

    // Ensure directory exists
    await fs.ensureDir(path.dirname(fullPath));

    // Create file
    await fs.writeFile(fullPath, content, 'utf8');

    const stats = await fs.stat(fullPath);

    res.json({
      success: true,
      path: filePath,
      size: stats.size,
      created: stats.birthtime.toISOString()
    });

  } catch (error) {
    console.error('File create error:', error);
    res.status(500).json({
      error: 'Failed to create file',
      details: error.message
    });
  }
});

// Delete file
app.delete('/api/files/*', async (req, res) => {
  try {
    const filePath = req.params[0];
    const fullPath = path.join(PROJECT_ROOT, filePath);

    // Security check
    if (!fullPath.startsWith(PROJECT_ROOT)) {
      return res.status(403).json({ error: 'Access denied: outside project root' });
    }

    // Check if file exists
    const exists = await fs.pathExists(fullPath);
    if (!exists) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Delete file
    await fs.remove(fullPath);

    res.json({
      success: true,
      path: filePath,
      deleted: new Date().toISOString()
    });

  } catch (error) {
    console.error('File delete error:', error);
    res.status(500).json({
      error: 'Failed to delete file',
      details: error.message
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    details: process.env.NODE_ENV === 'development' ? error.message : undefined
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Start server
server.listen(PORT, () => {
  console.log(`🚀 AIVA IDE Server running on port ${PORT}`);
  console.log(`📁 Project root: ${PROJECT_ROOT}`);
  console.log(`🔗 WebSocket enabled for real-time collaboration`);
  console.log(`🤖 OpenAI integration: ${process.env.OPENAI_API_KEY ? 'Enabled' : 'Disabled'}`);
  console.log(`🌐 Health check: http://localhost:${PORT}/api/health`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

module.exports = app;
