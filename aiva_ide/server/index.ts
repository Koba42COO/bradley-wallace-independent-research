import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// GPT API Routes
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model = 'gpt-4' } = req.body;

    // Import OpenAI dynamically to avoid issues if not installed
    const { OpenAI } = await import('openai');

    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const completion = await openai.chat.completions.create({
      model,
      messages,
      max_tokens: 1000,
      temperature: 0.7,
    });

    const response = completion.choices[0]?.message?.content || 'No response generated';

    res.json({
      response,
      usage: completion.usage,
      model: completion.model
    });
  } catch (error) {
    console.error('GPT API Error:', error);
    res.status(500).json({
      error: 'Failed to get GPT response',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Code completion endpoint
app.post('/api/complete', async (req, res) => {
  try {
    const { code, language, context } = req.body;

    const { OpenAI } = await import('openai');
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const prompt = `Complete the following ${language} code. Provide only the completion without explanation:

${context ? `Context: ${context}\n\n` : ''}Code to complete:
${code}`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 500,
      temperature: 0.3,
      stop: ['\n\n', '```']
    });

    const completedCode = completion.choices[0]?.message?.content?.trim() || '';

    res.json({
      completion: completedCode,
      usage: completion.usage
    });
  } catch (error) {
    console.error('Code completion error:', error);
    res.status(500).json({
      error: 'Failed to complete code',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// File system operations (basic implementation)
app.get('/api/files', (req, res) => {
  // This would integrate with actual file system
  // For now, return mock data
  res.json({
    files: [
      { name: 'src', type: 'directory', path: 'src' },
      { name: 'package.json', type: 'file', path: 'package.json' },
      { name: 'README.md', type: 'file', path: 'README.md' }
    ]
  });
});

// WebSocket connection for real-time features
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('join-room', (roomId) => {
    socket.join(roomId);
    console.log(`Client ${socket.id} joined room ${roomId}`);
  });

  socket.on('code-change', (data) => {
    // Broadcast code changes to other clients in the same room
    socket.to(data.roomId).emit('code-update', data);
  });

  socket.on('cursor-move', (data) => {
    socket.to(data.roomId).emit('cursor-update', data);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Start server
server.listen(PORT, () => {
  console.log(`AIVA IDE Server running on port ${PORT}`);
  console.log(`WebSocket server ready for real-time collaboration`);
});
