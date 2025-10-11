# AIVA IDE - AI-Powered Development Environment

AIVA IDE is a comprehensive AI-powered development environment that combines real-time collaborative editing, AI-assisted coding, and intelligent file management in a modern web-based interface.

## Features

### 🤖 AI-Powered Development
- **GPT Integration**: Chat with AI assistants for code help and explanations
- **Code Completion**: AI-powered code completion and suggestions
- **Intelligent Assistance**: Context-aware coding support

### 🔄 Real-Time Collaboration
- **WebSocket Connections**: Real-time collaborative editing
- **Multi-User Support**: Multiple developers can work simultaneously
- **Live Cursor Tracking**: See where other users are editing
- **Room-Based Collaboration**: Organize work in different collaboration rooms

### 📁 Advanced File Management
- **Secure File Operations**: Safe read/write/create/delete operations
- **Project Structure**: Navigate and manage project files
- **File Security**: Sandboxed file access within project boundaries

### 🎨 Modern UI/UX
- **React-Based Interface**: Fast and responsive web interface
- **Code Editor**: Syntax-highlighted code editing
- **Split-Panel Layout**: Efficient workspace organization

## Architecture

```
AIVA IDE
├── Client (React + TypeScript)
│   ├── Real-time collaboration via WebSocket
│   ├── AI chat interface
│   └── File management UI
├── Server (Node.js + Express + Socket.IO)
│   ├── REST API for file operations
│   ├── WebSocket for real-time features
│   ├── OpenAI integration
│   └── Security middleware
└── Database Integration
    ├── Project file system access
    ├── User session management
    └── Collaboration state
```

## Prerequisites

- Node.js 16+
- npm or yarn
- OpenAI API key (for AI features)

## Installation

### 1. Install Server Dependencies
```bash
cd aiva_ide/server
npm install
```

### 2. Install Client Dependencies
```bash
cd aiva_ide/client
npm install
```

### 3. Environment Setup
Create a `.env` file in the server directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PORT=3001
NODE_ENV=development
```

## Running the Application

### Development Mode

1. **Start the Server:**
```bash
cd aiva_ide/server
npm run dev
```

2. **Start the Client (in a new terminal):**
```bash
cd aiva_ide/client
npm start
```

3. **Open your browser:**
Navigate to `http://localhost:3000`

### Production Mode

1. **Build the Client:**
```bash
cd aiva_ide/client
npm run build
```

2. **Start the Server:**
```bash
cd aiva_ide/server
npm start
```

## API Endpoints

### REST API
- `GET /api/health` - Health check
- `GET /api/files` - List files and directories
- `GET /api/files/*` - Read file content
- `POST /api/files/*` - Write file content
- `POST /api/files` - Create new file
- `DELETE /api/files/*` - Delete file
- `POST /api/chat` - AI chat completion
- `POST /api/complete` - Code completion

### WebSocket Events
- `join-room` - Join collaboration room
- `leave-room` - Leave collaboration room
- `code-change` - Real-time code changes
- `cursor-move` - Cursor position updates
- `user-joined` - User joined room
- `user-left` - User left room

## Usage

### File Management
1. **Browse Files**: Use the file explorer sidebar to navigate your project
2. **Edit Files**: Click on any file to open it in the editor
3. **Save Changes**: Click the save button to persist changes
4. **Create Files**: Use the file creation functionality

### AI Assistance
1. **Chat with AI**: Use the chat panel to ask questions about your code
2. **Code Completion**: Select code and request AI completion
3. **Contextual Help**: Get explanations and suggestions

### Collaboration
1. **Join Rooms**: Connect to collaboration rooms by room ID
2. **Real-time Editing**: See changes from other users instantly
3. **Cursor Tracking**: Follow other users' cursors in real-time

## Security Features

- **Path Sandboxing**: File operations are restricted to project root
- **Input Validation**: All API inputs are validated
- **Rate Limiting**: Built-in rate limiting for API calls
- **CORS Protection**: Cross-origin request protection
- **Helmet Security**: Security headers and middleware

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required for AI features
- `PORT` - Server port (default: 3001)
- `NODE_ENV` - Environment mode (development/production)

### Client Configuration
The client automatically connects to the server on port 3001. Update the API base URL in `src/lib/api.ts` if needed.

## Development

### Project Structure
```
aiva_ide/
├── client/                 # React frontend
│   ├── public/            # Static assets
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── lib/          # Utilities and API client
│   │   └── App.tsx       # Main application
│   └── package.json
├── server/                 # Node.js backend
│   ├── src/
│   │   └── index.js      # Server entry point
│   └── package.json
└── README.md
```

### Adding New Features
1. **Client Features**: Add components in `client/src/components/`
2. **Server Features**: Add routes in `server/src/index.js`
3. **API Extensions**: Update the API client in `client/src/lib/api.ts`

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**
- Ensure the server is running on port 3001
- Check firewall settings
- Verify CORS configuration

**AI Features Not Working**
- Verify OpenAI API key is set correctly
- Check API key permissions and credits
- Ensure network connectivity to OpenAI

**File Operations Failing**
- Check file permissions
- Ensure paths are within project root
- Verify server has read/write access

### Logs
- Server logs are output to console
- Client errors appear in browser console
- Check network tab for API request failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Wallace Quantum Resonance Framework research suite.

## Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

---

**Built with ❤️ by Bradley Wallace - Advancing AI-assisted development**
