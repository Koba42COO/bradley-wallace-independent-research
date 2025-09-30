# AIVA IDE - AI-Powered Integrated Development Environment

AIVA IDE is a modern, AI-powered development environment built with React, TypeScript, and Monaco Editor, featuring GPT integration for intelligent code assistance.

## Features

- **AI-Powered Code Completion**: Get intelligent code suggestions using GPT models
- **Advanced Code Editor**: Monaco Editor with syntax highlighting, auto-completion, and AI-assisted coding
- **Real-time AI Chat**: Interactive chat interface for coding questions and debugging help
- **File Explorer**: Navigate and manage your project files
- **Integrated Terminal**: Run commands and scripts directly in the IDE
- **WebSocket Collaboration**: Real-time collaborative editing capabilities
- **Dark/Light Theme Support**: Modern UI with customizable themes

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Monaco Editor
- **Backend**: Node.js, Express, Socket.IO
- **AI Integration**: OpenAI GPT API
- **UI Components**: Radix UI, Lucide Icons

## Prerequisites

- Node.js 18+
- npm or yarn
- OpenAI API key

## Installation

1. **Clone and setup the project**:
```bash
cd aiva_ide
npm install
```

2. **Install client dependencies**:
```bash
cd client
npm install
cd ..
```

3. **Install server dependencies**:
```bash
cd server
npm install
cd ..
```

4. **Environment Setup**:
Create a `.env` file in the server directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PORT=3001
NODE_ENV=development
```

## Running the Application

1. **Start the backend server**:
```bash
npm run server:dev
```

2. **Start the frontend (in a new terminal)**:
```bash
npm run client:dev
```

3. **Or run both together**:
```bash
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:3001

## Usage

### Code Editor
- Open files from the file explorer
- Use Ctrl+Space for AI code completion
- Click "AI Suggestions" to get code improvement recommendations
- Save files with the Save button
- Run code with the Run button

### AI Chat
- Ask coding questions in natural language
- Get debugging help and code explanations
- Request code refactoring suggestions

### File Management
- Browse project files in the explorer
- Create new files and folders
- Edit and save files

## API Endpoints

### GPT Integration
- `POST /api/chat` - Chat with GPT models
- `POST /api/complete` - Get code completions

### File System
- `GET /api/files` - List project files
- `GET /api/files/:path` - Read file content
- `POST /api/files/:path` - Write file content
- `POST /api/files` - Create new file
- `DELETE /api/files/:path` - Delete file

## WebSocket Events

- `join-room` - Join a collaborative editing session
- `code-change` - Broadcast code changes
- `cursor-move` - Share cursor positions

## Configuration

### Monaco Editor Settings
Customize editor behavior in `client/src/components/Editor.tsx`:
- Theme configuration
- Completion providers
- Keyboard shortcuts

### AI Model Settings
Configure GPT models and parameters in `server/index.ts`:
- Model selection (GPT-4, GPT-3.5-turbo)
- Temperature and token limits
- API rate limiting

## Development

### Project Structure
```
aiva_ide/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── lib/           # Utilities and API client
│   │   └── hooks/         # React hooks
├── server/                 # Node.js backend
│   ├── src/
│   │   ├── routes/        # API routes
│   │   └── services/      # Business logic
└── shared/                 # Shared types and schemas
```

### Adding New Features
1. Create components in `client/src/components/`
2. Add API endpoints in `server/src/routes/`
3. Update types in `shared/` directory
4. Add tests for new functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Join our Discord community

---

Built with ❤️ using modern web technologies and AI
