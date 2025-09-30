# AIVA Local Assistant - Cursor Extension

A VS Code/Cursor extension that integrates with your local AIVA node for private, secure AI-assisted coding.

## Features

- **🤖 Local AI Chat**: Chat with your private AI assistant running on your machine
- **✨ Inline Code Completion**: Get AI-powered code suggestions as you type
- **🔍 Code Analysis**: Explain, refactor, optimize, and debug code
- **📊 Project Analysis**: Get insights about your project structure
- **🔒 Privacy First**: All AI processing happens locally - no data leaves your machine

## Prerequisites

1. **AIVA Local Node**: You need a running AIVA local node server
   ```bash
   # From the aiva_local_node directory
   ./scripts/setup.sh
   ./start.sh
   ```

2. **Cursor IDE**: This extension is designed for Cursor IDE

## Installation

### Option 1: From VS Code Marketplace (Recommended)
1. Open Cursor IDE
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "AIVA Local Assistant"
4. Click Install

### Option 2: From Source
1. Clone this repository
2. Run `npm install`
3. Run `npm run compile`
4. Press F5 to launch extension development host
5. In the new window, test the extension

## Configuration

After installation, configure the extension:

1. Open Settings (Ctrl+,)
2. Search for "AIVA"
3. Set your AIVA server URL (default: `http://localhost:8000`)

### Available Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiva.server.url` | `http://localhost:8000` | URL of your AIVA inference server |
| `aiva.model.name` | `mixtral-8x7b-instruct` | Model name to use |
| `aiva.completion.maxTokens` | `512` | Maximum tokens for code completion |
| `aiva.chat.maxTokens` | `1024` | Maximum tokens for chat responses |
| `aiva.temperature` | `0.7` | Sampling temperature |
| `aiva.enableInlineSuggestions` | `true` | Enable automatic inline suggestions |

## Usage

### Chat Interface
1. Open the AIVA Chat panel (Ctrl+Shift+C or click the status bar icon)
2. Type your questions or requests
3. Get AI assistance with coding problems, explanations, and more

### Inline Code Completion
- Just start typing - AIVA will suggest completions automatically
- Press `Ctrl+Shift+A` to manually trigger completion
- Works in all supported languages

### Code Commands
Select code and use these commands:

- **Explain Code** (Right-click → AIVA: Explain Code): Get detailed explanations
- **Refactor Code** (Right-click → AIVA: Refactor Code): Improve code structure
- **Optimize Code** (Right-click → AIVA: Optimize Code): Performance improvements
- **Debug Code** (Right-click → AIVA: Debug Code): Find and fix issues

### Project Analysis
- Right-click on a folder in the explorer
- Select "AIVA: Analyze Project" for insights about your codebase

## Supported Languages

The extension works with all programming languages supported by your AIVA model, including:

- Python, JavaScript, TypeScript
- Java, C#, C++, Go, Rust
- PHP, Ruby, Swift, Kotlin
- And many more...

## Keyboard Shortcuts

| Shortcut | Command |
|----------|---------|
| `Ctrl+Shift+A` | Trigger inline code completion |
| `Ctrl+Shift+C` | Open AIVA chat panel |

## Troubleshooting

### Connection Issues
**Problem**: "Cannot connect to AIVA server"
**Solution**:
1. Ensure your AIVA local node is running: `./start.sh`
2. Check the server URL in settings
3. Verify the server is accessible: `curl http://localhost:8000/health`

### No Completions
**Problem**: Inline suggestions not appearing
**Solution**:
1. Check that inline suggestions are enabled in settings
2. Ensure you have sufficient context (at least 10 characters)
3. Try the manual completion shortcut

### Slow Responses
**Problem**: AI responses are slow
**Solutions**:
- Reduce `maxTokens` settings
- Use a smaller/faster model
- Check GPU utilization on your AIVA server
- Consider using a GPU with more VRAM

### Extension Not Loading
**Problem**: Extension fails to activate
**Solution**:
1. Check the Cursor output panel for errors
2. Ensure all dependencies are installed
3. Try reloading the window (Ctrl+Shift+P → "Developer: Reload Window")

## Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   Cursor IDE    │◄──────────────────► │  AIVA Local     │
│                 │                     │  Node Server    │
│ ┌─────────────┐ │                     │                 │
│ │AIVA Extension│ │                     │ ┌─────────────┐ │
│ │             │ │                     │ │ vLLM Engine  │ │
│ │ • Chat UI   │ │                     │ │             │ │
│ │ • Completion│ │                     │ │ • Mixtral    │ │
│ │ • Commands  │ │                     │ │ • CodeLlama  │ │
│ └─────────────┘ │                     │ └─────────────┘ │
└─────────────────┘                     └─────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────┐
                                   │   Vector DB     │
                                   │ (ChromaDB)      │
                                   │                 │
                                   │ • Memory       │
                                   │ • Context      │
                                   │ • Knowledge    │
                                   └─────────────────┘
```

## Privacy & Security

- **Zero Data Exfiltration**: All processing happens locally
- **No External APIs**: No calls to OpenAI or other cloud services
- **Local Models**: Your AI models stay on your hardware
- **Encrypted Storage**: Conversation history is stored locally
- **Configurable Permissions**: Control what data the AI can access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with your local AIVA node
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/aiva-local-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/aiva-local-assistant/discussions)
- **Documentation**: [AIVA Local Node Docs](../aiva_local_node/README.md)

---

**Built for privacy-conscious developers who want AI assistance without compromising their data sovereignty.**
