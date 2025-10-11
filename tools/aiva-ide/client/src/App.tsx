import React, { useState, useEffect, useRef } from 'react';
import { RealtimeClient, gptApi, fileApi } from './lib/api';
import './App.css';

interface FileItem {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
}

function App() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<Array<{role: string, content: string}>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [currentRoom, setCurrentRoom] = useState('default-room');

  const realtimeClient = useRef<RealtimeClient | null>(null);
  const editorRef = useRef<any>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    realtimeClient.current = new RealtimeClient();

    // Connect to default room
    connectToRoom(currentRoom);

    // Load initial file list
    loadFiles();

    return () => {
      if (realtimeClient.current) {
        realtimeClient.current.disconnect();
      }
    };
  }, []);

  const connectToRoom = async (roomId: string) => {
    try {
      if (realtimeClient.current) {
        await realtimeClient.current.connect(roomId);
        setIsConnected(true);
        setCurrentRoom(roomId);

        // Set up event listeners
        realtimeClient.current.onCodeChange((data) => {
          console.log('Received code change:', data);
          // Update editor content if it's from another user
          if (data.userId !== 'local' && editorRef.current) {
            setFileContent(data.content);
          }
        });

        realtimeClient.current.onUserJoined((data) => {
          console.log('User joined:', data.userId);
        });

        realtimeClient.current.onUserLeft((data) => {
          console.log('User left:', data.userId);
        });
      }
    } catch (error) {
      console.error('Failed to connect to room:', error);
      setIsConnected(false);
    }
  };

  const loadFiles = async () => {
    try {
      const response = await fileApi.getFiles();
      setFiles(response.files || []);
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  };

  const loadFileContent = async (filePath: string) => {
    try {
      const response = await fileApi.readFile(filePath);
      setFileContent(response.content);
      setSelectedFile(filePath);
    } catch (error) {
      console.error('Failed to load file:', error);
    }
  };

  const saveFileContent = async () => {
    if (!selectedFile) return;

    try {
      await fileApi.writeFile(selectedFile, fileContent);

      // Emit code change for real-time collaboration
      if (realtimeClient.current && realtimeClient.current.isConnected()) {
        realtimeClient.current.emitCodeChange({
          filePath: selectedFile,
          content: fileContent,
          userId: 'local'
        });
      }

      console.log('File saved successfully');
    } catch (error) {
      console.error('Failed to save file:', error);
    }
  };

  const handleFileContentChange = (newContent: string) => {
    setFileContent(newContent);

    // Debounced real-time sync
    if (realtimeClient.current && realtimeClient.current.isConnected()) {
      clearTimeout(window.syncTimeout);
      window.syncTimeout = setTimeout(() => {
        realtimeClient.current?.emitCodeChange({
          filePath: selectedFile,
          content: newContent,
          userId: 'local'
        });
      }, 500);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;

    const userMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    try {
      const response = await gptApi.chat([...chatMessages, userMessage]);
      const assistantMessage = { role: 'assistant', content: response.message };
      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send chat message:', error);
      const errorMessage = { role: 'assistant', content: 'Sorry, I encountered an error processing your request.' };
      setChatMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleCodeCompletion = async () => {
    if (!selectedFile || !fileContent) return;

    try {
      // Get the current line or selection for completion
      const lines = fileContent.split('\n');
      const lastLine = lines[lines.length - 1] || '';

      const response = await gptApi.completeCode(lastLine, 'javascript', fileContent);
      const newContent = fileContent + response.completion;
      setFileContent(newContent);

      // Emit the change
      if (realtimeClient.current && realtimeClient.current.isConnected()) {
        realtimeClient.current.emitCodeChange({
          filePath: selectedFile,
          content: newContent,
          userId: 'local'
        });
      }
    } catch (error) {
      console.error('Code completion failed:', error);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>AIVA IDE - AI-Powered Development Environment</h1>
        <div className="status">
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢' : '🔴'} {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="room-info">Room: {currentRoom}</span>
        </div>
      </header>

      <div className="main-content">
        <div className="sidebar">
          <div className="file-explorer">
            <h3>File Explorer</h3>
            <button onClick={loadFiles} className="refresh-btn">🔄 Refresh</button>
            <div className="file-list">
              {files.map((file) => (
                <div
                  key={file.path}
                  className={`file-item ${file.type} ${selectedFile === file.path ? 'selected' : ''}`}
                  onClick={() => file.type === 'file' && loadFileContent(file.path)}
                >
                  {file.type === 'directory' ? '📁' : '📄'} {file.name}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="editor-panel">
          <div className="editor-toolbar">
            <button onClick={saveFileContent} disabled={!selectedFile}>
              💾 Save
            </button>
            <button onClick={handleCodeCompletion} disabled={!selectedFile}>
              ✨ Complete Code
            </button>
            <span className="current-file">
              {selectedFile ? `📝 ${selectedFile}` : 'No file selected'}
            </span>
          </div>

          <textarea
            className="code-editor"
            value={fileContent}
            onChange={(e) => handleFileContentChange(e.target.value)}
            placeholder="Select a file to start editing..."
            disabled={!selectedFile}
          />
        </div>

        <div className="chat-panel">
          <div className="chat-header">
            <h3>AI Assistant</h3>
          </div>

          <div className="chat-messages">
            {chatMessages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong> {msg.content}
              </div>
            ))}
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              placeholder="Ask me anything about your code..."
              className="chat-input"
            />
            <button onClick={sendChatMessage} className="send-btn">
              📤 Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
