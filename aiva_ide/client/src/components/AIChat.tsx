import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { gptApi } from '@/lib/api';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

const AIChat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI coding assistant powered by GPT. How can I help you with your code today?',
      role: 'assistant',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Convert messages to OpenAI format
      const conversationHistory = messages.concat(userMessage).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await gptApi.chat(conversationHistory);

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        role: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Failed to get AI response:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error while processing your request. Please check your OpenAI API key configuration.',
        role: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b">
        <h3 className="font-semibold flex items-center gap-2">
          <Bot className="h-4 w-4" />
          AI Assistant
        </h3>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
              )}
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground ml-auto'
                    : 'bg-muted'
                }`}
              >
                <p className="text-sm">{message.content}</p>
              </div>
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about coding..."
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            disabled={isLoading}
          />
          <Button onClick={handleSend} size="icon" disabled={isLoading || !input.trim()}>
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AIChat;
