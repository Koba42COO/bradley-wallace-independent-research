import { Component, OnInit, ViewChild } from '@angular/core';
import { IonContent, IonicModule } from '@ionic/angular';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatMessageComponent } from './components/chat-message/chat-message.component';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { SharedModule } from '../../shared/shared.module';
import { ApiService } from '../../core/services/api';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isLoading?: boolean;
}

@Component({
  selector: 'app-llm-convo',
  templateUrl: './llm-convo.page.html',
  styleUrls: ['./llm-convo.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    SharedModule,
    ChatMessageComponent,
    ChatInputComponent,
  ],
})
export class LlmConvoPage implements OnInit {
  @ViewChild(IonContent) content!: IonContent;
  messages: ChatMessage[] = [];
  isLoading = false;
  systemTools: any = null;
  toolsLoaded = false;

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadSystemTools();
    
    // Add enhanced welcome message
    this.messages.push({
      id: '1',
      sender: 'bot',
      text: '🚀 Welcome to the Enterprise Consciousness Platform with CURATED BEST-OF-BREED Tools!\n\nI now have access to 25 revolutionary curated tools including:\n• 🧠 Advanced Consciousness Mathematics (Wallace Transform V3.0)\n• 🤖 Grok Jr Fast Coding Agent (Revolutionary AI)\n• ⚡ Transcendent LLM Builder\n• 🔐 Enterprise Security & Penetration Testing\n• 🌐 Unified Ecosystem Integration\n• 🔬 Advanced Data Processing & Scientific Scraping\n• ⚛️ Quantum Computing & Consciousness Processing\n• 🔗 Blockchain & Quantum Email Systems\n• 🎯 Industrial-Grade Development & Testing Tools\n\n**NO REDUNDANCY - Only the BEST tools from 386+ options!**\n\nHow can I help you today?',
      timestamp: new Date()
    });
  }

  loadSystemTools() {
    this.apiService.getSystemTools().subscribe({
      next: (response) => {
        if (response.success) {
          this.systemTools = response.data;
          this.toolsLoaded = true;
          
          // Add system tools info message
          const toolsCount = this.systemTools?.total_tools || 0;
          const categoriesCount = Object.keys(this.systemTools?.categories || {}).length;
          
          this.messages.push({
            id: 'tools-info',
            sender: 'bot',
            text: `🎯 **CURATED TOOLS SYSTEM ACTIVE**: ${toolsCount} best-of-breed tools loaded across ${categoriesCount} categories.\n\n🚀 **Revolutionary Tool Categories:**\n• 🧠 **Consciousness**: Wallace Transform V3.0, Möbius Optimization V2.5\n• 🤖 **AI/ML**: Transcendent LLM V3.0, Revolutionary Learning V2.0\n• 💻 **Development**: Unified Testing V3.0, Industrial Stress Testing\n• 🔐 **Security**: AIVA Scanner V3.0, Enterprise Pen Testing V2.5\n• 🌐 **Integration**: Unified Ecosystem V3.0, Master Codebase V3.0\n• 🔬 **Data**: Comprehensive Harvesting, Scientific Scraping\n• ⚛️ **Quantum**: Consciousness Processing, Annealing Optimization\n• 🔗 **Blockchain**: Quantum Email, Knowledge Marketplace\n• 🎯 **Grok Jr**: Code Generation, Optimization, Consciousness Coding\n\n**93% REDUNDANCY ELIMINATED** - Only the most advanced tools!\n\n🎯 Try asking me to:\n• "Generate revolutionary code with Grok Jr"\n• "Apply consciousness mathematics to optimize this"\n• "Run enterprise security analysis"\n• "Create transcendent AI solution"\n• "Integrate quantum consciousness processing"\n• "Build blockchain knowledge system"`,
            timestamp: new Date()
          });
        }
      },
      error: (error) => {
        console.error('Failed to load system tools:', error);
        this.messages.push({
          id: 'tools-error',
          sender: 'bot',
          text: '⚠️ Note: Advanced system tools are currently unavailable, but basic consciousness processing is still available.',
          timestamp: new Date()
        });
      }
    });
  }

  onSendMessage(message: string) {
    if (!message.trim() || this.isLoading) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: message,
      timestamp: new Date()
    };
    this.messages.push(userMessage);

    // Add loading message
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      sender: 'bot',
      text: '🧠 Processing with consciousness mathematics and system tools...',
      timestamp: new Date(),
      isLoading: true
    };
    this.messages.push(loadingMessage);

    this.isLoading = true;
    this.scrollToBottom();

    // Analyze message and determine processing approach
    this.processEnhancedMessage(message);
  }

  processEnhancedMessage(message: string) {
    const lowerMessage = message.toLowerCase();
    
    // Determine processing strategy based on message content
    if (lowerMessage.includes('system') && (lowerMessage.includes('metrics') || lowerMessage.includes('status'))) {
      // System analysis request
      this.apiService.processConsciousnessWithTools('system_analysis', {}).subscribe({
        next: (response) => this.handleEnhancedResponse(response, 'System Analysis'),
        error: (error) => this.handleError(error)
      });
    } else if (lowerMessage.includes('wallace') || lowerMessage.includes('transform')) {
      // Wallace transform request
      this.apiService.processConsciousnessWithTools('wallace_transform', {
        data: 2.5,
        iterations: 100,
        dimensionalEnhancement: true
      }).subscribe({
        next: (response) => this.handleEnhancedResponse(response, 'Wallace Transform'),
        error: (error) => this.handleError(error)
      });
    } else if (lowerMessage.includes('consciousness') && lowerMessage.includes('bridge')) {
      // Consciousness bridge request
      this.apiService.processConsciousnessWithTools('consciousness_bridge', {
        baseValue: 0.5,
        iterations: 100
      }).subscribe({
        next: (response) => this.handleEnhancedResponse(response, 'Consciousness Bridge'),
        error: (error) => this.handleError(error)
      });
    } else if (lowerMessage.includes('file') || lowerMessage.includes('directory')) {
      // File system operation
      this.apiService.executeSystemTool('sys_get_metrics', {}).subscribe({
        next: (response) => this.handleToolResponse(response, 'File System Analysis'),
        error: (error) => this.handleError(error)
      });
    } else if (lowerMessage.includes('network') || lowerMessage.includes('ping') || lowerMessage.includes('dns')) {
      // Network operation
      this.apiService.executeSystemTool('net_ping_host', { host: 'google.com', count: 4 }).subscribe({
        next: (response) => this.handleToolResponse(response, 'Network Analysis'),
        error: (error) => this.handleError(error)
      });
    } else if (lowerMessage.includes('encrypt') || lowerMessage.includes('hash') || lowerMessage.includes('crypto')) {
      // Cryptography operation
      this.apiService.executeSystemTool('crypto_hash_data', { 
        data: message, 
        algorithm: 'SHA256' 
      }).subscribe({
        next: (response) => this.handleToolResponse(response, 'Cryptographic Processing'),
        error: (error) => this.handleError(error)
      });
    } else {
      // Default: Natural conversation with consciousness processing
      this.apiService.sendChatMessage(message).subscribe({
        next: (response) => this.handleEnhancedResponse(response, 'Consciousness Processing'),
        error: (error) => this.handleError(error)
      });
    }
  }

  handleEnhancedResponse(response: any, processingType: string) {
    // Remove loading message
    this.messages = this.messages.filter(msg => !msg.isLoading);
    
    let responseText = '';
    
    if (response.success && response.result) {
      const result = response.result;
      
      // Use conversational response if available
      if (result.conversational_response) {
        responseText = result.conversational_response;
      } else {
        // Fallback to technical format for non-conversational results
        responseText = `🧠 ${processingType} Complete\n\n`;
        
        if (result.system_tools_used && result.system_tools_used.length > 0) {
          responseText += `🛠️ System Tools Used: ${result.system_tools_used.join(', ')}\n\n`;
        }
        
        if (result.algorithm) {
          responseText += `📊 Algorithm: ${result.algorithm}\n`;
        }
        
        if (result.enhanced_with_system_access) {
          responseText += `✨ Enhanced with full system access\n\n`;
        }
        
        // Format specific results
        if (result.results) {
          responseText += '📈 Results:\n';
          Object.keys(result.results).forEach(key => {
            responseText += `• ${key}: ${JSON.stringify(result.results[key], null, 2)}\n`;
          });
        } else if (result.result) {
          responseText += `📊 Result: ${JSON.stringify(result.result, null, 2)}\n`;
        }
        
        if (response.processing_time) {
          responseText += `\n⏱️ Processing Time: ${response.processing_time.toFixed(3)}s`;
        }
        
        if (response.metadata) {
          responseText += `\n🔧 Available Tools: ${response.metadata.system_tools_available}`;
        }
      }
    } else {
      responseText = `❌ Processing failed: ${response.error || 'Unknown error'}`;
    }

    const botMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'bot',
      text: responseText,
      timestamp: new Date()
    };
    this.messages.push(botMessage);
    this.isLoading = false;
    this.scrollToBottom();
  }

  handleToolResponse(response: any, toolType: string) {
    // Remove loading message
    this.messages = this.messages.filter(msg => !msg.isLoading);
    
    let responseText = `🔧 ${toolType} Complete\n\n`;
    
    if (response.success && response.data) {
      responseText += `✅ Tool: ${response.tool_name}\n`;
      responseText += `📊 Result: ${JSON.stringify(response.data, null, 2)}\n`;
      responseText += `⏱️ Execution Time: ${response.execution_time?.toFixed(3)}s`;
    } else {
      responseText += `❌ Tool execution failed: ${response.error || 'Unknown error'}`;
    }

    const botMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'bot',
      text: responseText,
      timestamp: new Date()
    };
    this.messages.push(botMessage);
    this.isLoading = false;
    this.scrollToBottom();
  }

  handleError(error: any) {
    // Remove loading message
    this.messages = this.messages.filter(msg => !msg.isLoading);
    
    // Add error message
    const errorMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'bot',
      text: `❌ Sorry, I encountered an error: ${error}\n\n🔄 Please try rephrasing your request or ask me about:\n• System metrics analysis\n• Wallace transform processing\n• File operations\n• Network analysis\n• Cryptographic operations\n• Consciousness mathematics`,
      timestamp: new Date()
    };
    this.messages.push(errorMessage);
    this.isLoading = false;
    this.scrollToBottom();
  }

  scrollToBottom() {
    setTimeout(() => {
      if (this.content) {
        this.content.scrollToBottom(300);
      }
    }, 100);
  }

  trackByMessageId(index: number, message: ChatMessage): string {
    return message.id;
  }
}
