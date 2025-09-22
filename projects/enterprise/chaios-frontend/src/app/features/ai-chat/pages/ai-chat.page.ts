import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { 
  IonContent, 
  IonHeader, 
  IonTitle, 
  IonToolbar, 
  IonFooter,
  IonItem,
  IonTextarea,
  IonButton,
  IonIcon,
  IonSelect,
  IonSelectOption,
  IonCard,
  IonCardContent,
  IonSpinner,
  IonBadge,
  IonMenuButton,
  IonButtons,
  IonGrid,
  IonRow,
  IonCol,
  IonChip,
  IonLabel
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  sendOutline, 
  micOutline, 
  attachOutline,
  settingsOutline,
  refreshOutline,
  copyOutline,
  shareOutline,
  thumbsUpOutline,
  thumbsDownOutline
} from 'ionicons/icons';
import { Subscription } from 'rxjs';

import { ApiService } from '../../../core/api.service';
import { AuthService, User } from '../../../core/services/auth.service';
import { WebSocketService } from '../../../core/websocket.service';
import { NotificationService } from '../../../core/services/notification.service';
import { ChatMessageComponent } from '../components/chat-message.component';

export interface ChatConversation {
  id: string;
  messages: ChatMessageItem[];
  provider: 'openai' | 'anthropic' | 'google';
  timestamp: Date;
  title?: string;
}

export interface ChatMessageItem {
  id: string;
  content: string;
  isBot: boolean;
  timestamp: Date;
  provider?: string;
  metadata?: any;
  status?: 'sending' | 'sent' | 'error';
}

@Component({
  selector: 'app-ai-chat',
  templateUrl: './ai-chat.page.html',
  styleUrls: ['./ai-chat.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    IonContent,
    IonHeader,
    IonTitle,
    IonToolbar,
    IonFooter,
    IonItem,
    IonTextarea,
    IonButton,
    IonIcon,
    IonSelect,
    IonSelectOption,
    IonCard,
    IonCardContent,
    IonSpinner,
    IonBadge,
    IonMenuButton,
    IonButtons,
    IonGrid,
    IonRow,
    IonCol,
    IonChip,
    IonLabel,
    ChatMessageComponent
  ]
})
export class AiChatPage implements OnInit, OnDestroy {
  @ViewChild(IonContent, { static: true }) content!: IonContent;
  @ViewChild('messageInput', { static: true }) messageInput!: ElementRef;

  currentMessage = '';
  selectedProvider: 'openai' | 'anthropic' | 'google' = 'openai';
  isLoading = false;
  isTyping = false;
  
  conversation: ChatConversation = {
    id: this.generateId(),
    messages: [],
    provider: this.selectedProvider,
    timestamp: new Date()
  };

  currentUser: User | null = null;
  connectionStatus = 'disconnected';
  
  // Predefined prompts for consciousness and mathematics
  quickPrompts = [
    'Explain consciousness mathematics',
    'What is the golden ratio in nature?',
    'How does quantum computing work?',
    'Calculate Riemann zeta zeros',
    'Optimize AI performance',
    'Analyze complex systems'
  ];

  providers = [
    { value: 'openai', label: 'ChatGPT', icon: '🤖', color: 'primary' },
    { value: 'anthropic', label: 'Claude', icon: '🧠', color: 'secondary' },
    { value: 'google', label: 'Gemini', icon: '💎', color: 'tertiary' }
  ];

  private subscriptions: Subscription[] = [];

  constructor(
    private apiService: ApiService,
    private authService: AuthService,
    private webSocketService: WebSocketService,
    private notificationService: NotificationService
  ) {
    this.initializeIcons();
  }

  ngOnInit() {
    this.setupSubscriptions();
    this.initializeChat();
    this.loadChatHistory();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  private initializeIcons() {
    addIcons({
      'send-outline': sendOutline,
      'mic-outline': micOutline,
      'attach-outline': attachOutline,
      'settings-outline': settingsOutline,
      'refresh-outline': refreshOutline,
      'copy-outline': copyOutline,
      'share-outline': shareOutline,
      'thumbs-up-outline': thumbsUpOutline,
      'thumbs-down-outline': thumbsDownOutline
    });
  }

  private setupSubscriptions() {
    // Auth state
    this.subscriptions.push(
      this.authService.currentUser$.subscribe(user => {
        this.currentUser = user;
      })
    );

    // WebSocket connection status
    this.subscriptions.push(
      this.webSocketService.connectionStatus$.subscribe(status => {
        this.connectionStatus = status;
      })
    );

    // WebSocket chat messages
    this.subscriptions.push(
      this.webSocketService.chatMessages$.subscribe(message => {
        this.handleWebSocketMessage(message);
      })
    );
  }

  private initializeChat() {
    // Add welcome message
    const welcomeMessage: ChatMessageItem = {
      id: this.generateId(),
      content: `Welcome to chAIos AI Chat! I'm your consciousness-enhanced AI assistant. I can help you with:

🧠 **Consciousness Mathematics** - Explore the Wallace Transform and consciousness optimization
📐 **Mathematical Calculations** - Golden ratio, Riemann zeta function, complex analysis  
⚛️ **Quantum Computing** - Quantum algorithms and simulation
🔬 **Scientific Research** - Advanced computational analysis
🎯 **AI Optimization** - Performance enhancement and system analysis

What would you like to explore today?`,
      isBot: true,
      timestamp: new Date(),
      provider: this.selectedProvider
    };

    this.conversation.messages = [welcomeMessage];
    this.scrollToBottom();
  }

  async sendMessage() {
    if (!this.currentMessage.trim() || this.isLoading || !this.currentUser) {
      return;
    }

    const userMessage: ChatMessageItem = {
      id: this.generateId(),
      content: this.currentMessage,
      isBot: false,
      timestamp: new Date(),
      status: 'sending'
    };

    this.conversation.messages.push(userMessage);
    this.scrollToBottom();

    const messageContent = this.currentMessage;
    this.currentMessage = '';
    this.isLoading = true;
    this.isTyping = true;

    try {
      // Send via WebSocket for real-time updates
      this.webSocketService.sendChatMessage(messageContent, this.selectedProvider);

      // Also send via HTTP API
      const chatMessage: ChatMessage = {
        content: messageContent,
        provider: this.selectedProvider,
        timestamp: new Date(),
        userId: this.currentUser.id
      };

      const response = await this.apiService.sendChatMessage(chatMessage).toPromise();
      
      userMessage.status = 'sent';
      
      if (response) {
        const botMessage: ChatMessageItem = {
          id: this.generateId(),
          content: response.conversational_response || response.content,
          isBot: true,
          timestamp: new Date(),
          provider: response.provider,
          metadata: response
        };

        this.conversation.messages.push(botMessage);
        this.scrollToBottom();
      }

    } catch (error: any) {
      console.error('Chat error:', error);
      userMessage.status = 'error';
      
      const errorMessage: ChatMessageItem = {
        id: this.generateId(),
        content: `I apologize, but I encountered an error: ${error.message || 'Unknown error'}. Please try again.`,
        isBot: true,
        timestamp: new Date(),
        provider: this.selectedProvider
      };

      this.conversation.messages.push(errorMessage);
      this.notificationService.showNotification('Failed to send message', 'error');
      
    } finally {
      this.isLoading = false;
      this.isTyping = false;
      this.scrollToBottom();
    }
  }

  selectQuickPrompt(prompt: string) {
    this.currentMessage = prompt;
    this.sendMessage();
  }

  onProviderChange() {
    console.log('Provider changed to:', this.selectedProvider);
    this.conversation.provider = this.selectedProvider;
    
    // Add provider change notification
    const changeMessage: ChatMessageItem = {
      id: this.generateId(),
      content: `Switched to ${this.getProviderLabel(this.selectedProvider)}. How can I assist you?`,
      isBot: true,
      timestamp: new Date(),
      provider: this.selectedProvider
    };

    this.conversation.messages.push(changeMessage);
    this.scrollToBottom();
  }

  clearConversation() {
    this.conversation.messages = [];
    this.initializeChat();
    this.notificationService.showNotification('Conversation cleared', 'info');
  }

  async copyMessage(message: ChatMessageItem) {
    try {
      await navigator.clipboard.writeText(message.content);
      this.notificationService.showNotification('Message copied to clipboard', 'success');
    } catch (error) {
      console.error('Failed to copy message:', error);
      this.notificationService.showNotification('Failed to copy message', 'error');
    }
  }

  shareMessage(message: ChatMessageItem) {
    if (navigator.share) {
      navigator.share({
        title: 'chAIos AI Chat',
        text: message.content,
        url: window.location.href
      }).catch(console.error);
    } else {
      this.copyMessage(message);
    }
  }

  rateMessage(message: ChatMessageItem, rating: 'up' | 'down') {
    console.log('Rating message:', message.id, rating);
    this.notificationService.showNotification(
      `Thank you for your ${rating === 'up' ? 'positive' : 'negative'} feedback!`, 
      'info'
    );
    
    // TODO: Send rating to backend
  }

  // Keyboard event handling
  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  // Voice input (placeholder)
  startVoiceInput() {
    this.notificationService.showNotification('Voice input coming soon!', 'info');
  }

  // File attachment (placeholder)
  attachFile() {
    this.notificationService.showNotification('File attachment coming soon!', 'info');
  }

  private handleWebSocketMessage(message: any) {
    if (message.type === 'chat_message' && message.data) {
      const botMessage: ChatMessageItem = {
        id: this.generateId(),
        content: message.data.content,
        isBot: true,
        timestamp: new Date(message.data.timestamp),
        provider: message.data.provider
      };

      this.conversation.messages.push(botMessage);
      this.scrollToBottom();
      this.isLoading = false;
      this.isTyping = false;
    }
  }

  private async loadChatHistory() {
    if (!this.currentUser) return;

    try {
      const history = await this.apiService.getChatHistory(this.currentUser.id, 20).toPromise();
      
      if (history && history.length > 0) {
        console.log('Loaded chat history:', history.length, 'messages');
        // TODO: Process and display chat history
      }
    } catch (error) {
      console.warn('Failed to load chat history:', error);
    }
  }

  private scrollToBottom() {
    setTimeout(() => {
      this.content.scrollToBottom(300);
    }, 100);
  }

  private generateId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getProviderLabel(provider: string): string {
    const providerData = this.providers.find(p => p.value === provider);
    return providerData ? providerData.label : provider;
  }

  // Utility methods
  getProviderIcon(provider: string): string {
    const providerData = this.providers.find(p => p.value === provider);
    return providerData ? providerData.icon : '🤖';
  }

  getProviderColor(provider: string): string {
    const providerData = this.providers.find(p => p.value === provider);
    return providerData ? providerData.color : 'primary';
  }

  isMessageFromCurrentProvider(message: ChatMessageItem): boolean {
    return !message.provider || message.provider === this.selectedProvider;
  }

  getConnectionStatusColor(): string {
    switch (this.connectionStatus) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'error': return 'danger';
      default: return 'medium';
    }
  }

  trackByMessageId(index: number, message: ChatMessageItem): string {
    return message.id;
  }
}

