import { Injectable } from '@angular/core';
import { Observable, Subject, BehaviorSubject, timer, EMPTY } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { retryWhen, delay, tap, catchError, takeUntil, filter } from 'rxjs/operators';

import { environment } from '../../../environments/environment';
import { AuthService } from './auth.service';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
  id?: string;
}

export interface ConsciousnessUpdate {
  type: 'consciousness_update';
  data: {
    performance_gain: number;
    correlation: number;
    processing_time: number;
    status: string;
  };
}

export interface QuantumUpdate {
  type: 'quantum_simulation_update';
  data: {
    qubits: number;
    fidelity: number;
    progress: number;
    status: string;
  };
}

export interface SystemAlert {
  type: 'system_alert';
  data: {
    level: 'info' | 'warning' | 'error';
    message: string;
    component?: string;
  };
}

export interface ChatMessage {
  type: 'chat_message';
  data: {
    content: string;
    provider: string;
    userId: string;
    timestamp: string;
  };
}

export type WebSocketMessageTypes = 
  | ConsciousnessUpdate 
  | QuantumUpdate 
  | SystemAlert 
  | ChatMessage 
  | WebSocketMessage;

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket$: WebSocketSubject<any> | null = null;
  private messagesSubject$ = new Subject<WebSocketMessageTypes>();
  private connectionStatusSubject$ = new BehaviorSubject<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 5000; // 5 seconds
  private destroy$ = new Subject<void>();

  // Public observables
  public messages$ = this.messagesSubject$.asObservable();
  public connectionStatus$ = this.connectionStatusSubject$.asObservable();

  // Message type specific observables
  public consciousnessUpdates$ = this.messages$.pipe(
    filter((msg): msg is ConsciousnessUpdate => msg.type === 'consciousness_update'),
    tap(msg => console.log('Received consciousness message:', msg)),
    catchError(err => {
      console.error('Error in consciousness updates stream:', err);
      return EMPTY;
    })
  );

  public quantumUpdates$ = this.messages$.pipe(
    filter((msg): msg is QuantumUpdate => msg.type === 'quantum_simulation_update'),
    tap(msg => console.log('Received quantum message:', msg)),
    catchError(err => {
      console.error('Error in quantum updates stream:', err);
      return EMPTY;
    })
  );

  public systemAlerts$ = this.messages$.pipe(
    filter((msg): msg is SystemAlert => msg.type === 'system_alert'),
    tap(msg => console.log('Received system alert:', msg)),
    catchError(err => {
      console.error('Error in system alerts stream:', err);
      return EMPTY;
    })
  );

  public chatMessages$ = this.messages$.pipe(
    filter((msg): msg is ChatMessage => msg.type === 'chat_message'),
    tap(msg => console.log('Received chat message:', msg)),
    catchError(err => {
      console.error('Error in chat messages stream:', err);
      return EMPTY;
    })
  );

  constructor(private authService: AuthService) {
    console.log('🔌 WebSocketService initialized');
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Observable<WebSocketMessageTypes> {
    if (this.socket$) {
      return this.messages$;
    }

    console.log('🔌 Connecting to WebSocket:', environment.wsUrl);
    this.connectionStatusSubject$.next('connecting');

    // Get auth token for WebSocket authentication
    const token = localStorage.getItem('access_token');
    const wsUrl = token 
      ? `${environment.wsUrl}/ws?token=${token}`
      : `${environment.wsUrl}/ws`;

    this.socket$ = webSocket({
      url: wsUrl,
      openObserver: {
        next: () => {
          console.log('✅ WebSocket connected');
          this.connectionStatusSubject$.next('connected');
          this.reconnectAttempts = 0;
        }
      },
      closeObserver: {
        next: () => {
          console.log('🔌 WebSocket disconnected');
          this.connectionStatusSubject$.next('disconnected');
          this.socket$ = null;
          this.scheduleReconnect();
        }
      }
    });

    // Subscribe to socket messages
    this.socket$.pipe(
      takeUntil(this.destroy$),
      retryWhen(errors => 
        errors.pipe(
          tap(error => {
            console.error('❌ WebSocket error:', error);
            this.connectionStatusSubject$.next('error');
          }),
          delay(this.reconnectInterval)
        )
      ),
      catchError(error => {
        console.error('❌ WebSocket fatal error:', error);
        this.connectionStatusSubject$.next('error');
        return EMPTY;
      })
    ).subscribe({
      next: (message) => {
        this.handleMessage(message);
      },
      error: (error) => {
        console.error('❌ WebSocket subscription error:', error);
        this.connectionStatusSubject$.next('error');
      }
    });

    return this.messages$;
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    console.log('🔌 Disconnecting WebSocket');
    
    this.destroy$.next();
    
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
    }
    
    this.connectionStatusSubject$.next('disconnected');
    this.reconnectAttempts = 0;
  }

  /**
   * Send message to WebSocket server
   */
  sendMessage(message: WebSocketMessage): void {
    if (this.socket$ && this.connectionStatusSubject$.value === 'connected') {
      const messageWithTimestamp = {
        ...message,
        timestamp: new Date().toISOString(),
        id: this.generateMessageId()
      };
      
      console.log('📤 Sending WebSocket message:', messageWithTimestamp);
      this.socket$.next(messageWithTimestamp);
    } else {
      console.warn('⚠️ Cannot send message: WebSocket not connected');
    }
  }

  /**
   * Subscribe to consciousness processing updates
   */
  subscribeToConsciousness(): void {
    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'consciousness_updates' }
    });
  }

  /**
   * Subscribe to quantum simulation updates
   */
  subscribeToQuantum(): void {
    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'quantum_updates' }
    });
  }

  /**
   * Subscribe to system alerts
   */
  subscribeToSystemAlerts(): void {
    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'system_alerts' }
    });
  }

  /**
   * Subscribe to chat messages
   */
  subscribeToChat(userId: string): void {
    this.sendMessage({
      type: 'subscribe',
      data: { 
        channel: 'chat_messages',
        userId: userId
      }
    });
  }

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(channel: string): void {
    this.sendMessage({
      type: 'unsubscribe',
      data: { channel: channel }
    });
  }

  /**
   * Send heartbeat/ping message
   */
  sendHeartbeat(): void {
    this.sendMessage({
      type: 'ping',
      data: { timestamp: Date.now() }
    });
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): string {
    return this.connectionStatusSubject$.value;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connectionStatusSubject$.value === 'connected';
  }

  // Private methods

  private handleMessage(message: any): void {
    try {
      console.log('📥 Received WebSocket message:', message);
      
      // Add timestamp if not present
      if (!message.timestamp) {
        message.timestamp = new Date().toISOString();
      }

      // Emit to appropriate subject
      this.messagesSubject$.next(message);

      // Handle specific message types
      switch (message.type) {
        case 'pong':
          // Handle heartbeat response
          break;
        case 'error':
          console.error('🔥 WebSocket server error:', message.data);
          break;
        case 'connection_established':
          console.log('🤝 WebSocket connection established:', message.data);
          break;
        default:
          // Generic message handling
          break;
      }
    } catch (error) {
      console.error('❌ Error handling WebSocket message:', error);
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
      
      console.log(`🔄 Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
      
      timer(delay).subscribe(() => {
        if (this.connectionStatusSubject$.value !== 'connected') {
          console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          this.connect().subscribe();
        }
      });
    } else {
      console.error('❌ Max reconnect attempts reached. WebSocket connection failed.');
      this.connectionStatusSubject$.next('error');
    }
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Utility methods for specific chAIos features

  /**
   * Request consciousness processing status
   */
  requestConsciousnessStatus(): void {
    this.sendMessage({
      type: 'request_status',
      data: { component: 'consciousness' }
    });
  }

  /**
   * Request quantum simulation status
   */
  requestQuantumStatus(): void {
    this.sendMessage({
      type: 'request_status',
      data: { component: 'quantum' }
    });
  }

  /**
   * Send chat message via WebSocket
   */
  sendChatMessage(content: string, provider: string = 'openai'): void {
    const user = this.authService.getCurrentUser();
    if (!user) {
      console.warn('⚠️ Cannot send chat message: User not authenticated');
      return;
    }

    this.sendMessage({
      type: 'chat_message',
      data: {
        content: content,
        provider: provider,
        userId: user.id,
        timestamp: new Date().toISOString()
      }
    });
  }

  /**
   * Start consciousness processing session
   */
  startConsciousnessSession(parameters: any): void {
    this.sendMessage({
      type: 'start_consciousness_session',
      data: parameters
    });
  }

  /**
   * Start quantum simulation session
   */
  startQuantumSession(parameters: any): void {
    this.sendMessage({
      type: 'start_quantum_session',
      data: parameters
    });
  }

  /**
   * Stop active session
   */
  stopSession(sessionType: 'consciousness' | 'quantum'): void {
    this.sendMessage({
      type: 'stop_session',
      data: { sessionType: sessionType }
    });
  }
}
