import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  IonCard, 
  IonCardContent, 
  IonButton, 
  IonIcon, 
  IonText,
  IonBadge,
  IonSpinner
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  copyOutline, 
  shareOutline, 
  thumbsUpOutline, 
  thumbsDownOutline,
  checkmarkOutline,
  closeOutline,
  timeOutline
} from 'ionicons/icons';

import { ChatMessageItem } from '../pages/ai-chat.page';

@Component({
  selector: 'app-chat-message',
  template: `
    <div class="message-container" [class.user-message]="!message.isBot" [class.bot-message]="message.isBot">
      
      <!-- Bot Avatar -->
      <div class="message-avatar" *ngIf="message.isBot">
        <span class="avatar-icon">{{ getProviderIcon() }}</span>
      </div>

      <!-- Message Content -->
      <div class="message-content">
        
        <!-- Message Header -->
        <div class="message-header" *ngIf="message.isBot">
          <span class="provider-name">{{ getProviderName() }}</span>
          <span class="message-time">{{ formatTime(message.timestamp) }}</span>
        </div>

        <!-- Message Bubble -->
        <div class="message-bubble" [class.user-bubble]="!message.isBot" [class.bot-bubble]="message.isBot">
          
          <!-- Message Status Icon -->
          <div class="message-status" *ngIf="!message.isBot && message.status">
            <ion-spinner name="crescent" *ngIf="message.status === 'sending'"></ion-spinner>
            <ion-icon name="checkmark-outline" *ngIf="message.status === 'sent'" color="success"></ion-icon>
            <ion-icon name="close-outline" *ngIf="message.status === 'error'" color="danger"></ion-icon>
          </div>

          <!-- Message Text -->
          <div class="message-text" [innerHTML]="formatMessage(message.content)"></div>

          <!-- Metadata -->
          <div class="message-metadata" *ngIf="message.metadata">
            <ion-badge color="medium" class="metadata-badge">
              Response time: {{ message.metadata.processing_time?.toFixed(3) }}s
            </ion-badge>
          </div>

        </div>

        <!-- Message Actions -->
        <div class="message-actions" *ngIf="message.isBot">
          
          <ion-button 
            fill="clear" 
            size="small" 
            (click)="onCopy()"
            class="action-button">
            <ion-icon name="copy-outline" slot="icon-only"></ion-icon>
          </ion-button>

          <ion-button 
            fill="clear" 
            size="small" 
            (click)="onShare()"
            class="action-button">
            <ion-icon name="share-outline" slot="icon-only"></ion-icon>
          </ion-button>

          <ion-button 
            fill="clear" 
            size="small" 
            (click)="onRate('up')"
            class="action-button">
            <ion-icon name="thumbs-up-outline" slot="icon-only"></ion-icon>
          </ion-button>

          <ion-button 
            fill="clear" 
            size="small" 
            (click)="onRate('down')"
            class="action-button">
            <ion-icon name="thumbs-down-outline" slot="icon-only"></ion-icon>
          </ion-button>

        </div>

        <!-- User Message Time -->
        <div class="user-message-time" *ngIf="!message.isBot">
          <span>{{ formatTime(message.timestamp) }}</span>
        </div>

      </div>

      <!-- User Avatar -->
      <div class="message-avatar user-avatar" *ngIf="!message.isBot">
        <span class="avatar-icon">👤</span>
      </div>

    </div>
  `,
  styleUrls: ['./chat-message.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    IonCard,
    IonCardContent,
    IonButton,
    IonIcon,
    IonText,
    IonBadge,
    IonSpinner
  ]
})
export class ChatMessageComponent {
  @Input() message!: ChatMessageItem;
  @Input() currentProvider: string = 'openai';
  
  @Output() copy = new EventEmitter<ChatMessageItem>();
  @Output() share = new EventEmitter<ChatMessageItem>();
  @Output() rate = new EventEmitter<{ message: ChatMessageItem, rating: 'up' | 'down' }>();

  private providers = {
    'openai': { name: 'ChatGPT', icon: '🤖' },
    'anthropic': { name: 'Claude', icon: '🧠' },
    'google': { name: 'Gemini', icon: '💎' }
  };

  constructor() {
    this.initializeIcons();
  }

  private initializeIcons() {
    addIcons({
      'copy-outline': copyOutline,
      'share-outline': shareOutline,
      'thumbs-up-outline': thumbsUpOutline,
      'thumbs-down-outline': thumbsDownOutline,
      'checkmark-outline': checkmarkOutline,
      'close-outline': closeOutline,
      'time-outline': timeOutline
    });
  }

  getProviderIcon(): string {
    const provider = this.message.provider || this.currentProvider;
    return this.providers[provider as keyof typeof this.providers]?.icon || '🤖';
  }

  getProviderName(): string {
    const provider = this.message.provider || this.currentProvider;
    return this.providers[provider as keyof typeof this.providers]?.name || 'AI';
  }

  formatTime(timestamp: Date): string {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  formatMessage(content: string): string {
    // Basic markdown-like formatting
    let formatted = content
      // Bold text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Italic text
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Code blocks
      .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      // Inline code
      .replace(/`(.*?)`/g, '<code>$1</code>')
      // Line breaks
      .replace(/\n/g, '<br>')
      // Mathematical expressions (basic)
      .replace(/φ/g, '<span class="math-symbol">φ</span>')
      .replace(/σ/g, '<span class="math-symbol">σ</span>')
      .replace(/π/g, '<span class="math-symbol">π</span>')
      .replace(/∞/g, '<span class="math-symbol">∞</span>')
      // Highlight consciousness-related terms
      .replace(/(consciousness|quantum|golden ratio|zeta|optimization)/gi, '<span class="highlight-term">$1</span>');

    return formatted;
  }

  onCopy() {
    this.copy.emit(this.message);
  }

  onShare() {
    this.share.emit(this.message);
  }

  onRate(rating: 'up' | 'down') {
    this.rate.emit({ message: this.message, rating });
  }
}

