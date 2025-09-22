import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isLoading?: boolean;
}

@Component({
  selector: 'app-chat-message',
  templateUrl: './chat-message.component.html',
  styleUrls: ['./chat-message.component.scss'],
  standalone: true,
  imports: [CommonModule, IonicModule],
})
export class ChatMessageComponent {
  @Input() message!: ChatMessage;

  formatMessage(text: string): string {
    // Basic formatting for JSON responses
    try {
      const parsed = JSON.parse(text);
      return `<pre><code>${JSON.stringify(parsed, null, 2)}</code></pre>`;
    } catch {
      // Not JSON, return as is with basic HTML formatting
      return text
        .replace(/\n/g, '<br>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
    }
  }

  formatTimestamp(timestamp: Date): string {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }
}
