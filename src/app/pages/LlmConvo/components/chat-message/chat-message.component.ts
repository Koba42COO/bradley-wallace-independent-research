import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';

interface ChatMessage {
  sender: 'user' | 'bot';
  text: string;
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
}
