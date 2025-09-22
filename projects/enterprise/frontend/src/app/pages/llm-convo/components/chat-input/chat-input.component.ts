import { Component, EventEmitter, Output, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';

@Component({
  selector: 'app-chat-input',
  templateUrl: './chat-input.component.html',
  styleUrls: ['./chat-input.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule, IonicModule],
})
export class ChatInputComponent {
  @Output() sendMessage = new EventEmitter<string>();
  @Input() disabled: boolean = false;

  userInput: string = '';

  onSendMessage() {
    if (this.userInput.trim() && !this.disabled) {
      this.sendMessage.emit(this.userInput.trim());
      this.userInput = '';
    }
  }

  onInputChange() {
    // Auto-resize handled by autoGrow
    // Could add typing indicators here in the future
  }
}
