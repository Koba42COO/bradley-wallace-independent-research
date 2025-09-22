import { Component, EventEmitter, Output, Input } from '@angular/core';

@Component({
  selector: 'app-chat-input',
  templateUrl: './chat-input.component.html',
  styleUrls: ['./chat-input.component.scss'],
})
export class ChatInputComponent {
  @Output() sendMessage = new EventEmitter<string>();
  @Input() disabled: boolean = false;
  newMessage: string = '';

  constructor() { }

  sendMessage(event: Event) {
    event.preventDefault();
    if (this.newMessage.trim().length > 0) {
      this.sendMessage.emit(this.newMessage);
      this.newMessage = '';
    }
  }
}
