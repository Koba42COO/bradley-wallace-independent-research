import { Component, OnInit, ViewChild } from '@angular/core';
import { IonContent, IonicModule } from '@ionic/angular';
import { Store } from '@ngrx/store';
import { Observable } from 'rxjs';
import { AppState } from '../../store';
import { sendMessage } from '../../store/chat.actions';
import { selectAllMessages, selectChatLoading } from '../../store/chat.selectors';
import { tap } from 'rxjs/operators';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatMessageComponent } from './components/chat-message/chat-message.component';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { SharedModule } from 'src/app/shared/shared.module';

interface ChatMessage {
  sender: 'user' | 'bot';
  text: string;
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
    ChatInputComponent
  ],
})
export class LlmConvoPage implements OnInit {
  @ViewChild(IonContent, { static: false }) content!: IonContent;
  messages$: Observable<ChatMessage[]>;
  isLoading$: Observable<boolean>;

  constructor(private store: Store<AppState>) {
    this.messages$ = this.store.select(selectAllMessages).pipe(
      tap(() => this.scrollToBottom())
    );
    this.isLoading$ = this.store.select(selectChatLoading);
  }

  ngOnInit() {
    // A welcome message could be dispatched from an effect upon connection
  }

  onSendMessage(messageText: string) {
    if (messageText.trim().length > 0) {
      this.store.dispatch(sendMessage({ message: messageText }));
    }
  }

  private scrollToBottom() {
    setTimeout(() => {
      this.content?.scrollToBottom(300);
    }, 100);
  }
}
