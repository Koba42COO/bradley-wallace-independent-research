import { createReducer, on } from '@ngrx/store';
import * as ChatActions from './chat.actions';

export interface ChatState {
  messages: any[];
  loading: boolean;
}

export const initialState: ChatState = {
  messages: [],
  loading: false,
};

export const chatReducer = createReducer(
  initialState,
  on(ChatActions.sendMessage, (state, { message }) => ({
    ...state,
    messages: [...state.messages, { sender: 'user', text: message }],
    loading: true,
  })),
  on(ChatActions.receiveMessage, (state, { message }) => ({
    ...state,
    messages: [...state.messages, { sender: 'bot', text: message.payload.reply }],
    loading: false,
  }))
);
