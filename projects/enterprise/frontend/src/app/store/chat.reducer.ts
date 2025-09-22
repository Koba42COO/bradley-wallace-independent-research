import { createReducer, on } from '@ngrx/store';
import * as ChatActions from './chat.actions';

export interface ChatState {
  messages: ChatActions.ChatMessage[];
  loading: boolean;
  error: string | null;
  processingTime: number | null;
}

export const initialState: ChatState = {
  messages: [],
  loading: false,
  error: null,
  processingTime: null,
};

export const chatReducer = createReducer(
  initialState,
  on(ChatActions.sendMessage, (state) => ({
    ...state,
    loading: true,
    error: null,
  })),
  on(ChatActions.sendMessageSuccess, (state, { message }) => ({
    ...state,
    messages: [...state.messages, message],
    loading: false,
  })),
  on(ChatActions.receiveMessage, (state, { message }) => ({
    ...state,
    messages: [...state.messages, message],
  })),
  on(ChatActions.processConsciousness, (state) => ({
    ...state,
    loading: true,
    error: null,
  })),
  on(ChatActions.processConsciousnessSuccess, (state, { result, processingTime }) => ({
    ...state,
    loading: false,
    processingTime,
    messages: [...state.messages, {
      id: Date.now().toString(),
      sender: 'bot' as const,
      text: JSON.stringify(result, null, 2),
      timestamp: new Date(),
    }],
  })),
  on(ChatActions.processConsciousnessFailure, (state, { error }) => ({
    ...state,
    loading: false,
    error,
  })),
  on(ChatActions.loadMessagesSuccess, (state, { messages }) => ({
    ...state,
    messages,
    loading: false,
  })),
  on(ChatActions.clearMessages, (state) => ({
    ...state,
    messages: [],
  })),
  on(ChatActions.setLoading, (state, { loading }) => ({
    ...state,
    loading,
  }))
);
