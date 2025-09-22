import { createAction, props } from '@ngrx/store';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isLoading?: boolean;
}

export interface ProcessingRequest {
  input_type: string;
  algorithm: string;
  parameters: Record<string, any>;
  data: any;
}

// Chat Actions
export const sendMessage = createAction(
  '[Chat] Send Message',
  props<{ content: string }>()
);

export const sendMessageSuccess = createAction(
  '[Chat] Send Message Success',
  props<{ message: ChatMessage }>()
);

export const receiveMessage = createAction(
  '[Chat] Receive Message',
  props<{ message: ChatMessage }>()
);

export const processConsciousness = createAction(
  '[Chat] Process Consciousness',
  props<{ request: ProcessingRequest }>()
);

export const processConsciousnessSuccess = createAction(
  '[Chat] Process Consciousness Success',
  props<{ result: any; processingTime: number }>()
);

export const processConsciousnessFailure = createAction(
  '[Chat] Process Consciousness Failure',
  props<{ error: string }>()
);

export const loadMessages = createAction('[Chat] Load Messages');

export const loadMessagesSuccess = createAction(
  '[Chat] Load Messages Success',
  props<{ messages: ChatMessage[] }>()
);

export const clearMessages = createAction('[Chat] Clear Messages');

export const setLoading = createAction(
  '[Chat] Set Loading',
  props<{ loading: boolean }>()
);
