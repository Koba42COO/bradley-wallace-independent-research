import { createFeatureSelector, createSelector } from '@ngrx/store';
import { ChatState } from './chat.reducer';

export const selectChatState = createFeatureSelector<ChatState>('chat');

export const selectAllMessages = createSelector(
  selectChatState,
  (state: ChatState) => state.messages
);

export const selectChatLoading = createSelector(
  selectChatState,
  (state: ChatState) => state.loading
);
