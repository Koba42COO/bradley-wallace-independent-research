import { createSelector, createFeatureSelector } from '@ngrx/store';
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

export const selectChatError = createSelector(
  selectChatState,
  (state: ChatState) => state.error
);

export const selectProcessingTime = createSelector(
  selectChatState,
  (state: ChatState) => state.processingTime
);

export const selectLatestMessage = createSelector(
  selectAllMessages,
  (messages) => messages.length > 0 ? messages[messages.length - 1] : null
);
