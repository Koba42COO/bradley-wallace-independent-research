import { ActionReducerMap } from '@ngrx/store';
import * as fromChat from './chat.reducer';

export interface AppState {
  chat: fromChat.ChatState;
}

export const reducers: ActionReducerMap<AppState> = {
  chat: fromChat.chatReducer,
};
