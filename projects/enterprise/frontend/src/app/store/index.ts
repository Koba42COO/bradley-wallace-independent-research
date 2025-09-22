import { ActionReducerMap, MetaReducer } from '@ngrx/store';
import { environment } from '../../environments/environment';
import { authReducer, AuthState } from './auth.reducer';
import { chatReducer, ChatState } from './chat.reducer';

export interface AppState {
  auth: AuthState;
  chat: ChatState;
}

export const reducers: ActionReducerMap<AppState> = {
  auth: authReducer,
  chat: chatReducer,
};

export const metaReducers: MetaReducer<AppState>[] = !environment.production ? [] : [];
