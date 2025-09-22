import { createAction, props } from '@ngrx/store';

export const sendMessage = createAction(
  '[Chat] Send Message',
  props<{ message: string }>()
);

export const sendMessageSuccess = createAction(
  '[Chat] Send Message Success'
);

export const receiveMessage = createAction(
  '[Chat] Receive Message',
  props<{ message: any }>()
);
