import { Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { of } from 'rxjs';
import { catchError, map, mergeMap, tap } from 'rxjs/operators';
import { WebsocketService } from '../core/services/websocket.service';
import * as ChatActions from './chat.actions';

@Injectable()
export class ChatEffects {
  constructor(
    private actions$: Actions,
    private websocketService: WebsocketService
  ) {}

  connect$ = createEffect(() =>
    this.websocketService.connect().pipe(
      map(message => ChatActions.receiveMessage({ message })),
      catchError(error => of({ type: '[Chat] WebSocket Error', error }))
    )
  );

  sendMessage$ = createEffect(() =>
    this.actions$.pipe(
      ofType(ChatActions.sendMessage),
      tap(({ message }) => this.websocketService.sendMessage('chat_message', { message })),
      map(() => ChatActions.sendMessageSuccess())
    )
  );
}
