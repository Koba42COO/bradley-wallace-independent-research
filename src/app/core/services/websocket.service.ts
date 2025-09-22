import { Injectable } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { environment } from '../../../environments/environment';
import { Observable, Subject } from 'rxjs';
import { share, switchMap, retryWhen, delay } from 'rxjs/operators';

export interface Message {
  type: string;
  payload: any;
}

@Injectable({
  providedIn: 'root'
})
export class WebsocketService {
  private socket$: WebSocketSubject<Message>;
  private connection$: Observable<Message>;
  private connectionRetries = 5;

  constructor() {
    const wsUrl = environment.apiUrl.replace(/^http/, 'ws');
    const clientId = `client-${Math.random().toString(36).substr(2, 9)}`;
    this.socket$ = webSocket<Message>(`${wsUrl}/ws/${clientId}`);

    this.connection$ = this.socket$.pipe(
      share(),
      retryWhen(errors => errors.pipe(delay(2000)))
    );
  }

  public connect(): Observable<Message> {
    return this.connection$;
  }

  public sendMessage(type: string, payload: any): void {
    const message: Message = { type, payload };
    this.socket$.next(message);
  }

  public closeConnection(): void {
    this.socket$.complete();
  }
}
