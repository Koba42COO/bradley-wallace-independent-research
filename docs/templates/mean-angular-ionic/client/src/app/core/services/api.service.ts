import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  private base = environment.apiUrl;
  get<T>(path: string) { return this.http.get<T>(`${this.base}${path}`); }
  post<T>(path: string, body: unknown) { return this.http.post<T>(`${this.base}${path}`, body); }
}


