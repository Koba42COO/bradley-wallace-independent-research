import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { Data, DataResponse } from '../models/data.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) { }

  getData(params?: {
    category?: string;
    search?: string;
    page?: number;
    limit?: number;
  }): Observable<DataResponse> {
    let httpParams = new HttpParams();
    if (params) {
      if (params.category) httpParams = httpParams.set('category', params.category);
      if (params.search) httpParams = httpParams.set('search', params.search);
      if (params.page) httpParams = httpParams.set('page', params.page.toString());
      if (params.limit) httpParams = httpParams.set('limit', params.limit.toString());
    }
    return this.http.get<DataResponse>(`${this.apiUrl}/data`, { params: httpParams });
  }

  getDataById(id: string): Observable<{ success: boolean; data: Data }> {
    return this.http.get<{ success: boolean; data: Data }>(`${this.apiUrl}/data/${id}`);
  }

  createData(data: Partial<Data>): Observable<{ success: boolean; data: Data }> {
    return this.http.post<{ success: boolean; data: Data }>(`${this.apiUrl}/data`, data);
  }

  updateData(id: string, data: Partial<Data>): Observable<{ success: boolean; data: Data }> {
    return this.http.put<{ success: boolean; data: Data }>(`${this.apiUrl}/data/${id}`, data);
  }

  deleteData(id: string): Observable<{ success: boolean; message: string }> {
    return this.http.delete<{ success: boolean; message: string }>(`${this.apiUrl}/data/${id}`);
  }
}
