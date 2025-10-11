import { Injectable, inject } from '@angular/core';
import { ApiService } from '../../../core/services/api.service';
import { ToolsPort } from './tools.port';

@Injectable({ providedIn: 'root' })
export class ToolsService implements ToolsPort {
  private api = inject(ApiService);
  invoke(tool: string) { return this.api.post(`/tools/${tool}/invoke`, {}); }
  list() { return this.api.get<string[]>(`/tools`); }
}


