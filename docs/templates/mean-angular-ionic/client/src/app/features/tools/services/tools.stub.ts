import { Injectable } from '@angular/core';
import { of } from 'rxjs';
import { ToolsPort } from './tools.port';

@Injectable({ providedIn: 'root' })
export class ToolsStub implements ToolsPort {
  invoke(tool: string) { return of({ ok: true, tool, source: 'stub' }); }
  list() { return of(['ai-editor', 'img-gen']); }
}


