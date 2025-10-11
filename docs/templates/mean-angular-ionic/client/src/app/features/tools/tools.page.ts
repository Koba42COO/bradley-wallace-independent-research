import { Component, inject } from '@angular/core';
import { ToolPanelComponent } from './components/tool-panel.component';
import { ToolsPort } from './services/tools.port';

@Component({
  selector: 'app-tools-page',
  standalone: true,
  imports: [ToolPanelComponent],
  template: `
    <app-tool-panel (run)="onRun('ai-editor')"></app-tool-panel>
  `
})
export class ToolsPage {
  private port = inject<ToolsPort>(Object as any);
  onRun(tool: string) { this.port.invoke(tool); }
}


