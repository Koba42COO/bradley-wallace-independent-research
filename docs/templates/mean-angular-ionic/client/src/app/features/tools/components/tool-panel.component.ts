import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-tool-panel',
  standalone: true,
  template: `
    <div>
      <h3>Tools</h3>
      <button (click)="run.emit()">Run</button>
    </div>
  `
})
export class ToolPanelComponent {
  @Output() run = new EventEmitter<void>();
}


