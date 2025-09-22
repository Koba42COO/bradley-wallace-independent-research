import { Injectable } from '@angular/core';
import { AutomationService } from './automation.service';
import { OmniQuantumUniversalService } from './omni-quantum-universal.service';

@Injectable({ providedIn: 'root' })
export class TodoService {
  todos: any[] = [];

  constructor(private automationService: AutomationService, private omniService: OmniQuantumUniversalService) {}

  async addTodo(text: string) {
    let command = this.parseCommand(text);
    command = await this.omniService.enhanceWithConsciousness(command); // OMNI enhancement
    command = await this.omniService.quantumOptimize(command); // Quantum optimization
    command.transcendentUnity = this.omniService.getTranscendentMetrics().transcendentUnity; // Apply unity
    this.todos.push(command);
    this.automationService.automateCommand(command);
  }

  parseCommand(text: string) {
    const actions = {
      'open': text.match(/open\s+(\w+)/)?.[1],
      'move cursor': text.match(/move cursor to (\d+,\s*\d+)/)?.[1],
      'type': text.match(/type\s+(.+)/)?.[1]
    };
    return {
      action: Object.keys(actions).find(k => actions[k]),
      value: actions[Object.keys(actions).find(k => actions[k])] || text,
      priority: 'high'
    };
  }
}
