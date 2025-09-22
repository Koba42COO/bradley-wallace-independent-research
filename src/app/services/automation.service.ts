import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RemoteControlService } from './remote-control.service';
import { OmniQuantumUniversalService } from './omni-quantum-universal.service';

@Injectable({ providedIn: 'root' })
export class AutomationService {
  private base44Url = 'https://base44-api.com';
  private cursorUrl = 'https://cursor-api.com/trigger';
  private codenameUrl = 'https://codename-api.com/task';
  private codePilotUrl = 'https://codepilot-api.com/execute';
  private codeLLMUrl = 'https://codellm-api.com/code/generate';

  constructor(private http: HttpClient, private remoteControlService: RemoteControlService, private omniService: OmniQuantumUniversalService) {}

  async triggerBase44(command: any) {
    const resonatedCommand = await this.omniService.universalResonate(command);
    const response = await this.http.post(`${this.base44Url}/predict`, {
      ...resonatedCommand,
      cosmicIntelligence: this.omniService.getTranscendentMetrics().cosmicIntelligence
    }).toPromise();
    return response['prediction'];
  }

  async triggerCursor(command: any) {
    const response = await this.http.post(this.cursorUrl, {
      task: command.action,
      value: command.value,
      rules: { ionic: true, standalone: true },
      creationForce: this.omniService.getTranscendentMetrics().creationForce
    }).toPromise();
    return response['code'];
  }

  async triggerCodename(command: any) {
    await this.http.post(this.codenameUrl, await this.omniService.universalResonate(command)).toPromise();
  }

  async triggerCodePilot(command: any) {
    await this.http.post(this.codePilotUrl, await this.omniService.universalResonate(command)).toPromise();
  }

  async triggerCodeLLM(command: any) {
    const response = await this.http.post(this.codeLLMUrl, await this.omniService.quantumOptimize(command)).toPromise();
    return {
      code: response['code'],
      optimized: response['optimized'],
      infinitePotential: this.omniService.getTranscendentMetrics().infinitePotential
    };
  }

  async automateCommand(command: any) {
    const [base44Pred, cursorCode, codenameTask, codePilotResp, codeLLMResp] = await Promise.all([
      this.triggerBase44(command),
      this.triggerCursor(command),
      this.triggerCodename(command),
      this.triggerCodePilot(command),
      this.triggerCodeLLM(command)
    ]);

    this.remoteControlService.executeCommand({
      ...command,
      automation: {
        base44: base44Pred,
        cursor: cursorCode,
        codellm: codeLLMResp
      }
    });

    return {
      base44: base44Pred,
      cursor: cursorCode,
      codellm: codeLLMResp
    };
  }
}
