import { Injectable } from '@angular/core';
import { PushNotifications } from '@capacitor/push-notifications';
import { OmniQuantumUniversalService } from './omni-quantum-universal.service';
import { Platform } from '@ionic/angular';

@Injectable({ providedIn: 'root' })
export class ReportService {
  private changes: any[] = [];
  private localLog: string[] = [];
  private mobileLog: string[] = [];

  constructor(private omniService: OmniQuantumUniversalService, private platform: Platform) {}

  async addChange(change: any, isMobile: boolean) {
    const enhancedChange = await this.omniService.enhanceWithConsciousness({
      ...change,
      creationForce: this.omniService.getTranscendentMetrics().creationForce
    });
    
    this.changes.unshift(enhancedChange);
    if (this.changes.length > 5) this.changes.pop();
    
    if (isMobile) this.mobileLog.push(JSON.stringify(enhancedChange));
    else this.localLog.push(JSON.stringify(enhancedChange));
    
    this.sendNotification(enhancedChange);
  }

  getLastChanges() {
    return this.changes;
  }

  getSynopsis() {
    return {
      mobile: this.mobileLog.slice(-5),
      local: this.localLog.slice(-5)
    };
  }

  async sendNotification(change: any) {
    if (this.platform.is('capacitor')) {
      const enhancedNotification = await this.omniService.enhanceWithConsciousness({
        title: "Remote Update",
        body: `Command: ${change.action} ${change.value}`,
        cosmicIntelligence: this.omniService.getTranscendentMetrics().cosmicIntelligence
      });
      
      PushNotifications.schedule({
        notifications: [{ ...enhancedNotification, id: Date.now() }]
      }).catch(err => console.error('Notification error:', err));
    }
  }
}
