import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  IonContent, 
  IonHeader, 
  IonTitle, 
  IonToolbar,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
  IonGrid,
  IonRow,
  IonCol,
  IonButton,
  IonIcon,
  IonBadge,
  IonProgressBar,
  IonText,
  IonItem,
  IonLabel
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  analyticsOutline,
  trendingUpOutline,
  speedometerOutline,
  shieldOutline,
  peopleOutline
} from 'ionicons/icons';
import { Subscription } from 'rxjs';

import { AnalyticsService, SystemMetrics, PerformanceMetrics } from './services/analytics.service';

@Component({
  selector: 'app-analytics',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>Analytics Dashboard</ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true">
      <div class="analytics-container">
        
        <!-- System Metrics -->
        <ion-card *ngIf="systemMetrics">
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="speedometer-outline"></ion-icon>
              System Performance
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-grid>
              <ion-row>
                <ion-col size="6" size-md="3">
                  <ion-item>
                    <ion-label>CPU Usage</ion-label>
                    <ion-badge [color]="getCpuColor(systemMetrics.cpu.usage)">
                      {{ systemMetrics.cpu.usage.toFixed(1) }}%
                    </ion-badge>
                  </ion-item>
                  <ion-progress-bar [value]="systemMetrics.cpu.usage / 100" [color]="getCpuColor(systemMetrics.cpu.usage)"></ion-progress-bar>
                </ion-col>
                <ion-col size="6" size-md="3">
                  <ion-item>
                    <ion-label>Memory</ion-label>
                    <ion-badge [color]="getMemoryColor(systemMetrics.memory.percentage)">
                      {{ systemMetrics.memory.percentage.toFixed(1) }}%
                    </ion-badge>
                  </ion-item>
                  <ion-progress-bar [value]="systemMetrics.memory.percentage / 100" [color]="getMemoryColor(systemMetrics.memory.percentage)"></ion-progress-bar>
                </ion-col>
                <ion-col size="6" size-md="3">
                  <ion-item>
                    <ion-label>Disk Usage</ion-label>
                    <ion-badge color="tertiary">{{ systemMetrics.disk.percentage.toFixed(1) }}%</ion-badge>
                  </ion-item>
                  <ion-progress-bar [value]="systemMetrics.disk.percentage / 100" color="tertiary"></ion-progress-bar>
                </ion-col>
                <ion-col size="6" size-md="3">
                  <ion-item>
                    <ion-label>Network</ion-label>
                    <ion-badge color="success">{{ formatBytes(systemMetrics.network.bytesIn) }}/s</ion-badge>
                  </ion-item>
                </ion-col>
              </ion-row>
            </ion-grid>
          </ion-card-content>
        </ion-card>

        <!-- Performance Metrics -->
        <ion-card *ngIf="performanceMetrics">
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="trending-up-outline"></ion-icon>
              Performance Metrics
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-grid>
              <ion-row>
                <ion-col size="6" size-md="4">
                  <ion-item>
                    <ion-label>API Response Time</ion-label>
                    <ion-badge color="primary">{{ performanceMetrics.apiResponseTime.toFixed(0) }}ms</ion-badge>
                  </ion-item>
                </ion-col>
                <ion-col size="6" size-md="4">
                  <ion-item>
                    <ion-label>Throughput</ion-label>
                    <ion-badge color="secondary">{{ performanceMetrics.throughput.toFixed(0) }} req/s</ion-badge>
                  </ion-item>
                </ion-col>
                <ion-col size="6" size-md="4">
                  <ion-item>
                    <ion-label>Error Rate</ion-label>
                    <ion-badge [color]="getErrorRateColor(performanceMetrics.errorRate)">
                      {{ performanceMetrics.errorRate.toFixed(2) }}%
                    </ion-badge>
                  </ion-item>
                </ion-col>
                <ion-col size="6" size-md="4">
                  <ion-item>
                    <ion-label>Uptime</ion-label>
                    <ion-badge color="success">{{ performanceMetrics.uptime.toFixed(2) }}%</ion-badge>
                  </ion-item>
                </ion-col>
              </ion-row>
            </ion-grid>
          </ion-card-content>
        </ion-card>

        <!-- Placeholder for more analytics -->
        <ion-card>
          <ion-card-header>
            <ion-card-title>
              <ion-icon name="analytics-outline"></ion-icon>
              Advanced Analytics
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-text>
              <p>Comprehensive analytics dashboard coming soon...</p>
              <p>Features in development:</p>
              <ul>
                <li>User behavior tracking</li>
                <li>Consciousness processing insights</li>
                <li>Security threat analysis</li>
                <li>Business intelligence reports</li>
              </ul>
            </ion-text>
            <ion-button expand="block" fill="outline" disabled>
              <ion-icon name="analytics-outline" slot="start"></ion-icon>
              Advanced Analytics (Coming Soon)
            </ion-button>
          </ion-card-content>
        </ion-card>

      </div>
    </ion-content>
  `,
  styleUrls: ['./analytics.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    IonContent,
    IonHeader,
    IonTitle,
    IonToolbar,
    IonCard,
    IonCardHeader,
    IonCardTitle,
    IonCardContent,
    IonGrid,
    IonRow,
    IonCol,
    IonButton,
    IonIcon,
    IonBadge,
    IonProgressBar,
    IonText,
    IonItem,
    IonLabel
  ]
})
export class AnalyticsPage implements OnInit, OnDestroy {
  systemMetrics: SystemMetrics | null = null;
  performanceMetrics: PerformanceMetrics | null = null;
  
  private subscriptions: Subscription[] = [];

  constructor(private analyticsService: AnalyticsService) {
    this.initializeIcons();
  }

  ngOnInit() {
    this.subscriptions.push(
      this.analyticsService.systemMetrics$.subscribe(metrics => {
        this.systemMetrics = metrics;
      }),
      
      this.analyticsService.performanceMetrics$.subscribe(metrics => {
        this.performanceMetrics = metrics;
      })
    );
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  private initializeIcons() {
    addIcons({
      'analytics-outline': analyticsOutline,
      'trending-up-outline': trendingUpOutline,
      'speedometer-outline': speedometerOutline,
      'shield-outline': shieldOutline,
      'people-outline': peopleOutline
    });
  }

  getCpuColor(usage: number): string {
    if (usage > 80) return 'danger';
    if (usage > 60) return 'warning';
    return 'success';
  }

  getMemoryColor(usage: number): string {
    if (usage > 90) return 'danger';
    if (usage > 70) return 'warning';
    return 'primary';
  }

  getErrorRateColor(rate: number): string {
    if (rate > 5) return 'danger';
    if (rate > 2) return 'warning';
    return 'success';
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}
