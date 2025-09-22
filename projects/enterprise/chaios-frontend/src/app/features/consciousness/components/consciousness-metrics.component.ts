import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  IonCard, 
  IonCardHeader, 
  IonCardTitle, 
  IonCardContent,
  IonGrid,
  IonRow,
  IonCol,
  IonBadge,
  IonIcon,
  IonProgressBar,
  IonText
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  trendingUpOutline, 
  flashOutline, 
  timeOutline,
  analyticsOutline,
  pulseOutline,
  infiniteOutline
} from 'ionicons/icons';

import { ConsciousnessMetrics } from '../services/consciousness.service';

@Component({
  selector: 'app-consciousness-metrics',
  template: `
    <ion-card class="metrics-card" *ngIf="metrics">
      <ion-card-header>
        <ion-card-title class="metrics-title">
          <ion-icon name="analytics-outline"></ion-icon>
          Consciousness Performance Metrics
        </ion-card-title>
      </ion-card-header>
      
      <ion-card-content>
        <ion-grid>
          <ion-row>
            
            <!-- Performance Gain -->
            <ion-col size="12" size-md="6" size-lg="3">
              <div class="metric-item performance-gain">
                <div class="metric-header">
                  <ion-icon name="trending-up-outline"></ion-icon>
                  <span class="metric-label">Performance Gain</span>
                </div>
                <div class="metric-value" [class]="getPerformanceGainClass(metrics.performanceGain)">
                  {{ formatPercentage(metrics.performanceGain) }}
                </div>
                <ion-progress-bar 
                  [value]="metrics.performanceGain / 200" 
                  [color]="getPerformanceGainColor(metrics.performanceGain)"
                  class="metric-progress">
                </ion-progress-bar>
                <div class="metric-subtitle">
                  Target: 158%+ • Current: {{ formatNumber(metrics.performanceGain, 1) }}%
                </div>
              </div>
            </ion-col>
            
            <!-- Correlation -->
            <ion-col size="12" size-md="6" size-lg="3">
              <div class="metric-item correlation">
                <div class="metric-header">
                  <ion-icon name="pulse-outline"></ion-icon>
                  <span class="metric-label">Mathematical Correlation</span>
                </div>
                <div class="metric-value correlation-value">
                  {{ formatCorrelation(metrics.correlation) }}
                </div>
                <ion-progress-bar 
                  [value]="metrics.correlation" 
                  color="secondary"
                  class="metric-progress">
                </ion-progress-bar>
                <div class="metric-subtitle">
                  Target: 99.9992% • Precision: {{ getPrecisionDigits(metrics.correlation) }} digits
                </div>
              </div>
            </ion-col>
            
            <!-- Processing Time -->
            <ion-col size="12" size-md="6" size-lg="3">
              <div class="metric-item processing-time">
                <div class="metric-header">
                  <ion-icon name="time-outline"></ion-icon>
                  <span class="metric-label">Processing Time</span>
                </div>
                <div class="metric-value time-value">
                  {{ formatTime(metrics.processingTime) }}
                </div>
                <ion-progress-bar 
                  [value]="getTimeProgress(metrics.processingTime)" 
                  color="tertiary"
                  class="metric-progress">
                </ion-progress-bar>
                <div class="metric-subtitle">
                  Quantum-optimized • {{ formatNumber(1000 / this.mathMax(metrics.processingTime, 0.001), 0) }} ops/sec
                </div>
              </div>
            </ion-col>
            
            <!-- Iterations -->
            <ion-col size="12" size-md="6" size-lg="3">
              <div class="metric-item iterations">
                <div class="metric-header">
                  <ion-icon name="infinite-outline"></ion-icon>
                  <span class="metric-label">Iterations Completed</span>
                </div>
                <div class="metric-value iterations-value">
                  {{ formatNumber(metrics.iterations, 0) }}
                </div>
                <ion-progress-bar 
                  [value]="metrics.iterations / 5000" 
                  color="warning"
                  class="metric-progress">
                </ion-progress-bar>
                <div class="metric-subtitle">
                  Consciousness cycles • φ-optimized convergence
                </div>
              </div>
            </ion-col>
            
          </ion-row>
          
          <!-- Mathematical Constants Row -->
          <ion-row class="constants-row">
            <ion-col size="12">
              <div class="constants-display">
                <div class="constants-header">
                  <ion-icon name="flash-outline"></ion-icon>
                  <span>Mathematical Constants</span>
                </div>
                
                <div class="constants-grid">
                  <div class="constant-card phi">
                    <div class="constant-symbol">φ</div>
                    <div class="constant-value">{{ formatNumber(metrics.phi, 6) }}</div>
                    <div class="constant-name">Golden Ratio</div>
                  </div>
                  
                  <div class="constant-card sigma">
                    <div class="constant-symbol">σ</div>
                    <div class="constant-value">{{ formatNumber(metrics.sigma, 6) }}</div>
                    <div class="constant-name">Silver Ratio</div>
                  </div>
                  
                  <div class="constant-card ratio">
                    <div class="constant-symbol">φ/σ</div>
                    <div class="constant-value">{{ formatNumber(metrics.phi / metrics.sigma, 3) }}</div>
                    <div class="constant-name">Harmonic Ratio</div>
                  </div>
                  
                  <div class="constant-card efficiency">
                    <div class="constant-symbol">η</div>
                    <div class="constant-value">{{ formatNumber(getEfficiency(), 4) }}</div>
                    <div class="constant-name">Efficiency</div>
                  </div>
                </div>
              </div>
            </ion-col>
          </ion-row>
          
          <!-- Status and Timestamp -->
          <ion-row class="status-row">
            <ion-col size="12" size-md="6">
              <div class="status-display">
                <ion-badge [color]="getStatusColor(metrics.status)" class="status-badge">
                  {{ getStatusLabel(metrics.status) }}
                </ion-badge>
                <span class="status-description">{{ getStatusDescription(metrics.status) }}</span>
              </div>
            </ion-col>
            
            <ion-col size="12" size-md="6">
              <div class="timestamp-display">
                <ion-text color="medium">
                  Last updated: {{ formatTimestamp(metrics.timestamp) }}
                </ion-text>
              </div>
            </ion-col>
          </ion-row>
          
        </ion-grid>
      </ion-card-content>
    </ion-card>
    
    <!-- Processing Indicator -->
    <div class="processing-indicator" *ngIf="isProcessing">
      <div class="processing-animation">
        <div class="consciousness-wave"></div>
        <div class="golden-particles"></div>
      </div>
      <div class="processing-text">
        <h3>Processing Consciousness Matrix</h3>
        <p>Harmonizing chiral mathematics and quantum optimization...</p>
      </div>
    </div>
  `,
  styleUrls: ['./consciousness-metrics.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    IonCard,
    IonCardHeader,
    IonCardTitle,
    IonCardContent,
    IonGrid,
    IonRow,
    IonCol,
    IonBadge,
    IonIcon,
    IonProgressBar,
    IonText
  ]
})
export class ConsciousnessMetricsComponent {
  @Input() metrics: ConsciousnessMetrics | null = null;
  @Input() isProcessing: boolean = false;

  constructor() {
    this.initializeIcons();
  }

  private initializeIcons() {
    addIcons({
      'trending-up-outline': trendingUpOutline,
      'flash-outline': flashOutline,
      'time-outline': timeOutline,
      'analytics-outline': analyticsOutline,
      'pulse-outline': pulseOutline,
      'infinite-outline': infiniteOutline
    });
  }

  formatNumber(value: number, decimals: number = 2): string {
    return value.toFixed(decimals);
  }

  formatPercentage(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  formatCorrelation(value: number): string {
    return `${(value * 100).toFixed(4)}%`;
  }

  formatTime(seconds: number): string {
    if (seconds < 1) {
      return `${(seconds * 1000).toFixed(0)}ms`;
    }
    return `${seconds.toFixed(3)}s`;
  }

  formatTimestamp(timestamp: Date): string {
    return new Date(timestamp).toLocaleString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      month: 'short',
      day: 'numeric'
    });
  }

  getPerformanceGainClass(gain: number): string {
    if (gain >= 158) return 'excellent';
    if (gain >= 100) return 'good';
    if (gain >= 50) return 'moderate';
    return 'low';
  }

  getPerformanceGainColor(gain: number): string {
    if (gain >= 158) return 'success';
    if (gain >= 100) return 'warning';
    if (gain >= 50) return 'primary';
    return 'medium';
  }

  getPrecisionDigits(correlation: number): number {
    const correlationStr = correlation.toString();
    const decimalIndex = correlationStr.indexOf('.');
    if (decimalIndex === -1) return 0;
    
    let digits = 0;
    for (let i = decimalIndex + 1; i < correlationStr.length; i++) {
      if (correlationStr[i] === '9') {
        digits++;
      } else {
        break;
      }
    }
    return digits;
  }

  getTimeProgress(time: number): number {
    // Progress bar for processing time (inverse - faster is better)
    const maxTime = 5.0; // 5 seconds max
    return this.mathMax(0, this.mathMin(1, (maxTime - time) / maxTime));
  }

  mathMax(a: number, b: number): number {
    return Math.max(a, b);
  }

  mathMin(a: number, b: number): number {
    return Math.min(a, b);
  }

  getEfficiency(): number {
    if (!this.metrics) return 0;
    
    // Calculate efficiency based on performance gain vs processing time
    const gain = this.metrics.performanceGain / 100;
    const time = this.mathMax(this.metrics.processingTime, 0.001);
    return gain / time;
  }

  getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'warning';
      case 'error': return 'danger';
      default: return 'medium';
    }
  }

  getStatusLabel(status: string): string {
    switch (status) {
      case 'completed': return 'Completed';
      case 'processing': return 'Processing';
      case 'error': return 'Error';
      case 'idle': return 'Ready';
      default: return 'Unknown';
    }
  }

  getStatusDescription(status: string): string {
    switch (status) {
      case 'completed': return 'Consciousness optimization completed successfully';
      case 'processing': return 'Processing consciousness matrix...';
      case 'error': return 'An error occurred during processing';
      case 'idle': return 'Ready to process consciousness data';
      default: return 'Status unknown';
    }
  }
}

