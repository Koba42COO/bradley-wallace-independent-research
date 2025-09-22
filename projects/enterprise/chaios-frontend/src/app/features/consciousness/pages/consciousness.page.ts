import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { 
  IonContent, 
  IonHeader, 
  IonTitle, 
  IonToolbar, 
  IonCard, 
  IonCardHeader, 
  IonCardTitle, 
  IonCardContent,
  IonButton,
  IonIcon,
  IonBadge,
  IonProgressBar,
  IonRange,
  IonItem,
  IonLabel,
  IonInput,
  IonSelect,
  IonSelectOption,
  IonGrid,
  IonRow,
  IonCol,
  IonMenuButton,
  IonButtons,
  IonSpinner,
  IonChip,
  IonText
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  playOutline, 
  pauseOutline, 
  stopOutline, 
  refreshOutline,
  analyticsOutline,
  downloadOutline,
  settingsOutline,
  informationCircleOutline,
  trendingUpOutline,
  pulseOutline
} from 'ionicons/icons';
import { Subscription } from 'rxjs';

import { ConsciousnessService, ConsciousnessMetrics, ConsciousnessVisualization } from '../services/consciousness.service';
import { GoldenSpiralComponent } from '../components/golden-spiral.component';
import { ConsciousnessMetricsComponent } from '../components/consciousness-metrics.component';
// import { HarmonicVisualizerComponent } from '../components/harmonic-visualizer.component';

@Component({
  selector: 'app-consciousness',
  templateUrl: './consciousness.page.html',
  styleUrls: ['./consciousness.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    IonContent,
    IonHeader,
    IonTitle,
    IonToolbar,
    IonCard,
    IonCardHeader,
    IonCardTitle,
    IonCardContent,
    IonButton,
    IonIcon,
    IonBadge,
    IonProgressBar,
    IonRange,
    IonItem,
    IonLabel,
    IonInput,
    IonSelect,
    IonSelectOption,
    IonGrid,
    IonRow,
    IonCol,
    IonMenuButton,
    IonButtons,
    IonSpinner,
    IonChip,
    IonText,
    GoldenSpiralComponent,
    ConsciousnessMetricsComponent
    // HarmonicVisualizerComponent - coming soon
  ]
})
export class ConsciousnessPage implements OnInit, OnDestroy {
  @ViewChild(IonContent, { static: true }) content!: IonContent;

  // Processing parameters
  selectedAlgorithm = 'wallace_transform';
  iterations = 1000;
  optimizationLevel = 'maximum';
  
  // State management
  currentMetrics: ConsciousnessMetrics | null = null;
  visualization: ConsciousnessVisualization | null = null;
  isProcessing = false;
  isMonitoring = false;
  
  // Algorithm options
  algorithms = [
    { value: 'wallace_transform', label: 'Wallace Transform', description: 'Primary consciousness optimization' },
    { value: 'golden_ratio_enhancement', label: 'Golden Ratio Enhancement', description: 'φ-based mathematical optimization' },
    { value: 'harmonic_alignment', label: 'Harmonic Alignment', description: 'Frequency-based consciousness tuning' },
    { value: 'quantum_consciousness', label: 'Quantum Consciousness', description: 'Quantum-enhanced processing' },
    { value: 'chiral_optimization', label: 'Chiral Optimization', description: 'Chiral mathematics integration' }
  ];

  optimizationLevels = [
    { value: 'basic', label: 'Basic', multiplier: 1.0 },
    { value: 'enhanced', label: 'Enhanced', multiplier: 1.5 },
    { value: 'maximum', label: 'Maximum', multiplier: 2.0 },
    { value: 'transcendent', label: 'Transcendent', multiplier: 3.14159 }
  ];

  private subscriptions: Subscription[] = [];

  constructor(private consciousnessService: ConsciousnessService) {
    this.initializeIcons();
  }

  ngOnInit() {
    this.setupSubscriptions();
    this.loadInitialData();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
    this.consciousnessService.stopRealTimeMonitoring();
  }

  private initializeIcons() {
    addIcons({
      'play-outline': playOutline,
      'pause-outline': pauseOutline,
      'stop-outline': stopOutline,
      'refresh-outline': refreshOutline,
      'analytics-outline': analyticsOutline,
      'download-outline': downloadOutline,
      'settings-outline': settingsOutline,
      'information-circle-outline': informationCircleOutline,
      'trending-up-outline': trendingUpOutline,
      'pulse-outline': pulseOutline
    });
  }

  private setupSubscriptions() {
    // Consciousness metrics
    this.subscriptions.push(
      this.consciousnessService.metrics$.subscribe(metrics => {
        this.currentMetrics = metrics;
      })
    );

    // Processing state
    this.subscriptions.push(
      this.consciousnessService.isProcessing$.subscribe(processing => {
        this.isProcessing = processing;
      })
    );

    // Visualization data
    this.subscriptions.push(
      this.consciousnessService.visualization$.subscribe(viz => {
        this.visualization = viz;
      })
    );
  }

  private async loadInitialData() {
    try {
      // Load current metrics
      this.currentMetrics = this.consciousnessService.getCurrentMetrics();
      
      // Load processing history if needed
      const history = await this.consciousnessService.getProcessingHistory();
      console.log('Consciousness processing history:', history.length, 'entries');
      
    } catch (error) {
      console.warn('Failed to load initial consciousness data:', error);
    }
  }

  async startProcessing() {
    if (this.isProcessing) return;

    try {
      console.log('🧠 Starting consciousness processing:', {
        algorithm: this.selectedAlgorithm,
        iterations: this.iterations,
        optimization: this.optimizationLevel
      });

      await this.consciousnessService.processConsciousness(
        this.selectedAlgorithm,
        this.iterations
      );

    } catch (error) {
      console.error('Failed to start consciousness processing:', error);
    }
  }

  stopProcessing() {
    console.log('⏹️ Stopping consciousness processing');
    this.consciousnessService.stopRealTimeMonitoring();
  }

  resetProcessing() {
    console.log('🔄 Resetting consciousness processing');
    this.consciousnessService.resetMetrics();
    this.currentMetrics = this.consciousnessService.getCurrentMetrics();
  }

  toggleRealTimeMonitoring() {
    if (this.isMonitoring) {
      this.consciousnessService.stopRealTimeMonitoring();
      this.isMonitoring = false;
    } else {
      this.subscriptions.push(
        this.consciousnessService.startRealTimeMonitoring().subscribe()
      );
      this.isMonitoring = true;
    }
  }

  async runDemo() {
    console.log('🎭 Running consciousness processing demo');
    await this.consciousnessService.runDemo();
  }

  onAlgorithmChange() {
    console.log('Algorithm changed to:', this.selectedAlgorithm);
  }

  onIterationsChange() {
    console.log('Iterations changed to:', this.iterations);
  }

  onOptimizationLevelChange() {
    console.log('Optimization level changed to:', this.optimizationLevel);
  }

  downloadResults() {
    if (!this.currentMetrics) return;

    const data = {
      metrics: this.currentMetrics,
      visualization: this.visualization,
      timestamp: new Date().toISOString(),
      algorithm: this.selectedAlgorithm,
      parameters: {
        iterations: this.iterations,
        optimization: this.optimizationLevel
      }
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { 
      type: 'application/json' 
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `consciousness-results-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Utility methods
  getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'warning';
      case 'error': return 'danger';
      default: return 'medium';
    }
  }

  getPerformanceGainColor(gain: number): string {
    if (gain > 150) return 'success';
    if (gain > 100) return 'warning';
    if (gain > 50) return 'primary';
    return 'medium';
  }

  formatNumber(value: number, decimals: number = 2): string {
    return value.toFixed(decimals);
  }

  formatPercentage(value: number): string {
    return `${(value * 100).toFixed(4)}%`;
  }

  getAlgorithmDescription(algorithm: string): string {
    const algo = this.algorithms.find(a => a.value === algorithm);
    return algo?.description || 'Advanced consciousness processing algorithm';
  }

  getOptimizationMultiplier(): number {
    const level = this.optimizationLevels.find(l => l.value === this.optimizationLevel);
    return level?.multiplier || 1.0;
  }

  getEstimatedProcessingTime(): number {
    const baseTime = this.iterations / 1000; // Base time in seconds
    const multiplier = this.getOptimizationMultiplier();
    return baseTime * multiplier;
  }

  // Mathematical calculations
  calculatePhi(): number {
    return (1 + Math.sqrt(5)) / 2;
  }

  calculateSigma(): number {
    return 1 / this.calculatePhi() - 1;
  }

  // Progress calculation
  getProcessingProgress(): number {
    if (!this.currentMetrics || this.currentMetrics.status !== 'processing') {
      return 0;
    }
    
    // Simulate progress based on time elapsed
    const elapsed = Date.now() - this.currentMetrics.timestamp.getTime();
    const estimated = this.getEstimatedProcessingTime() * 1000;
    return Math.min(elapsed / estimated, 0.95); // Max 95% until completion
  }
}

