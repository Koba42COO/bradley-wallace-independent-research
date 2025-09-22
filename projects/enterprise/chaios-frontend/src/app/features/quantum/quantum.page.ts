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
  IonGrid,
  IonRow,
  IonCol,
  IonButton,
  IonIcon,
  IonBadge,
  IonItem,
  IonLabel,
  IonSelect,
  IonSelectOption,
  IonRange,
  IonCheckbox,
  IonProgressBar,
  IonText,
  IonFab,
  IonFabButton,
  IonSegment,
  IonSegmentButton,
  IonList,
  IonItemSliding,
  IonItemOptions,
  IonItemOption,
  IonAlert,
  AlertController,
  ToastController
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  atOutline,
  playOutline,
  pauseOutline,
  stopOutline,
  refreshOutline,
  settingsOutline,
  analyticsOutline,
  downloadOutline,
  expandOutline,
  contractOutline,
  eyeOutline,
  flashOutline,
  pulseOutline,
  infiniteOutline,
  layersOutline,
  gridOutline,
  barChartOutline,
  trendingUpOutline,
  speedometerOutline,
  nuclearOutline,
  radioOutline
} from 'ionicons/icons';
import { Subscription } from 'rxjs';

import { 
  QuantumService, 
  QuantumSystem, 
  QuantumSimulationConfig, 
  QuantumConsciousnessMetrics,
  QuantumMeasurement 
} from './services/quantum.service';

@Component({
  selector: 'app-quantum',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>
          <div class="page-title">
            <ion-icon name="at-outline" class="title-icon"></ion-icon>
            Quantum Consciousness Simulation
            <ion-badge color="secondary" class="version-badge">v2.0</ion-badge>
          </div>
        </ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true" class="quantum-content">
      
      <!-- Status Bar -->
      <div class="status-bar">
        <div class="status-content">
          <div class="system-status">
            <ion-icon name="nuclear-outline" class="status-icon"></ion-icon>
            <span class="status-text">{{ currentSystem?.name || 'No System' }}</span>
            <ion-badge [color]="getStatusColor(currentSystem?.status)" class="status-badge">
              {{ getStatusLabel(currentSystem?.status) }}
            </ion-badge>
          </div>
          
          <div class="quick-stats" *ngIf="metrics">
            <div class="stat-item">
              <span class="stat-label">Coherence</span>
              <span class="stat-value">{{ formatNumber(metrics.quantumCoherence * 100, 1) }}%</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Entanglement</span>
              <span class="stat-value">{{ formatNumber(metrics.entanglementEntropy, 2) }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Consciousness</span>
              <span class="stat-value">{{ formatNumber(metrics.consciousnessIndex * 100, 1) }}%</span>
            </div>
          </div>

          <div class="control-buttons">
            <ion-button 
              fill="outline" 
              size="small" 
              [color]="isSimulating ? 'danger' : 'primary'"
              (click)="toggleSimulation()"
              [disabled]="!currentSystem">
              <ion-icon [name]="isSimulating ? 'pause-outline' : 'play-outline'" slot="start"></ion-icon>
              {{ isSimulating ? 'Pause' : 'Start' }}
            </ion-button>
            
            <ion-button 
              fill="outline" 
              size="small" 
              color="warning"
              (click)="stopSimulation()"
              [disabled]="!isSimulating">
              <ion-icon name="stop-outline" slot="start"></ion-icon>
              Stop
            </ion-button>
            
            <ion-button 
              fill="outline" 
              size="small" 
              color="secondary"
              (click)="resetSystem()">
              <ion-icon name="refresh-outline" slot="start"></ion-icon>
              Reset
            </ion-button>
          </div>
        </div>
      </div>

      <!-- Main Content Tabs -->
      <div class="quantum-tabs">
        <ion-segment 
          [(ngModel)]="selectedTab" 
          (ionChange)="onTabChange($event)"
          color="primary"
          class="tab-segment">
          
          <ion-segment-button value="simulation">
            <ion-icon name="at-outline"></ion-icon>
            <ion-label>Simulation</ion-label>
          </ion-segment-button>
          
          <ion-segment-button value="visualization">
            <ion-icon name="eye-outline"></ion-icon>
            <ion-label>Visualization</ion-label>
          </ion-segment-button>
          
          <ion-segment-button value="metrics">
            <ion-icon name="analytics-outline"></ion-icon>
            <ion-label>Metrics</ion-label>
          </ion-segment-button>
          
          <ion-segment-button value="circuit">
            <ion-icon name="grid-outline"></ion-icon>
            <ion-label>Circuit</ion-label>
          </ion-segment-button>
          
        </ion-segment>
      </div>

      <!-- Tab Content -->
      <div class="tab-content">
        
        <!-- Simulation Tab -->
        <div class="tab-pane" [class.active]="selectedTab === 'simulation'">
          <ion-grid>
            <ion-row>
              
              <!-- Configuration Panel -->
              <ion-col size="12" size-lg="4">
                <ion-card class="config-card">
                  <ion-card-header>
                    <ion-card-title>
                      <ion-icon name="settings-outline"></ion-icon>
                      Simulation Configuration
                    </ion-card-title>
                  </ion-card-header>
                  
                  <ion-card-content>
                    <div class="config-form">
                      
                      <!-- Algorithm Selection -->
                      <ion-item class="config-item">
                        <ion-label position="stacked">Quantum Algorithm</ion-label>
                        <ion-select 
                          [(ngModel)]="simulationConfig.algorithm" 
                          interface="popover"
                          class="algorithm-select">
                          <ion-select-option value="consciousness">Consciousness Simulation</ion-select-option>
                          <ion-select-option value="grover">Grover's Algorithm</ion-select-option>
                          <ion-select-option value="shor">Shor's Algorithm</ion-select-option>
                          <ion-select-option value="quantum_fourier">Quantum Fourier Transform</ion-select-option>
                          <ion-select-option value="variational">Variational Quantum Eigensolver</ion-select-option>
                        </ion-select>
                      </ion-item>

                      <!-- Qubits -->
                      <div class="range-control">
                        <ion-label class="range-label">Qubits: {{ simulationConfig.qubits }}</ion-label>
                        <ion-range
                          min="2"
                          max="16"
                          step="1"
                          [(ngModel)]="simulationConfig.qubits"
                          color="primary"
                          class="config-range">
                          <div slot="start">2</div>
                          <div slot="end">16</div>
                        </ion-range>
                      </div>

                      <!-- Iterations -->
                      <div class="range-control">
                        <ion-label class="range-label">Iterations: {{ simulationConfig.iterations }}</ion-label>
                        <ion-range
                          min="100"
                          max="10000"
                          step="100"
                          [(ngModel)]="simulationConfig.iterations"
                          color="secondary"
                          class="config-range">
                          <div slot="start">100</div>
                          <div slot="end">10K</div>
                        </ion-range>
                      </div>

                      <!-- Decoherence -->
                      <div class="range-control">
                        <ion-label class="range-label">Decoherence: {{ formatNumber(simulationConfig.decoherence, 3) }}</ion-label>
                        <ion-range
                          min="0"
                          max="0.1"
                          step="0.001"
                          [(ngModel)]="simulationConfig.decoherence"
                          color="tertiary"
                          class="config-range">
                          <div slot="start">0</div>
                          <div slot="end">0.1</div>
                        </ion-range>
                      </div>

                      <!-- Temperature -->
                      <div class="range-control">
                        <ion-label class="range-label">Temperature: {{ formatNumber(simulationConfig.temperature, 2) }}K</ion-label>
                        <ion-range
                          min="0.001"
                          max="1"
                          step="0.001"
                          [(ngModel)]="simulationConfig.temperature"
                          color="warning"
                          class="config-range">
                          <div slot="start">1mK</div>
                          <div slot="end">1K</div>
                        </ion-range>
                      </div>

                      <!-- Noise Level -->
                      <div class="range-control">
                        <ion-label class="range-label">Noise Level: {{ formatNumber(simulationConfig.noise, 3) }}</ion-label>
                        <ion-range
                          min="0"
                          max="0.05"
                          step="0.001"
                          [(ngModel)]="simulationConfig.noise"
                          color="danger"
                          class="config-range">
                          <div slot="start">0</div>
                          <div slot="end">5%</div>
                        </ion-range>
                      </div>

                      <!-- Advanced Options -->
                      <div class="advanced-options" [class.expanded]="showAdvancedOptions">
                        <ion-button 
                          fill="clear" 
                          size="small"
                          (click)="showAdvancedOptions = !showAdvancedOptions"
                          class="advanced-toggle">
                          <ion-icon [name]="showAdvancedOptions ? 'contract-outline' : 'expand-outline'" slot="start"></ion-icon>
                          Advanced Options
                        </ion-button>

                        <div class="advanced-content" *ngIf="showAdvancedOptions">
                          <ion-item>
                            <ion-checkbox 
                              [(ngModel)]="simulationConfig.parameters['enable_error_correction']"
                              color="primary">
                            </ion-checkbox>
                            <ion-label class="checkbox-label">Error Correction</ion-label>
                          </ion-item>

                          <ion-item>
                            <ion-checkbox 
                              [(ngModel)]="simulationConfig.parameters['quantum_annealing']"
                              color="secondary">
                            </ion-checkbox>
                            <ion-label class="checkbox-label">Quantum Annealing</ion-label>
                          </ion-item>

                          <ion-item>
                            <ion-checkbox 
                              [(ngModel)]="simulationConfig.parameters['consciousness_coupling']"
                              color="tertiary">
                            </ion-checkbox>
                            <ion-label class="checkbox-label">Consciousness Coupling</ion-label>
                          </ion-item>
                        </div>
                      </div>

                    </div>
                  </ion-card-content>
                </ion-card>
              </ion-col>

              <!-- Quantum State Visualization -->
              <ion-col size="12" size-lg="8">
                <ion-card class="visualization-card">
                  <ion-card-header>
                    <ion-card-title>
                      <ion-icon name="radio-outline"></ion-icon>
                      Quantum State Visualization
                      <div class="visualization-controls">
                        <ion-button fill="clear" size="small" (click)="toggleFullscreen()">
                          <ion-icon [name]="isFullscreen ? 'contract-outline' : 'expand-outline'"></ion-icon>
                        </ion-button>
                      </div>
                    </ion-card-title>
                  </ion-card-header>
                  
                  <ion-card-content class="visualization-content">
                    <div class="quantum-display" [class.fullscreen]="isFullscreen">
                      
                      <!-- State Vector Display -->
                      <div class="state-vector-container" #stateVectorContainer>
                        <canvas 
                          #quantumCanvas
                          class="quantum-canvas"
                          [class.animating]="isSimulating">
                        </canvas>
                        
                        <!-- Overlay Information -->
                        <div class="canvas-overlay" *ngIf="showOverlay">
                          <div class="overlay-stats">
                            <div class="stat-group">
                              <span class="stat-title">System State</span>
                              <div class="stat-value">{{ currentSystem?.qubits }} qubits</div>
                              <div class="stat-value">{{ getTotalStates() }} states</div>
                            </div>
                            <div class="stat-group">
                              <span class="stat-title">Entanglement</span>
                              <div class="stat-value">{{ formatNumber(getAverageEntanglement(), 3) }}</div>
                            </div>
                            <div class="stat-group">
                              <span class="stat-title">Coherence</span>
                              <div class="stat-value">{{ formatNumber(getAverageCoherence(), 3) }}</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <!-- Measurement Panel -->
                      <div class="measurement-panel" *ngIf="currentSystem">
                        <h4>Qubit Measurements</h4>
                        <div class="qubit-grid">
                          <div 
                            *ngFor="let qubit of getQubitArray(); let i = index"
                            class="qubit-item"
                            [class.measured]="isQubitMeasured(i)">
                            <div class="qubit-label">Q{{ i }}</div>
                            <ion-button 
                              size="small" 
                              fill="outline"
                              [disabled]="isSimulating || isQubitMeasured(i)"
                              (click)="measureQubit(i)"
                              class="measure-button">
                              Measure
                            </ion-button>
                            <div class="qubit-result" *ngIf="getQubitMeasurement(i) as measurement">
                              {{ measurement.result }} ({{ formatNumber(measurement.probability, 3) }})
                            </div>
                          </div>
                        </div>
                      </div>

                    </div>
                  </ion-card-content>
                </ion-card>
              </ion-col>

            </ion-row>
          </ion-grid>
        </div>

        <!-- Visualization Tab -->
        <div class="tab-pane" [class.active]="selectedTab === 'visualization'">
          <div class="visualization-pane">
            <ion-card>
              <ion-card-header>
                <ion-card-title>Quantum Circuit Visualizer</ion-card-title>
              </ion-card-header>
              <ion-card-content>
                <p>Advanced quantum circuit visualization coming soon...</p>
                <ion-button expand="block" fill="outline" disabled>
                  <ion-icon name="eye-outline" slot="start"></ion-icon>
                  Circuit Visualizer (In Development)
                </ion-button>
              </ion-card-content>
            </ion-card>
          </div>
        </div>

        <!-- Metrics Tab -->
        <div class="tab-pane" [class.active]="selectedTab === 'metrics'">
          <div class="metrics-pane">
            <ion-card>
              <ion-card-header>
                <ion-card-title>Quantum Consciousness Metrics</ion-card-title>
              </ion-card-header>
              <ion-card-content>
                <div *ngIf="metrics; else noMetrics">
                  <ion-grid>
                    <ion-row>
                      <ion-col size="6">
                        <ion-item>
                          <ion-label>Quantum Coherence</ion-label>
                          <ion-badge color="primary">{{ formatNumber(metrics.quantumCoherence * 100, 1) }}%</ion-badge>
                        </ion-item>
                      </ion-col>
                      <ion-col size="6">
                        <ion-item>
                          <ion-label>Entanglement Entropy</ion-label>
                          <ion-badge color="secondary">{{ formatNumber(metrics.entanglementEntropy, 2) }}</ion-badge>
                        </ion-item>
                      </ion-col>
                    </ion-row>
                  </ion-grid>
                </div>
                <ng-template #noMetrics>
                  <p>Run a quantum simulation to see metrics...</p>
                </ng-template>
              </ion-card-content>
            </ion-card>
          </div>
        </div>

        <!-- Circuit Tab -->
        <div class="tab-pane" [class.active]="selectedTab === 'circuit'">
          <div class="circuit-pane">
            <ion-card>
              <ion-card-header>
                <ion-card-title>Quantum Circuit Designer</ion-card-title>
              </ion-card-header>
              <ion-card-content>
                <p>Interactive quantum circuit designer coming soon...</p>
                <ion-button expand="block" fill="outline" disabled>
                  <ion-icon name="grid-outline" slot="start"></ion-icon>
                  Circuit Designer (In Development)
                </ion-button>
              </ion-card-content>
            </ion-card>
          </div>
        </div>

      </div>

      <!-- Simulation Log -->
      <ion-card class="log-card" [class.collapsed]="isLogCollapsed">
        <ion-card-header class="log-header" (click)="toggleLog()">
          <ion-card-title>
            <ion-icon name="bar-chart-outline"></ion-icon>
            Simulation Log
            <ion-badge color="primary" class="log-count">{{ simulationLogs.length }}</ion-badge>
            <ion-button fill="clear" size="small" class="log-toggle">
              <ion-icon [name]="isLogCollapsed ? 'expand-outline' : 'contract-outline'"></ion-icon>
            </ion-button>
          </ion-card-title>
        </ion-card-header>
        
        <ion-card-content class="log-content" *ngIf="!isLogCollapsed">
          <div class="log-container">
            <div 
              *ngFor="let log of simulationLogs; trackBy: trackByLog; let i = index"
              class="log-entry"
              [class]="getLogClass(log)">
              {{ log }}
            </div>
            
            <div class="log-empty" *ngIf="simulationLogs.length === 0">
              <ion-icon name="pulse-outline"></ion-icon>
              <p>No simulation logs yet. Start a simulation to see activity.</p>
            </div>
          </div>
        </ion-card-content>
      </ion-card>

      <!-- FAB for Export -->
      <ion-fab vertical="bottom" horizontal="end" class="export-fab">
        <ion-fab-button color="tertiary" (click)="showExportOptions()">
          <ion-icon name="download-outline"></ion-icon>
        </ion-fab-button>
      </ion-fab>

    </ion-content>
  `,
  styleUrls: ['./quantum.page.scss'],
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
    IonGrid,
    IonRow,
    IonCol,
    IonButton,
    IonIcon,
    IonBadge,
    IonItem,
    IonLabel,
    IonSelect,
    IonSelectOption,
    IonRange,
    IonCheckbox,
    IonProgressBar,
    IonText,
    IonFab,
    IonFabButton,
    IonSegment,
    IonSegmentButton,
    IonList,
    IonItemSliding,
    IonItemOptions,
    IonItemOption
  ]
})
export class QuantumPage implements OnInit, OnDestroy {
  @ViewChild('quantumCanvas', { static: false }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('stateVectorContainer', { static: false }) containerRef!: ElementRef<HTMLDivElement>;

  selectedTab: string = 'simulation';
  currentSystem: QuantumSystem | null = null;
  isSimulating: boolean = false;
  metrics: QuantumConsciousnessMetrics | null = null;
  simulationLogs: string[] = [];
  
  simulationConfig: QuantumSimulationConfig = {
    qubits: 4,
    iterations: 1000,
    decoherence: 0.001,
    temperature: 0.01,
    noise: 0.001,
    algorithm: 'consciousness',
    parameters: {
      enable_error_correction: false,
      quantum_annealing: false,
      consciousness_coupling: true
    }
  };

  showAdvancedOptions: boolean = false;
  isFullscreen: boolean = false;
  showOverlay: boolean = true;
  isLogCollapsed: boolean = true;

  private subscriptions: Subscription[] = [];

  constructor(
    private quantumService: QuantumService,
    private alertController: AlertController,
    private toastController: ToastController
  ) {
    this.initializeIcons();
  }

  ngOnInit() {
    this.subscribeToQuantumService();
    this.initializeCanvas();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  private initializeIcons() {
    addIcons({
      'at-outline': atOutline,
      'play-outline': playOutline,
      'pause-outline': pauseOutline,
      'stop-outline': stopOutline,
      'refresh-outline': refreshOutline,
      'settings-outline': settingsOutline,
      'analytics-outline': analyticsOutline,
      'download-outline': downloadOutline,
      'expand-outline': expandOutline,
      'contract-outline': contractOutline,
      'eye-outline': eyeOutline,
      'flash-outline': flashOutline,
      'pulse-outline': pulseOutline,
      'infinite-outline': infiniteOutline,
      'layers-outline': layersOutline,
      'grid-outline': gridOutline,
      'bar-chart-outline': barChartOutline,
      'trending-up-outline': trendingUpOutline,
      'speedometer-outline': speedometerOutline,
      'nuclear-outline': nuclearOutline,
      'radio-outline': radioOutline
    });
  }

  private subscribeToQuantumService() {
    this.subscriptions.push(
      this.quantumService.currentSystem$.subscribe(system => {
        this.currentSystem = system;
        this.updateCanvasVisualization();
      }),

      this.quantumService.isSimulating$.subscribe(simulating => {
        this.isSimulating = simulating;
      }),

      this.quantumService.metrics$.subscribe(metrics => {
        this.metrics = metrics;
      }),

      this.quantumService.simulationLog$.subscribe(logs => {
        this.simulationLogs = logs;
      }),

      this.quantumService.quantumStateStream$.subscribe(states => {
        if (states) {
          this.updateStateVisualization(states);
        }
      })
    );
  }

  private initializeCanvas() {
    // Canvas initialization will be handled after view init
    setTimeout(() => {
      if (this.canvasRef) {
        this.setupCanvas();
        this.updateCanvasVisualization();
      }
    }, 100);
  }

  private setupCanvas() {
    const canvas = this.canvasRef.nativeElement;
    const container = this.containerRef.nativeElement;
    
    // Set canvas size to container size
    const resizeCanvas = () => {
      canvas.width = container.clientWidth;
      canvas.height = Math.max(400, container.clientHeight);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
  }

  private updateCanvasVisualization() {
    if (!this.canvasRef || !this.currentSystem) return;

    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw quantum state visualization
    this.drawQuantumStates(ctx, canvas.width, canvas.height);
  }

  private drawQuantumStates(ctx: CanvasRenderingContext2D, width: number, height: number) {
    if (!this.currentSystem) return;

    const states = this.currentSystem.states;
    const numStates = states.length;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.3;

    // Draw state circle
    ctx.strokeStyle = '#3880ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw individual states
    states.forEach((state, index) => {
      const angle = (2 * Math.PI * index) / numStates;
      const x = centerX + radius * Math.cos(angle) * state.amplitude;
      const y = centerY + radius * Math.sin(angle) * state.amplitude;

      // State point
      ctx.fillStyle = `hsl(${(angle * 180 / Math.PI) % 360}, 70%, ${50 + state.probability * 50}%)`;
      ctx.beginPath();
      ctx.arc(x, y, 3 + state.probability * 5, 0, 2 * Math.PI);
      ctx.fill();

      // Phase line
      if (state.superposition) {
        const phaseX = x + 20 * Math.cos(state.phase);
        const phaseY = y + 20 * Math.sin(state.phase);
        
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(phaseX, phaseY);
        ctx.stroke();
      }

      // State label
      if (state.probability > 0.01) {
        ctx.fillStyle = '#000';
        ctx.font = '10px monospace';
        ctx.fillText(`|${index.toString(2).padStart(this.currentSystem!.qubits, '0')}⟩`, x + 10, y - 10);
      }
    });

    // Draw entanglement connections
    this.drawEntanglementConnections(ctx, centerX, centerY, radius);
  }

  private drawEntanglementConnections(ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number) {
    if (!this.currentSystem) return;

    const states = this.currentSystem.states;
    
    // Draw connections between entangled states
    for (let i = 0; i < states.length; i++) {
      for (let j = i + 1; j < states.length; j++) {
        const entanglement = Math.min(states[i].entanglement, states[j].entanglement);
        
        if (entanglement > 0.1) {
          const angle1 = (2 * Math.PI * i) / states.length;
          const angle2 = (2 * Math.PI * j) / states.length;
          
          const x1 = centerX + radius * Math.cos(angle1) * states[i].amplitude;
          const y1 = centerY + radius * Math.sin(angle1) * states[i].amplitude;
          const x2 = centerX + radius * Math.cos(angle2) * states[j].amplitude;
          const y2 = centerY + radius * Math.sin(angle2) * states[j].amplitude;

          ctx.strokeStyle = `rgba(255, 100, 100, ${entanglement})`;
          ctx.lineWidth = entanglement * 3;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }
  }

  private updateStateVisualization(states: any) {
    // Update real-time state visualization
    this.updateCanvasVisualization();
  }

  async toggleSimulation() {
    if (this.isSimulating) {
      this.quantumService.stopSimulation();
    } else {
      await this.startSimulation();
    }
  }

  async startSimulation() {
    try {
      await this.quantumService.startSimulation(this.simulationConfig);
      
      const toast = await this.toastController.create({
        message: 'Quantum simulation started successfully',
        duration: 2000,
        color: 'success',
        position: 'top'
      });
      await toast.present();
      
    } catch (error) {
      const toast = await this.toastController.create({
        message: `Failed to start simulation: ${error}`,
        duration: 3000,
        color: 'danger',
        position: 'top'
      });
      await toast.present();
    }
  }

  stopSimulation() {
    this.quantumService.stopSimulation();
  }

  async resetSystem() {
    const alert = await this.alertController.create({
      header: 'Reset Quantum System',
      message: 'This will reset all quantum states and clear the simulation log. Continue?',
      buttons: [
        {
          text: 'Cancel',
          role: 'cancel'
        },
        {
          text: 'Reset',
          role: 'destructive',
          handler: () => {
            this.quantumService.resetSystem();
          }
        }
      ]
    });

    await alert.present();
  }

  measureQubit(qubit: number) {
    try {
      const measurement = this.quantumService.measureQubit(qubit);
      
      // Show measurement result
      this.toastController.create({
        message: `Qubit ${qubit} measured: ${measurement.result} (probability: ${this.formatNumber(measurement.probability, 3)})`,
        duration: 2000,
        color: 'primary',
        position: 'top'
      }).then(toast => toast.present());
      
    } catch (error) {
      this.toastController.create({
        message: `Measurement failed: ${error}`,
        duration: 3000,
        color: 'danger',
        position: 'top'
      }).then(toast => toast.present());
    }
  }

  onTabChange(event: any) {
    this.selectedTab = event.detail.value;
  }

  onCircuitChanged(circuit: any) {
    // Handle circuit changes from designer
    console.log('Circuit changed:', circuit);
  }

  toggleFullscreen() {
    this.isFullscreen = !this.isFullscreen;
  }

  toggleLog() {
    this.isLogCollapsed = !this.isLogCollapsed;
  }

  async showExportOptions() {
    const alert = await this.alertController.create({
      header: 'Export Quantum System',
      message: 'Choose export format:',
      buttons: [
        {
          text: 'JSON Data',
          handler: () => this.exportAsJSON()
        },
        {
          text: 'CSV Metrics',
          handler: () => this.exportAsCSV()
        },
        {
          text: 'PNG Image',
          handler: () => this.exportAsPNG()
        },
        {
          text: 'Cancel',
          role: 'cancel'
        }
      ]
    });

    await alert.present();
  }

  exportAsJSON() {
    const data = this.quantumService.exportSystemState();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    this.downloadFile(blob, `quantum-system-${Date.now()}.json`);
  }

  exportAsCSV() {
    // Export metrics as CSV
    if (!this.metrics) return;

    const csv = Object.entries(this.metrics)
      .map(([key, value]) => `${key},${value}`)
      .join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    this.downloadFile(blob, `quantum-metrics-${Date.now()}.csv`);
  }

  exportAsPNG() {
    if (!this.canvasRef) return;
    
    const canvas = this.canvasRef.nativeElement;
    canvas.toBlob(blob => {
      if (blob) {
        this.downloadFile(blob, `quantum-visualization-${Date.now()}.png`);
      }
    });
  }

  private downloadFile(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Helper methods
  formatNumber(value: number, decimals: number = 2): string {
    return value.toFixed(decimals);
  }

  getStatusColor(status?: string): string {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'error': return 'danger';
      default: return 'medium';
    }
  }

  getStatusLabel(status?: string): string {
    switch (status) {
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      case 'idle': return 'Ready';
      default: return 'Unknown';
    }
  }

  getTotalStates(): number {
    return this.currentSystem ? Math.pow(2, this.currentSystem.qubits) : 0;
  }

  getAverageEntanglement(): number {
    if (!this.currentSystem) return 0;
    const total = this.currentSystem.states.reduce((sum, state) => sum + state.entanglement, 0);
    return total / this.currentSystem.states.length;
  }

  getAverageCoherence(): number {
    if (!this.currentSystem) return 0;
    const total = this.currentSystem.states.reduce((sum, state) => sum + state.coherence, 0);
    return total / this.currentSystem.states.length;
  }

  getQubitArray(): number[] {
    return this.currentSystem ? Array.from({length: this.currentSystem.qubits}, (_, i) => i) : [];
  }

  isQubitMeasured(qubit: number): boolean {
    return this.currentSystem?.measurements.some(m => m.qubit === qubit) || false;
  }

  getQubitMeasurement(qubit: number): QuantumMeasurement | undefined {
    return this.currentSystem?.measurements.find(m => m.qubit === qubit);
  }

  getLogClass(log: string): string {
    if (log.includes('ERROR')) return 'error';
    if (log.includes('WARNING')) return 'warning';
    if (log.includes('SUCCESS')) return 'success';
    return 'info';
  }

  trackByLog(index: number, log: string): string {
    return log;
  }
}
