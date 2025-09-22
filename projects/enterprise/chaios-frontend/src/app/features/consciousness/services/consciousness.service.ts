import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval, map, switchMap, catchError, of } from 'rxjs';

import { ApiService } from '../../../core/api.service';
import { WebSocketService } from '../../../core/websocket.service';
import { NotificationService } from '../../../core/notification.service';

export interface ConsciousnessMetrics {
  performanceGain: number;
  correlation: number;
  processingTime: number;
  iterations: number;
  phi: number;
  sigma: number;
  status: 'idle' | 'processing' | 'completed' | 'error';
  timestamp: Date;
}

export interface WallaceTransformData {
  input: number[];
  output: number[];
  transformation_matrix: number[][];
  eigenvalues: number[];
  performance_metrics: {
    gain: number;
    correlation: number;
    processing_time: number;
  };
}

export interface ConsciousnessVisualization {
  spiralPoints: { x: number; y: number; phi: number }[];
  harmonicFrequencies: number[];
  correlationMatrix: number[][];
  phiRatioEvolution: number[];
  energyLevels: number[];
}

@Injectable({
  providedIn: 'root'
})
export class ConsciousnessService {
  private metricsSubject = new BehaviorSubject<ConsciousnessMetrics>({
    performanceGain: 0,
    correlation: 0,
    processingTime: 0,
    iterations: 0,
    phi: 1.618034,
    sigma: 0.381966,
    status: 'idle',
    timestamp: new Date()
  });

  public metrics$ = this.metricsSubject.asObservable();

  private processingSubject = new BehaviorSubject<boolean>(false);
  public isProcessing$ = this.processingSubject.asObservable();

  private visualizationSubject = new BehaviorSubject<ConsciousnessVisualization | null>(null);
  public visualization$ = this.visualizationSubject.asObservable();

  // Real-time metrics update interval
  private metricsInterval$ = interval(2000).pipe(
    switchMap(() => this.updateRealTimeMetrics()),
    catchError(error => {
      console.warn('Failed to update real-time metrics:', error);
      return of(null);
    })
  );

  constructor(
    private apiService: ApiService,
    private webSocketService: WebSocketService,
    private notificationService: NotificationService
  ) {
    this.setupWebSocketSubscriptions();
    console.log('🧠 ConsciousnessService initialized');
  }

  /**
   * Process consciousness data using Wallace Transform
   */
  async processConsciousness(
    algorithm: string = 'wallace_transform',
    iterations: number = 1000,
    inputData?: number[]
  ): Promise<ConsciousnessResponse> {
    
    this.processingSubject.next(true);
    this.updateMetrics({ status: 'processing' });

    const request: ConsciousnessRequest = {
      algorithm: algorithm,
      parameters: {
        iterations: iterations,
        phi: this.calculatePhi(),
        sigma: this.calculateSigma(),
        optimization_level: 'maximum',
        consciousness_enhancement: true,
        harmonic_alignment: true
      },
      input_data: inputData || this.generateRandomData(100)
    };

    try {
      console.log('🧠 Processing consciousness data:', request);
      
      // Start WebSocket session for real-time updates
      this.webSocketService.startConsciousnessSession(request.parameters);

      const response = await this.apiService.processConsciousness(request).toPromise();
      
      if (response) {
        this.updateMetrics({
          performanceGain: response.performance_gain,
          correlation: response.correlation,
          processingTime: response.processing_time,
          iterations: iterations,
          status: 'completed',
          timestamp: new Date()
        });

        // Generate visualization data
        this.generateVisualization(response);

        // Show notification
        this.notificationService.notifyConsciousnessUpdate(
          response.performance_gain,
          response.correlation,
          response.processing_time
        );

        console.log('✅ Consciousness processing completed:', response);
      }

      return response!;

    } catch (error: any) {
      console.error('❌ Consciousness processing failed:', error);
      
      this.updateMetrics({ 
        status: 'error',
        timestamp: new Date()
      });

      this.notificationService.showNotification(
        `Consciousness processing failed: ${error.message}`,
        'error'
      );

      throw error;

    } finally {
      this.processingSubject.next(false);
    }
  }

  /**
   * Get consciousness processing history
   */
  async getProcessingHistory(): Promise<any[]> {
    try {
      return await this.apiService.getConsciousnessHistory().toPromise() || [];
    } catch (error) {
      console.warn('Failed to load consciousness history:', error);
      return [];
    }
  }

  /**
   * Start real-time metrics monitoring
   */
  startRealTimeMonitoring(): Observable<ConsciousnessMetrics | null> {
    console.log('🔄 Starting real-time consciousness monitoring');
    return this.metricsInterval$;
  }

  /**
   * Stop real-time monitoring
   */
  stopRealTimeMonitoring(): void {
    console.log('⏹️ Stopping real-time consciousness monitoring');
    this.webSocketService.stopSession('consciousness');
  }

  /**
   * Generate golden ratio spiral data for visualization
   */
  generateGoldenSpiral(points: number = 1000): { x: number; y: number; phi: number }[] {
    const spiral = [];
    const phi = this.calculatePhi();
    
    for (let i = 0; i < points; i++) {
      const angle = i * (Math.PI / 180) * 2;
      const radius = Math.pow(phi, angle / (Math.PI / 2));
      
      spiral.push({
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        phi: radius
      });
    }
    
    return spiral;
  }

  /**
   * Calculate harmonic frequencies based on consciousness data
   */
  calculateHarmonicFrequencies(data: number[]): number[] {
    const phi = this.calculatePhi();
    const fundamentalFreq = 432; // Hz - consciousness frequency
    
    return [
      fundamentalFreq,
      fundamentalFreq * phi,
      fundamentalFreq * phi * phi,
      fundamentalFreq / phi,
      fundamentalFreq * Math.PI,
      fundamentalFreq * Math.E
    ];
  }

  /**
   * Generate consciousness correlation matrix
   */
  generateCorrelationMatrix(size: number = 10): number[][] {
    const matrix: number[][] = [];
    const phi = this.calculatePhi();
    
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
        } else {
          // Generate correlation based on phi ratio and distance
          const distance = Math.abs(i - j);
          const correlation = Math.pow(phi, -distance) * Math.random() * 0.5 + 0.5;
          matrix[i][j] = Math.min(correlation, 1.0);
        }
      }
    }
    
    return matrix;
  }

  /**
   * Simulate phi ratio evolution during processing
   */
  simulatePhiEvolution(steps: number = 100): number[] {
    const evolution = [];
    const targetPhi = this.calculatePhi();
    
    for (let i = 0; i < steps; i++) {
      const progress = i / steps;
      const noise = (Math.random() - 0.5) * 0.01;
      const value = 1.0 + progress * (targetPhi - 1.0) + noise;
      evolution.push(Math.max(1.0, Math.min(2.0, value)));
    }
    
    return evolution;
  }

  /**
   * Get current consciousness metrics
   */
  getCurrentMetrics(): ConsciousnessMetrics {
    return this.metricsSubject.value;
  }

  /**
   * Reset consciousness metrics
   */
  resetMetrics(): void {
    this.metricsSubject.next({
      performanceGain: 0,
      correlation: 0,
      processingTime: 0,
      iterations: 0,
      phi: this.calculatePhi(),
      sigma: this.calculateSigma(),
      status: 'idle',
      timestamp: new Date()
    });
    
    this.visualizationSubject.next(null);
  }

  // Private helper methods

  private setupWebSocketSubscriptions(): void {
    this.webSocketService.consciousnessUpdates$.subscribe(update => {
      if (update.type === 'consciousness_update' && update.data) {
        this.updateMetrics({
          performanceGain: update.data.performance_gain,
          correlation: update.data.correlation,
          processingTime: update.data.processing_time,
          status: update.data.status as any,
          timestamp: new Date()
        });
      }
    });
  }

  private updateMetrics(updates: Partial<ConsciousnessMetrics>): void {
    const currentMetrics = this.metricsSubject.value;
    const newMetrics = { ...currentMetrics, ...updates };
    this.metricsSubject.next(newMetrics);
  }

  private async updateRealTimeMetrics(): Promise<ConsciousnessMetrics | null> {
    try {
      const metrics = await this.apiService.getConsciousnessMetrics().toPromise();
      
      if (metrics) {
        this.updateMetrics({
          performanceGain: metrics.performance_gain || 0,
          correlation: metrics.correlation || 0,
          processingTime: metrics.processing_time || 0,
          timestamp: new Date()
        });
        
        return this.metricsSubject.value;
      }
    } catch (error) {
      // Silently handle errors for real-time updates
    }
    
    return null;
  }

  private generateVisualization(response: ConsciousnessResponse): void {
    const visualization: ConsciousnessVisualization = {
      spiralPoints: this.generateGoldenSpiral(500),
      harmonicFrequencies: this.calculateHarmonicFrequencies([response.performance_gain]),
      correlationMatrix: this.generateCorrelationMatrix(8),
      phiRatioEvolution: this.simulatePhiEvolution(100),
      energyLevels: this.generateEnergyLevels(response.performance_gain)
    };
    
    this.visualizationSubject.next(visualization);
  }

  private generateEnergyLevels(performanceGain: number): number[] {
    const levels = [];
    const baseLevel = performanceGain / 100;
    
    for (let i = 0; i < 10; i++) {
      const level = baseLevel * Math.pow(this.calculatePhi(), i / 3) * (0.8 + Math.random() * 0.4);
      levels.push(Math.min(level, 1.0));
    }
    
    return levels;
  }

  private generateRandomData(size: number): number[] {
    return Array.from({ length: size }, () => Math.random() * 2 - 1);
  }

  private calculatePhi(): number {
    return (1 + Math.sqrt(5)) / 2;
  }

  private calculateSigma(): number {
    return 1 / this.calculatePhi() - 1;
  }

  // Demo methods for development

  /**
   * Run demo consciousness processing
   */
  async runDemo(): Promise<void> {
    console.log('🎭 Running consciousness processing demo');
    
    try {
      await this.processConsciousness(
        'wallace_transform_demo',
        500,
        this.generateRandomData(50)
      );
    } catch (error) {
      console.warn('Demo failed, using simulated data');
      
      // Simulate successful processing for demo
      this.updateMetrics({
        performanceGain: 158.7,
        correlation: 0.999992,
        processingTime: 0.247,
        iterations: 500,
        status: 'completed',
        timestamp: new Date()
      });

      this.generateVisualization({
        performance_gain: 158.7,
        correlation: 0.999992,
        processing_time: 0.247,
        status: 'completed'
      });
    }
  }
}

