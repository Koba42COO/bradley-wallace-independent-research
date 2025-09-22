import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, Subject } from 'rxjs';
import { map, catchError, retry, timeout } from 'rxjs/operators';
import { io, Socket } from 'socket.io-client';

// Consciousness Mathematics Constants
const PHI = (1 + Math.sqrt(5)) / 2; // Golden ratio: 1.618034
const CONSCIOUSNESS_RATIO = 79/21;

// Interfaces
export interface OptimizationResult {
  optimizationId: string;
  result: {
    optimizedMatrix: number[][];
    performance: {
      processingTime: number;
      speedupFactor: number;
      complexityReduction: string;
      consciousnessLevel: number;
      improvementPercent: number;
    };
    metadata: {
      algorithm: string;
      phi: number;
      consciousnessRatio: number;
      timestamp: number;
    };
  };
  success: boolean;
  message: string;
}

export interface DashboardStats {
  stats: {
    totalOptimizations: number;
    avgSpeedup: number;
    avgProcessingTime: number;
    avgConsciousnessLevel: number;
    avgImprovement: number;
    totalProcessingTime: number;
  };
  trends: Array<{
    performance: { improvementPercent: number };
    metadata: { timestamp: string };
  }>;
  consciousness: {
    currentLevel: number;
    phi: number;
    enhancement: string;
  };
  system: {
    status: string;
    complexityReduction: string;
    architecture: string;
  };
}

export interface SystemStatus {
  systemStatus: string;
  activeOptimizations: number;
  performance: {
    avgSpeedupFactor: number;
    complexityReduction: string;
    consciousnessLevel: number;
    kLoopProduction: string;
  };
  infrastructure: {
    pdvm: string;
    qvm: string;
    consciousnessMath: string;
  };
  timestamp: string;
}

@Injectable({
  providedIn: 'root'
})
export class CUDNTService {
  private readonly apiUrl = 'http://localhost:3000/api';
  private socket: Socket;

  // State management
  private systemStatusSubject = new BehaviorSubject<SystemStatus | null>(null);
  private isProcessingSubject = new BehaviorSubject<boolean>(false);

  // Public observables
  public systemStatus$ = this.systemStatusSubject.asObservable();
  public isProcessing$ = this.isProcessingSubject.asObservable();

  // Real-time updates
  private realtimeUpdates = new Subject<any>();
  public realtimeUpdates$ = this.realtimeUpdates.asObservable();

  constructor(private http: HttpClient) {
    this.initializeWebSocket();
    this.loadSystemStatus();
  }

  private initializeWebSocket() {
    this.socket = io('http://localhost:3000');

    this.socket.on('connect', () => {
      console.log('Connected to CUDNT real-time updates');
    });

    this.socket.on('status_update', (data) => {
      this.realtimeUpdates.next(data);
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from CUDNT server');
    });
  }

  // System health check
  checkHealth(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`).pipe(
      timeout(5000),
      retry(2),
      catchError(this.handleError)
    );
  }

  // Load system status
  loadSystemStatus(): void {
    this.http.get<SystemStatus>(`${this.apiUrl}/status/realtime`).pipe(
      timeout(5000),
      catchError(this.handleError)
    ).subscribe(status => {
      this.systemStatusSubject.next(status);
    });
  }

  // Matrix optimization
  optimizeMatrix(matrix: number[][], target?: number[][], userId?: string): Observable<OptimizationResult> {
    this.isProcessingSubject.next(true);

    const payload = {
      matrix,
      target,
      userId: userId || 'anonymous'
    };

    return this.http.post<OptimizationResult>(`${this.apiUrl}/optimize/matrix`, payload).pipe(
      timeout(30000), // 30 second timeout for complex optimizations
      map(result => {
        this.isProcessingSubject.next(false);
        return result;
      }),
      catchError(error => {
        this.isProcessingSubject.next(false);
        return this.handleError(error);
      })
    );
  }

  // Get user dashboard data
  getDashboardData(userId: string): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(`${this.apiUrl}/dashboard/${userId}`).pipe(
      timeout(10000),
      retry(1),
      catchError(this.handleError)
    );
  }

  // Get user optimizations
  getUserOptimizations(userId: string, limit = 10, skip = 0): Observable<any> {
    return this.http.get(`${this.apiUrl}/optimizations/${userId}?limit=${limit}&skip=${skip}`).pipe(
      timeout(10000),
      retry(1),
      catchError(this.handleError)
    );
  }

  // Calculate theoretical performance
  calculateTheoreticalPerformance(matrixSize: number): any {
    const classicalComplexity = Math.pow(matrixSize, 2);
    const cudntComplexity = Math.pow(matrixSize, 1.44);
    const theoreticalSpeedup = classicalComplexity / cudntComplexity;

    return {
      matrixSize,
      classicalComplexity: classicalComplexity.toLocaleString(),
      cudntComplexity: cudntComplexity.toFixed(0),
      theoreticalSpeedup: theoreticalSpeedup.toFixed(1) + 'x',
      phi: PHI,
      consciousnessRatio: CONSCIOUSNESS_RATIO
    };
  }

  // Generate test matrix
  generateTestMatrix(size: number): number[][] {
    const matrix: number[][] = [];
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        matrix[i][j] = Math.random() * 100;
      }
    }
    return matrix;
  }

  private handleError(error: any): Observable<never> {
    console.error('CUDNT Service Error:', error);
    throw error;
  }
}
