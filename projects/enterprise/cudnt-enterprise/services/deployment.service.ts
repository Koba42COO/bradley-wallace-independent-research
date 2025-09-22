import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, Subject } from 'rxjs';

// Deployment Interfaces
export interface DeploymentEnvironment {
  name: string;
  type: 'development' | 'staging' | 'production';
  status: 'healthy' | 'degraded' | 'down';
  version: string;
  instances: number;
  resources: {
    cpu: number;
    memory: number;
    storage: number;
  };
  endpoints: {
    api: string;
    frontend: string;
    monitoring: string;
  };
}

export interface DeploymentPipeline {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'failed';
  stages: {
    name: string;
    status: string;
    duration: number;
    logs: string[];
  }[];
  triggeredBy: string;
  startTime: number;
  endTime?: number;
}

export interface InfrastructureMetrics {
  kubernetes: {
    nodes: number;
    pods: number;
    services: number;
    healthyNodes: number;
  };
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkTraffic: number;
  };
  performance: {
    averageResponseTime: number;
    requestsPerSecond: number;
    errorRate: number;
    uptime: number;
  };
}

@Injectable({
  providedIn: 'root'
})
export class DeploymentService {
  private readonly apiUrl = 'http://localhost:3000/api';

  // State management
  private environmentsSubject = new BehaviorSubject<DeploymentEnvironment[]>([]);
  private pipelinesSubject = new BehaviorSubject<DeploymentPipeline[]>([]);
  private infrastructureSubject = new BehaviorSubject<InfrastructureMetrics | null>(null);

  public environments$ = this.environmentsSubject.asObservable();
  public pipelines$ = this.pipelinesSubject.asObservable();
  public infrastructure$ = this.infrastructureSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadEnvironments();
    this.startInfrastructureMonitoring();
  }

  // Load deployment environments
  loadEnvironments(): void {
    this.http.get<DeploymentEnvironment[]>(`${this.apiUrl}/deployment/environments`)
      .subscribe(environments => {
        this.environmentsSubject.next(environments);
      });
  }

  // Trigger deployment pipeline
  triggerDeployment(environment: string, version: string): Observable<DeploymentPipeline> {
    return this.http.post<DeploymentPipeline>(`${this.apiUrl}/deployment/trigger`, {
      environment,
      version,
      consciousnessOptimized: true
    });
  }

  // Get deployment pipelines
  getDeploymentPipelines(): Observable<DeploymentPipeline[]> {
    return this.http.get<DeploymentPipeline[]>(`${this.apiUrl}/deployment/pipelines`);
  }

  // Infrastructure monitoring
  private startInfrastructureMonitoring(): void {
    setInterval(() => {
      this.http.get<InfrastructureMetrics>(`${this.apiUrl}/deployment/infrastructure-metrics`)
        .subscribe(metrics => {
          this.infrastructureSubject.next(metrics);
        });
    }, 10000); // Update every 10 seconds
  }

  // Auto-scaling configuration
  configureAutoScaling(environment: string, config: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/deployment/autoscaling/${environment}`, config);
  }

  // Rollback deployment
  rollbackDeployment(environment: string, version: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/deployment/rollback`, {
      environment,
      version
    });
  }

  // Health checks
  performHealthCheck(environment: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/deployment/health/${environment}`);
  }
}
