import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject, interval } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';

// Analytics Interfaces
export interface PerformanceMetrics {
  timestamp: number;
  userId: string;
  optimizationId: string;
  metrics: {
    inputSize: number;
    processingTime: number;
    speedupFactor: number;
    consciousnessLevel: number;
    memoryUsage: number;
    cpuUtilization: number;
    energyEfficiency: number;
  };
  system: {
    architecture: string;
    coreCount: number;
    availableMemory: number;
    platform: string;
  };
}

export interface BusinessAnalytics {
  revenue: {
    monthly: number;
    quarterly: number;
    annual: number;
    growth: number;
  };
  customers: {
    total: number;
    active: number;
    churned: number;
    newThisMonth: number;
  };
  usage: {
    totalOptimizations: number;
    avgOptimizationsPerUser: number;
    peakConcurrentUsers: number;
    systemUtilization: number;
  };
  performance: {
    avgSpeedupFactor: number;
    avgConsciousnessLevel: number;
    systemReliability: number;
    customerSatisfaction: number;
  };
}

export interface PredictiveInsights {
  churnRisk: {
    userId: string;
    riskScore: number;
    reasons: string[];
    recommendations: string[];
  }[];
  growthOpportunities: {
    segment: string;
    potential: number;
    confidence: number;
    timeframe: string;
  }[];
  systemOptimizations: {
    component: string;
    currentPerformance: number;
    optimizedPerformance: number;
    implementationCost: number;
  }[];
}

@Injectable({
  providedIn: 'root'
})
export class AnalyticsService {
  private readonly apiUrl = 'http://localhost:3000/api';

  // Real-time analytics streams
  private metricsSubject = new BehaviorSubject<PerformanceMetrics[]>([]);
  private businessAnalyticsSubject = new BehaviorSubject<BusinessAnalytics | null>(null);
  private predictiveInsightsSubject = new BehaviorSubject<PredictiveInsights | null>(null);

  public metrics$ = this.metricsSubject.asObservable();
  public businessAnalytics$ = this.businessAnalyticsSubject.asObservable();
  public predictiveInsights$ = this.predictiveInsightsSubject.asObservable();

  constructor(private http: HttpClient) {
    this.startRealTimeAnalytics();
  }

  private startRealTimeAnalytics() {
    // Update metrics every 30 seconds
    interval(30000).pipe(
      switchMap(() => this.getRealtimeMetrics())
    ).subscribe(metrics => {
      this.metricsSubject.next(metrics);
    });

    // Update business analytics every 5 minutes
    interval(300000).pipe(
      switchMap(() => this.getBusinessAnalytics())
    ).subscribe(analytics => {
      this.businessAnalyticsSubject.next(analytics);
    });

    // Update predictive insights every hour
    interval(3600000).pipe(
      switchMap(() => this.getPredictiveInsights())
    ).subscribe(insights => {
      this.predictiveInsightsSubject.next(insights);
    });
  }

  // Real-time performance metrics
  getRealtimeMetrics(): Observable<PerformanceMetrics[]> {
    return this.http.get<PerformanceMetrics[]>(`${this.apiUrl}/analytics/realtime-metrics`);
  }

  // Business analytics dashboard
  getBusinessAnalytics(): Observable<BusinessAnalytics> {
    return this.http.get<BusinessAnalytics>(`${this.apiUrl}/analytics/business`);
  }

  // AI-powered predictive insights
  getPredictiveInsights(): Observable<PredictiveInsights> {
    return this.http.get<PredictiveInsights>(`${this.apiUrl}/analytics/predictive`);
  }

  // Advanced performance analysis
  getPerformanceAnalysis(timeRange: string = '7d'): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/performance-analysis?range=${timeRange}`);
  }

  // Consciousness mathematics correlation analysis
  getConsciousnessCorrelations(): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/consciousness-correlations`);
  }

  // Customer journey analytics
  getCustomerJourney(userId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/customer-journey/${userId}`);
  }

  // ROI and cost optimization analysis
  getROIAnalysis(): Observable<any> {
    return this.http.get(`${this.apiUrl}/analytics/roi-analysis`);
  }

  // Export analytics data
  exportAnalytics(format: 'csv' | 'json' | 'pdf', timeRange: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/analytics/export?format=${format}&range=${timeRange}`, {
      responseType: 'blob'
    });
  }
}
