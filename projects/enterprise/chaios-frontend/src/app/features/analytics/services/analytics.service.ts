import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval, combineLatest, map, startWith } from 'rxjs';
import { ApiService } from '../../../core/api.service';
import { WebSocketService } from '../../../core/websocket.service';

export interface SystemMetrics {
  cpu: {
    usage: number;
    cores: number;
    temperature: number;
    frequency: number;
  };
  memory: {
    used: number;
    total: number;
    available: number;
    percentage: number;
  };
  disk: {
    used: number;
    total: number;
    available: number;
    percentage: number;
    readSpeed: number;
    writeSpeed: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
    latency: number;
    bandwidth: number;
  };
  gpu: {
    usage: number;
    memory: number;
    temperature: number;
    fanSpeed: number;
  };
  timestamp: Date;
}

export interface PerformanceMetrics {
  apiResponseTime: number;
  databaseQueryTime: number;
  consciousnessProcessingTime: number;
  quantumSimulationTime: number;
  mathVisualizationTime: number;
  frontendRenderTime: number;
  totalRequestsPerSecond: number;
  errorRate: number;
  uptime: number;
  throughput: number;
  timestamp: Date;
}

export interface UsageAnalytics {
  totalUsers: number;
  activeUsers: number;
  newUsers: number;
  sessionDuration: number;
  pageViews: number;
  bounceRate: number;
  conversionRate: number;
  retentionRate: number;
  featureUsage: { [feature: string]: number };
  geographicDistribution: { [country: string]: number };
  deviceTypes: { [device: string]: number };
  browserDistribution: { [browser: string]: number };
  timestamp: Date;
}

export interface ConsciousnessAnalytics {
  totalProcessingRequests: number;
  averageProcessingTime: number;
  successRate: number;
  wallaceTransformUsage: number;
  moebiusOptimizationUsage: number;
  quantumConsciousnessRequests: number;
  averagePerformanceGain: number;
  correlationAccuracy: number;
  consciousnessIndexDistribution: number[];
  emergencePatterns: { [pattern: string]: number };
  timestamp: Date;
}

export interface AIToolAnalytics {
  totalToolExecutions: number;
  mostUsedTools: { [tool: string]: number };
  averageExecutionTime: { [tool: string]: number };
  successRates: { [tool: string]: number };
  errorPatterns: { [error: string]: number };
  llmIntegrationStats: {
    chatgpt: number;
    claude: number;
    gemini: number;
    other: number;
  };
  batchExecutionStats: {
    total: number;
    averageSize: number;
    successRate: number;
  };
  timestamp: Date;
}

export interface SecurityMetrics {
  totalRequests: number;
  blockedRequests: number;
  suspiciousActivity: number;
  failedLogins: number;
  bruteForceAttempts: number;
  sqlInjectionAttempts: number;
  xssAttempts: number;
  ddosAttempts: number;
  vulnerabilityScans: number;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  lastSecurityScan: Date;
  certificateExpiry: Date;
  timestamp: Date;
}

export interface BusinessMetrics {
  revenue: number;
  costs: number;
  profit: number;
  customerAcquisitionCost: number;
  lifetimeValue: number;
  churnRate: number;
  monthlyRecurringRevenue: number;
  apiCallsRevenue: number;
  subscriptionRevenue: number;
  growthRate: number;
  marketShare: number;
  competitorAnalysis: { [competitor: string]: number };
  timestamp: Date;
}

export interface AlertConfig {
  id: string;
  name: string;
  metric: string;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
  threshold: number;
  severity: 'info' | 'warning' | 'error' | 'critical';
  enabled: boolean;
  notifications: ('email' | 'sms' | 'slack' | 'webhook')[];
  cooldownPeriod: number; // minutes
}

export interface Alert {
  id: string;
  configId: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  timestamp: Date;
  acknowledged: boolean;
  resolvedAt?: Date;
  metadata: { [key: string]: any };
}

export interface DashboardWidget {
  id: string;
  type: 'metric' | 'chart' | 'table' | 'gauge' | 'heatmap' | 'map' | 'list';
  title: string;
  description: string;
  dataSource: string;
  config: { [key: string]: any };
  position: { x: number; y: number; width: number; height: number };
  refreshInterval: number; // seconds
  visible: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class AnalyticsService {
  private systemMetricsSubject = new BehaviorSubject<SystemMetrics | null>(null);
  private performanceMetricsSubject = new BehaviorSubject<PerformanceMetrics | null>(null);
  private usageAnalyticsSubject = new BehaviorSubject<UsageAnalytics | null>(null);
  private consciousnessAnalyticsSubject = new BehaviorSubject<ConsciousnessAnalytics | null>(null);
  private aiToolAnalyticsSubject = new BehaviorSubject<AIToolAnalytics | null>(null);
  private securityMetricsSubject = new BehaviorSubject<SecurityMetrics | null>(null);
  private businessMetricsSubject = new BehaviorSubject<BusinessMetrics | null>(null);
  private alertsSubject = new BehaviorSubject<Alert[]>([]);
  private alertConfigsSubject = new BehaviorSubject<AlertConfig[]>([]);
  private dashboardWidgetsSubject = new BehaviorSubject<DashboardWidget[]>([]);

  // Public observables
  public systemMetrics$ = this.systemMetricsSubject.asObservable();
  public performanceMetrics$ = this.performanceMetricsSubject.asObservable();
  public usageAnalytics$ = this.usageAnalyticsSubject.asObservable();
  public consciousnessAnalytics$ = this.consciousnessAnalyticsSubject.asObservable();
  public aiToolAnalytics$ = this.aiToolAnalyticsSubject.asObservable();
  public securityMetrics$ = this.securityMetricsSubject.asObservable();
  public businessMetrics$ = this.businessMetricsSubject.asObservable();
  public alerts$ = this.alertsSubject.asObservable();
  public alertConfigs$ = this.alertConfigsSubject.asObservable();
  public dashboardWidgets$ = this.dashboardWidgetsSubject.asObservable();

  // Combined analytics stream
  public allMetrics$ = combineLatest([
    this.systemMetrics$,
    this.performanceMetrics$,
    this.usageAnalytics$,
    this.consciousnessAnalytics$,
    this.aiToolAnalytics$,
    this.securityMetrics$,
    this.businessMetrics$
  ]).pipe(
    map(([system, performance, usage, consciousness, aiTool, security, business]) => ({
      system,
      performance,
      usage,
      consciousness,
      aiTool,
      security,
      business,
      lastUpdated: new Date()
    }))
  );

  // Real-time data updates
  private metricsUpdateInterval = interval(5000); // 5 seconds
  private alertCheckInterval = interval(30000); // 30 seconds

  constructor(private apiService: ApiService) {
    this.initializeDefaultAlertConfigs();
    this.initializeDefaultDashboard();
    this.startRealTimeUpdates();
  }

  private initializeDefaultAlertConfigs(): void {
    const defaultConfigs: AlertConfig[] = [
      {
        id: 'cpu-high',
        name: 'High CPU Usage',
        metric: 'system.cpu.usage',
        condition: 'greater_than',
        threshold: 80,
        severity: 'warning',
        enabled: true,
        notifications: ['email'],
        cooldownPeriod: 5
      },
      {
        id: 'memory-critical',
        name: 'Critical Memory Usage',
        metric: 'system.memory.percentage',
        condition: 'greater_than',
        threshold: 95,
        severity: 'critical',
        enabled: true,
        notifications: ['email', 'slack'],
        cooldownPeriod: 2
      },
      {
        id: 'api-slow',
        name: 'Slow API Response',
        metric: 'performance.apiResponseTime',
        condition: 'greater_than',
        threshold: 2000,
        severity: 'warning',
        enabled: true,
        notifications: ['email'],
        cooldownPeriod: 10
      },
      {
        id: 'error-rate-high',
        name: 'High Error Rate',
        metric: 'performance.errorRate',
        condition: 'greater_than',
        threshold: 5,
        severity: 'error',
        enabled: true,
        notifications: ['email', 'slack'],
        cooldownPeriod: 5
      },
      {
        id: 'security-threat',
        name: 'Security Threat Detected',
        metric: 'security.threatLevel',
        condition: 'equals',
        threshold: 3, // critical = 3
        severity: 'critical',
        enabled: true,
        notifications: ['email', 'sms', 'slack'],
        cooldownPeriod: 1
      }
    ];

    this.alertConfigsSubject.next(defaultConfigs);
  }

  private initializeDefaultDashboard(): void {
    const defaultWidgets: DashboardWidget[] = [
      {
        id: 'system-overview',
        type: 'metric',
        title: 'System Overview',
        description: 'Key system metrics at a glance',
        dataSource: 'system',
        config: { metrics: ['cpu.usage', 'memory.percentage', 'disk.percentage'] },
        position: { x: 0, y: 0, width: 4, height: 2 },
        refreshInterval: 5,
        visible: true
      },
      {
        id: 'performance-chart',
        type: 'chart',
        title: 'Performance Trends',
        description: 'API response time and throughput over time',
        dataSource: 'performance',
        config: { 
          chartType: 'line',
          metrics: ['apiResponseTime', 'throughput'],
          timeRange: '1h'
        },
        position: { x: 4, y: 0, width: 8, height: 4 },
        refreshInterval: 30,
        visible: true
      },
      {
        id: 'consciousness-metrics',
        type: 'gauge',
        title: 'Consciousness Processing',
        description: 'Real-time consciousness analytics',
        dataSource: 'consciousness',
        config: { 
          metrics: ['averagePerformanceGain', 'correlationAccuracy', 'successRate']
        },
        position: { x: 0, y: 2, width: 4, height: 3 },
        refreshInterval: 10,
        visible: true
      },
      {
        id: 'ai-tools-usage',
        type: 'chart',
        title: 'AI Tools Usage',
        description: 'Most popular AI tools and execution stats',
        dataSource: 'aiTool',
        config: { 
          chartType: 'bar',
          metric: 'mostUsedTools',
          limit: 10
        },
        position: { x: 0, y: 5, width: 6, height: 3 },
        refreshInterval: 60,
        visible: true
      },
      {
        id: 'security-status',
        type: 'metric',
        title: 'Security Status',
        description: 'Current security posture and threats',
        dataSource: 'security',
        config: { 
          metrics: ['threatLevel', 'blockedRequests', 'failedLogins'],
          alertOnCritical: true
        },
        position: { x: 6, y: 5, width: 6, height: 3 },
        refreshInterval: 15,
        visible: true
      },
      {
        id: 'user-analytics',
        type: 'table',
        title: 'User Analytics',
        description: 'User engagement and behavior metrics',
        dataSource: 'usage',
        config: { 
          columns: ['activeUsers', 'sessionDuration', 'bounceRate', 'conversionRate']
        },
        position: { x: 0, y: 8, width: 12, height: 3 },
        refreshInterval: 300,
        visible: true
      }
    ];

    this.dashboardWidgetsSubject.next(defaultWidgets);
  }

  private startRealTimeUpdates(): void {
    // System metrics updates
    this.metricsUpdateInterval.subscribe(() => {
      this.updateSystemMetrics();
      this.updatePerformanceMetrics();
      this.updateUsageAnalytics();
      this.updateConsciousnessAnalytics();
      this.updateAIToolAnalytics();
      this.updateSecurityMetrics();
      this.updateBusinessMetrics();
    });

    // Alert checking
    this.alertCheckInterval.subscribe(() => {
      this.checkAlerts();
    });
  }

  private async updateSystemMetrics(): Promise<void> {
    try {
      // Mock system monitoring for now
      const mockResult = { success: true, result: {
        cpu: { 
          usage: 45.2 + Math.random() * 20,
          cores: 8,
          temperature: 45 + Math.random() * 20,
          frequency: 2.4 + Math.random() * 1.6
        },
        memory: { 
          used: 8.2 * 1024 * 1024 * 1024,
          total: 16 * 1024 * 1024 * 1024,
          percentage: 51.25 + Math.random() * 20,
          available: 16384 - (Math.random() * 16000)
        },
        disk: {
          used: 256 * 1024 * 1024 * 1024,
          total: 512 * 1024 * 1024 * 1024,
          percentage: 50 + Math.random() * 30,
          available: 500000 + Math.random() * 500000,
          readSpeed: Math.random() * 200,
          writeSpeed: Math.random() * 150
        },
        network: {
          bytesIn: Math.random() * 1000000,
          bytesOut: Math.random() * 500000,
          packetsIn: Math.random() * 10000,
          packetsOut: Math.random() * 8000,
          latency: Math.random() * 50,
          bandwidth: 100 + Math.random() * 900
        },
        gpu: {
          usage: Math.random() * 100,
          memory: Math.random() * 8192,
          temperature: 50 + Math.random() * 30,
          fanSpeed: 1000 + Math.random() * 2000
        }
      }};

      if (mockResult?.success) {
        const metrics: SystemMetrics = {
          cpu: {
            usage: mockResult.result.cpu?.usage || Math.random() * 100,
            cores: mockResult.result.cpu?.cores || 8,
            temperature: mockResult.result.cpu?.temperature || 45 + Math.random() * 20,
            frequency: mockResult.result.cpu?.frequency || 2.4 + Math.random() * 1.6
          },
          memory: {
            used: mockResult.result.memory?.used || Math.random() * 16000,
            total: mockResult.result.memory?.total || 16384,
            available: mockResult.result.memory?.available || 16384 - (Math.random() * 16000),
            percentage: mockResult.result.memory?.percentage || Math.random() * 100
          },
          disk: {
            used: mockResult.result.disk?.used || Math.random() * 500000,
            total: mockResult.result.disk?.total || 1000000,
            available: mockResult.result.disk?.available || 500000 + Math.random() * 500000,
            percentage: mockResult.result.disk?.percentage || Math.random() * 100,
            readSpeed: mockResult.result.disk?.readSpeed || Math.random() * 200,
            writeSpeed: mockResult.result.disk?.writeSpeed || Math.random() * 150
          },
          network: {
            bytesIn: mockResult.result.network?.bytesIn || Math.random() * 1000000,
            bytesOut: mockResult.result.network?.bytesOut || Math.random() * 800000,
            packetsIn: mockResult.result.network?.packetsIn || Math.random() * 10000,
            packetsOut: mockResult.result.network?.packetsOut || Math.random() * 8000,
            latency: mockResult.result.network?.latency || Math.random() * 50,
            bandwidth: mockResult.result.network?.bandwidth || 100 + Math.random() * 900
          },
          gpu: {
            usage: mockResult.result.gpu?.usage || Math.random() * 100,
            memory: mockResult.result.gpu?.memory || Math.random() * 8192,
            temperature: mockResult.result.gpu?.temperature || 50 + Math.random() * 30,
            fanSpeed: mockResult.result.gpu?.fanSpeed || 1000 + Math.random() * 2000
          },
          timestamp: new Date()
        };

        this.systemMetricsSubject.next(metrics);
      }
    } catch (error) {
      console.error('Failed to update system metrics:', error);
      // Generate mock data on error
      this.generateMockSystemMetrics();
    }
  }

  private generateMockSystemMetrics(): void {
    const metrics: SystemMetrics = {
      cpu: {
        usage: 20 + Math.random() * 60,
        cores: 8,
        temperature: 45 + Math.random() * 20,
        frequency: 2.4 + Math.random() * 1.6
      },
      memory: {
        used: 8000 + Math.random() * 6000,
        total: 16384,
        available: 2000 + Math.random() * 6000,
        percentage: 50 + Math.random() * 40
      },
      disk: {
        used: 200000 + Math.random() * 300000,
        total: 1000000,
        available: 500000 + Math.random() * 300000,
        percentage: 30 + Math.random() * 50,
        readSpeed: 100 + Math.random() * 100,
        writeSpeed: 80 + Math.random() * 70
      },
      network: {
        bytesIn: Math.random() * 1000000,
        bytesOut: Math.random() * 800000,
        packetsIn: Math.random() * 10000,
        packetsOut: Math.random() * 8000,
        latency: 10 + Math.random() * 30,
        bandwidth: 500 + Math.random() * 500
      },
      gpu: {
        usage: Math.random() * 80,
        memory: Math.random() * 8192,
        temperature: 50 + Math.random() * 25,
        fanSpeed: 1200 + Math.random() * 1800
      },
      timestamp: new Date()
    };

    this.systemMetricsSubject.next(metrics);
  }

  private async updatePerformanceMetrics(): Promise<void> {
    // Generate mock performance metrics
    const metrics: PerformanceMetrics = {
      apiResponseTime: 100 + Math.random() * 500,
      databaseQueryTime: 50 + Math.random() * 200,
      consciousnessProcessingTime: 200 + Math.random() * 800,
      quantumSimulationTime: 500 + Math.random() * 1500,
      mathVisualizationTime: 100 + Math.random() * 400,
      frontendRenderTime: 16 + Math.random() * 50,
      totalRequestsPerSecond: 10 + Math.random() * 90,
      errorRate: Math.random() * 5,
      uptime: 99.5 + Math.random() * 0.5,
      throughput: 1000 + Math.random() * 4000,
      timestamp: new Date()
    };

    this.performanceMetricsSubject.next(metrics);
  }

  private async updateUsageAnalytics(): Promise<void> {
    // Generate mock usage analytics
    const metrics: UsageAnalytics = {
      totalUsers: 1000 + Math.floor(Math.random() * 5000),
      activeUsers: 100 + Math.floor(Math.random() * 500),
      newUsers: Math.floor(Math.random() * 50),
      sessionDuration: 300 + Math.random() * 1200, // 5-25 minutes
      pageViews: 5000 + Math.floor(Math.random() * 10000),
      bounceRate: 0.2 + Math.random() * 0.3,
      conversionRate: 0.05 + Math.random() * 0.15,
      retentionRate: 0.7 + Math.random() * 0.25,
      featureUsage: {
        'consciousness-processing': Math.floor(Math.random() * 1000),
        'quantum-simulation': Math.floor(Math.random() * 500),
        'math-visualization': Math.floor(Math.random() * 800),
        'ai-chat': Math.floor(Math.random() * 1200),
        'analytics': Math.floor(Math.random() * 300)
      },
      geographicDistribution: {
        'US': 40 + Math.random() * 20,
        'EU': 25 + Math.random() * 15,
        'Asia': 20 + Math.random() * 15,
        'Other': 15 + Math.random() * 10
      },
      deviceTypes: {
        'Desktop': 60 + Math.random() * 20,
        'Mobile': 25 + Math.random() * 15,
        'Tablet': 10 + Math.random() * 10,
        'Other': Math.random() * 5
      },
      browserDistribution: {
        'Chrome': 50 + Math.random() * 20,
        'Firefox': 20 + Math.random() * 10,
        'Safari': 15 + Math.random() * 10,
        'Edge': 10 + Math.random() * 10,
        'Other': Math.random() * 5
      },
      timestamp: new Date()
    };

    this.usageAnalyticsSubject.next(metrics);
  }

  private async updateConsciousnessAnalytics(): Promise<void> {
    // Generate mock consciousness analytics
    const metrics: ConsciousnessAnalytics = {
      totalProcessingRequests: Math.floor(Math.random() * 10000),
      averageProcessingTime: 200 + Math.random() * 800,
      successRate: 0.95 + Math.random() * 0.05,
      wallaceTransformUsage: Math.floor(Math.random() * 5000),
      moebiusOptimizationUsage: Math.floor(Math.random() * 3000),
      quantumConsciousnessRequests: Math.floor(Math.random() * 2000),
      averagePerformanceGain: 150 + Math.random() * 50,
      correlationAccuracy: 0.99 + Math.random() * 0.01,
      consciousnessIndexDistribution: Array.from({length: 10}, () => Math.random() * 100),
      emergencePatterns: {
        'harmonic-resonance': Math.floor(Math.random() * 500),
        'quantum-coherence': Math.floor(Math.random() * 400),
        'chiral-alignment': Math.floor(Math.random() * 600),
        'phi-optimization': Math.floor(Math.random() * 700)
      },
      timestamp: new Date()
    };

    this.consciousnessAnalyticsSubject.next(metrics);
  }

  private async updateAIToolAnalytics(): Promise<void> {
    // Generate mock AI tool analytics
    const metrics: AIToolAnalytics = {
      totalToolExecutions: Math.floor(Math.random() * 50000),
      mostUsedTools: {
        'grok_generate_code': Math.floor(Math.random() * 5000),
        'wallace_transform_advanced': Math.floor(Math.random() * 3000),
        'quantum_consciousness_simulator': Math.floor(Math.random() * 2000),
        'file_operations': Math.floor(Math.random() * 8000),
        'data_analysis': Math.floor(Math.random() * 4000),
        'research_assistant': Math.floor(Math.random() * 3500),
        'visualization_generator': Math.floor(Math.random() * 2500)
      },
      averageExecutionTime: {
        'grok_generate_code': 500 + Math.random() * 1000,
        'wallace_transform_advanced': 1000 + Math.random() * 2000,
        'quantum_consciousness_simulator': 2000 + Math.random() * 3000,
        'file_operations': 100 + Math.random() * 200,
        'data_analysis': 300 + Math.random() * 700
      },
      successRates: {
        'grok_generate_code': 0.95 + Math.random() * 0.05,
        'wallace_transform_advanced': 0.98 + Math.random() * 0.02,
        'quantum_consciousness_simulator': 0.92 + Math.random() * 0.08,
        'file_operations': 0.99 + Math.random() * 0.01,
        'data_analysis': 0.94 + Math.random() * 0.06
      },
      errorPatterns: {
        'timeout': Math.floor(Math.random() * 100),
        'invalid_parameters': Math.floor(Math.random() * 200),
        'insufficient_permissions': Math.floor(Math.random() * 50),
        'resource_unavailable': Math.floor(Math.random() * 75),
        'processing_error': Math.floor(Math.random() * 150)
      },
      llmIntegrationStats: {
        chatgpt: Math.floor(Math.random() * 10000),
        claude: Math.floor(Math.random() * 8000),
        gemini: Math.floor(Math.random() * 6000),
        other: Math.floor(Math.random() * 2000)
      },
      batchExecutionStats: {
        total: Math.floor(Math.random() * 1000),
        averageSize: 3 + Math.random() * 7,
        successRate: 0.88 + Math.random() * 0.12
      },
      timestamp: new Date()
    };

    this.aiToolAnalyticsSubject.next(metrics);
  }

  private async updateSecurityMetrics(): Promise<void> {
    // Generate mock security metrics
    const threatLevels = ['low', 'medium', 'high', 'critical'] as const;
    const metrics: SecurityMetrics = {
      totalRequests: Math.floor(Math.random() * 100000),
      blockedRequests: Math.floor(Math.random() * 1000),
      suspiciousActivity: Math.floor(Math.random() * 200),
      failedLogins: Math.floor(Math.random() * 500),
      bruteForceAttempts: Math.floor(Math.random() * 50),
      sqlInjectionAttempts: Math.floor(Math.random() * 25),
      xssAttempts: Math.floor(Math.random() * 30),
      ddosAttempts: Math.floor(Math.random() * 10),
      vulnerabilityScans: Math.floor(Math.random() * 100),
      threatLevel: threatLevels[Math.floor(Math.random() * threatLevels.length)],
      lastSecurityScan: new Date(Date.now() - Math.random() * 86400000), // Within last day
      certificateExpiry: new Date(Date.now() + 30 * 86400000 + Math.random() * 335 * 86400000), // 30-365 days
      timestamp: new Date()
    };

    this.securityMetricsSubject.next(metrics);
  }

  private async updateBusinessMetrics(): Promise<void> {
    // Generate mock business metrics
    const metrics: BusinessMetrics = {
      revenue: 10000 + Math.random() * 90000,
      costs: 5000 + Math.random() * 30000,
      profit: 5000 + Math.random() * 60000,
      customerAcquisitionCost: 50 + Math.random() * 200,
      lifetimeValue: 500 + Math.random() * 2000,
      churnRate: 0.05 + Math.random() * 0.15,
      monthlyRecurringRevenue: 8000 + Math.random() * 40000,
      apiCallsRevenue: 3000 + Math.random() * 15000,
      subscriptionRevenue: 5000 + Math.random() * 25000,
      growthRate: 0.1 + Math.random() * 0.4,
      marketShare: 0.01 + Math.random() * 0.05,
      competitorAnalysis: {
        'Competitor A': 0.15 + Math.random() * 0.1,
        'Competitor B': 0.12 + Math.random() * 0.08,
        'Competitor C': 0.08 + Math.random() * 0.06,
        'Others': 0.6 + Math.random() * 0.1
      },
      timestamp: new Date()
    };

    this.businessMetricsSubject.next(metrics);
  }

  private checkAlerts(): void {
    const configs = this.alertConfigsSubject.value;
    const currentAlerts = this.alertsSubject.value;
    
    configs.forEach(config => {
      if (!config.enabled) return;
      
      // Check if alert is in cooldown period
      const lastAlert = currentAlerts
        .filter(alert => alert.configId === config.id)
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];
      
      if (lastAlert && !lastAlert.acknowledged) {
        const cooldownEnd = new Date(lastAlert.timestamp.getTime() + config.cooldownPeriod * 60000);
        if (new Date() < cooldownEnd) {
          return; // Still in cooldown
        }
      }
      
      // Get current metric value
      const metricValue = this.getMetricValue(config.metric);
      if (metricValue === null) return;
      
      // Check condition
      let triggered = false;
      switch (config.condition) {
        case 'greater_than':
          triggered = metricValue > config.threshold;
          break;
        case 'less_than':
          triggered = metricValue < config.threshold;
          break;
        case 'equals':
          triggered = metricValue === config.threshold;
          break;
        case 'not_equals':
          triggered = metricValue !== config.threshold;
          break;
      }
      
      if (triggered) {
        this.createAlert(config, metricValue);
      }
    });
  }

  private getMetricValue(metricPath: string): number | null {
    const parts = metricPath.split('.');
    const category = parts[0];
    
    let data: any = null;
    switch (category) {
      case 'system':
        data = this.systemMetricsSubject.value;
        break;
      case 'performance':
        data = this.performanceMetricsSubject.value;
        break;
      case 'usage':
        data = this.usageAnalyticsSubject.value;
        break;
      case 'consciousness':
        data = this.consciousnessAnalyticsSubject.value;
        break;
      case 'security':
        data = this.securityMetricsSubject.value;
        break;
      default:
        return null;
    }
    
    if (!data) return null;
    
    // Navigate through nested properties
    let value = data;
    for (let i = 1; i < parts.length; i++) {
      if (value && typeof value === 'object' && parts[i] in value) {
        value = value[parts[i]];
      } else {
        return null;
      }
    }
    
    return typeof value === 'number' ? value : null;
  }

  private createAlert(config: AlertConfig, currentValue: number): void {
    const alert: Alert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      configId: config.id,
      message: `${config.name}: ${config.metric} is ${currentValue} (threshold: ${config.threshold})`,
      severity: config.severity,
      timestamp: new Date(),
      acknowledged: false,
      metadata: {
        metric: config.metric,
        currentValue,
        threshold: config.threshold,
        condition: config.condition
      }
    };
    
    const currentAlerts = this.alertsSubject.value;
    this.alertsSubject.next([alert, ...currentAlerts]);
    
    // Send notifications (mock implementation)
    this.sendAlertNotifications(alert, config);
  }

  private sendAlertNotifications(alert: Alert, config: AlertConfig): void {
    config.notifications.forEach(method => {
      switch (method) {
        case 'email':
          console.log(`📧 Email alert sent: ${alert.message}`);
          break;
        case 'sms':
          console.log(`📱 SMS alert sent: ${alert.message}`);
          break;
        case 'slack':
          console.log(`💬 Slack alert sent: ${alert.message}`);
          break;
        case 'webhook':
          console.log(`🔗 Webhook alert sent: ${alert.message}`);
          break;
      }
    });
  }

  // Public methods
  acknowledgeAlert(alertId: string): void {
    const alerts = this.alertsSubject.value;
    const updatedAlerts = alerts.map(alert => 
      alert.id === alertId 
        ? { ...alert, acknowledged: true }
        : alert
    );
    this.alertsSubject.next(updatedAlerts);
  }

  resolveAlert(alertId: string): void {
    const alerts = this.alertsSubject.value;
    const updatedAlerts = alerts.map(alert => 
      alert.id === alertId 
        ? { ...alert, acknowledged: true, resolvedAt: new Date() }
        : alert
    );
    this.alertsSubject.next(updatedAlerts);
  }

  addAlertConfig(config: Omit<AlertConfig, 'id'>): void {
    const newConfig: AlertConfig = {
      ...config,
      id: `config-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    };
    
    const configs = this.alertConfigsSubject.value;
    this.alertConfigsSubject.next([...configs, newConfig]);
  }

  updateAlertConfig(configId: string, updates: Partial<AlertConfig>): void {
    const configs = this.alertConfigsSubject.value;
    const updatedConfigs = configs.map(config =>
      config.id === configId
        ? { ...config, ...updates }
        : config
    );
    this.alertConfigsSubject.next(updatedConfigs);
  }

  removeAlertConfig(configId: string): void {
    const configs = this.alertConfigsSubject.value;
    const updatedConfigs = configs.filter(config => config.id !== configId);
    this.alertConfigsSubject.next(updatedConfigs);
  }

  addDashboardWidget(widget: Omit<DashboardWidget, 'id'>): void {
    const newWidget: DashboardWidget = {
      ...widget,
      id: `widget-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    };
    
    const widgets = this.dashboardWidgetsSubject.value;
    this.dashboardWidgetsSubject.next([...widgets, newWidget]);
  }

  updateDashboardWidget(widgetId: string, updates: Partial<DashboardWidget>): void {
    const widgets = this.dashboardWidgetsSubject.value;
    const updatedWidgets = widgets.map(widget =>
      widget.id === widgetId
        ? { ...widget, ...updates }
        : widget
    );
    this.dashboardWidgetsSubject.next(updatedWidgets);
  }

  removeDashboardWidget(widgetId: string): void {
    const widgets = this.dashboardWidgetsSubject.value;
    const updatedWidgets = widgets.filter(widget => widget.id !== widgetId);
    this.dashboardWidgetsSubject.next(updatedWidgets);
  }

  exportAnalytics(format: 'json' | 'csv' | 'pdf', timeRange: string): any {
    const data = {
      system: this.systemMetricsSubject.value,
      performance: this.performanceMetricsSubject.value,
      usage: this.usageAnalyticsSubject.value,
      consciousness: this.consciousnessAnalyticsSubject.value,
      aiTool: this.aiToolAnalyticsSubject.value,
      security: this.securityMetricsSubject.value,
      business: this.businessMetricsSubject.value,
      alerts: this.alertsSubject.value,
      exportedAt: new Date().toISOString(),
      timeRange
    };
    
    switch (format) {
      case 'json':
        return JSON.stringify(data, null, 2);
      case 'csv':
        return this.convertToCSV(data);
      case 'pdf':
        return this.generatePDFReport(data);
      default:
        return data;
    }
  }

  private convertToCSV(data: any): string {
    // Simplified CSV conversion
    const lines: string[] = [];
    
    // System metrics
    if (data.system) {
      lines.push('System Metrics');
      lines.push('Metric,Value');
      lines.push(`CPU Usage,${data.system.cpu.usage}`);
      lines.push(`Memory Usage,${data.system.memory.percentage}`);
      lines.push(`Disk Usage,${data.system.disk.percentage}`);
      lines.push('');
    }
    
    return lines.join('\n');
  }

  private generatePDFReport(data: any): string {
    // Mock PDF generation
    return `PDF Report Generated for ${data.exportedAt}`;
  }

  getHealthScore(): Observable<number> {
    return this.allMetrics$.pipe(
      map(metrics => {
        let score = 100;
        
        // System health impact
        if (metrics.system) {
          if (metrics.system.cpu.usage > 80) score -= 10;
          if (metrics.system.memory.percentage > 90) score -= 15;
          if (metrics.system.disk.percentage > 95) score -= 10;
        }
        
        // Performance impact
        if (metrics.performance) {
          if (metrics.performance.errorRate > 5) score -= 20;
          if (metrics.performance.apiResponseTime > 2000) score -= 15;
          if (metrics.performance.uptime < 99) score -= 25;
        }
        
        // Security impact
        if (metrics.security) {
          if (metrics.security.threatLevel === 'critical') score -= 30;
          else if (metrics.security.threatLevel === 'high') score -= 20;
          else if (metrics.security.threatLevel === 'medium') score -= 10;
        }
        
        return Math.max(0, score);
      })
    );
  }
}
