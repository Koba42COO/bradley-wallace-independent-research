import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, combineLatest, merge, EMPTY, of, throwError } from 'rxjs';
import { catchError, map, switchMap, tap, retry, timeout, shareReplay, distinctUntilChanged } from 'rxjs/operators';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';

import { environment } from '../../../environments/environment';
import { AuthService } from './auth.service';
import { ApiService } from './api.service';
import { WebSocketService } from './websocket.service';

/**
 * chAIos Orchestration Service
 * ============================
 * Central orchestration layer that coordinates all services
 * Acts as a "butler" managing authentication, data retrieval, and unified roles
 * Following Jeff's architectural pattern for service separation
 */

// Core Interfaces
export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency?: number;
  lastCheck: Date;
  error?: string;
}

export interface SystemState {
  authenticated: boolean;
  connected: boolean;
  services: ServiceHealth[];
  permissions: string[];
  user: any;
  features: { [key: string]: boolean };
}

export interface OrchestrationConfig {
  healthCheckInterval: number;
  retryAttempts: number;
  timeoutMs: number;
  enableCaching: boolean;
  enableOfflineMode: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class OrchestrationService {
  // State Management
  private systemState$ = new BehaviorSubject<SystemState>({
    authenticated: false,
    connected: false,
    services: [],
    permissions: [],
    user: null,
    features: {}
  });

  private config: OrchestrationConfig = {
    healthCheckInterval: 30000, // 30 seconds
    retryAttempts: 3,
    timeoutMs: 10000, // 10 seconds
    enableCaching: true,
    enableOfflineMode: true
  };

  // Service Registry
  private services = new Map<string, any>();
  private healthChecks = new Map<string, Observable<ServiceHealth>>();
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();

  constructor(
    private http: HttpClient,
    private authService: AuthService,
    private apiService: ApiService,
    private wsService: WebSocketService
  ) {
    this.initializeOrchestration();
  }

  /**
   * Initialize Orchestration System
   * ==============================
   * Sets up service coordination, health monitoring, and state management
   */
  private initializeOrchestration(): void {
    console.log('🎭 Initializing chAIos Orchestration Service');

    // Register core services
    this.registerService('auth', this.authService);
    this.registerService('api', this.apiService);
    this.registerService('websocket', this.wsService);

    // Set up service health monitoring
    this.setupHealthMonitoring();

    // Initialize state coordination
    this.setupStateCoordination();

    // Set up error handling
    this.setupErrorHandling();

    console.log('✅ chAIos Orchestration Service initialized');
  }

  /**
   * Service Registration
   * ===================
   * Register services for orchestrated management
   */
  registerService(name: string, service: any): void {
    this.services.set(name, service);
    console.log(`📋 Service registered: ${name}`);
  }

  getService<T>(name: string): T | undefined {
    return this.services.get(name) as T;
  }

  /**
   * Health Monitoring
   * ================
   * Monitor health of all registered services
   */
  private setupHealthMonitoring(): void {
    // Auth Service Health
    this.healthChecks.set('auth', this.createHealthCheck('auth', () => 
      this.authService.getCurrentUser().pipe(
        map(() => ({ status: 'healthy' as const })),
        catchError(() => of({ status: 'unhealthy' as const, error: 'Authentication unavailable' }))
      )
    ));

    // API Service Health
    this.healthChecks.set('api', this.createHealthCheck('api', () =>
      this.http.get(`${environment.apiUrl}/health`).pipe(
        timeout(5000),
        map(() => ({ status: 'healthy' as const })),
        catchError((error) => of({ 
          status: 'unhealthy' as const, 
          error: error.message || 'API unavailable' 
        }))
      )
    ));

    // WebSocket Health
    this.healthChecks.set('websocket', this.createHealthCheck('websocket', () =>
      of({ 
        status: this.wsService.isConnected() ? 'healthy' as const : 'unhealthy' as const,
        error: this.wsService.isConnected() ? undefined : 'WebSocket disconnected'
      })
    ));

    // Start health monitoring
    this.startHealthMonitoring();
  }

  private createHealthCheck(serviceName: string, checkFn: () => Observable<any>): Observable<ServiceHealth> {
    return checkFn().pipe(
      map((result) => ({
        service: serviceName,
        status: result.status || 'healthy',
        lastCheck: new Date(),
        error: result.error
      })),
      catchError((error) => of({
        service: serviceName,
        status: 'unhealthy' as const,
        lastCheck: new Date(),
        error: error.message || 'Health check failed'
      }))
    );
  }

  private startHealthMonitoring(): void {
    setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);

    // Initial health check
    this.performHealthChecks();
  }

  private performHealthChecks(): void {
    const healthObservables = Array.from(this.healthChecks.values());
    
    combineLatest(healthObservables).pipe(
      tap((healthResults) => {
        const currentState = this.systemState$.value;
        const updatedState = {
          ...currentState,
          services: healthResults,
          connected: healthResults.every(h => h.status !== 'unhealthy')
        };
        this.systemState$.next(updatedState);
      }),
      catchError((error) => {
        console.error('❌ Health check failed:', error);
        return EMPTY;
      })
    ).subscribe();
  }

  /**
   * State Coordination
   * =================
   * Coordinate state across all services
   */
  private setupStateCoordination(): void {
    // Listen to auth state changes
    this.authService.getCurrentUser().pipe(
      distinctUntilChanged(),
      tap((user) => {
        const currentState = this.systemState$.value;
        this.systemState$.next({
          ...currentState,
          authenticated: !!user,
          user: user,
          permissions: user?.permissions || []
        });
      })
    ).subscribe();

    // Listen to WebSocket connection changes
    this.wsService.connectionStatus$.pipe(
      distinctUntilChanged(),
      tap((connected) => {
        const currentState = this.systemState$.value;
        this.systemState$.next({
          ...currentState,
          connected: connected === 'connected'
        });
      })
    ).subscribe();
  }

  /**
   * Error Handling
   * ==============
   * Centralized error handling and recovery
   */
  private setupErrorHandling(): void {
    // Global error handler for HTTP errors
    this.http.get('/api/health').pipe(
      catchError((error: HttpErrorResponse) => {
        this.handleServiceError('api', error);
        return throwError(error);
      })
    ).subscribe();
  }

  private handleServiceError(serviceName: string, error: any): void {
    console.error(`❌ Service error [${serviceName}]:`, error);

    // Update service health
    const currentState = this.systemState$.value;
    const updatedServices = currentState.services.map(service => 
      service.service === serviceName 
        ? { ...service, status: 'unhealthy' as const, error: error.message }
        : service
    );

    this.systemState$.next({
      ...currentState,
      services: updatedServices
    });

    // Implement recovery strategies
    this.attemptServiceRecovery(serviceName, error);
  }

  private attemptServiceRecovery(serviceName: string, error: any): void {
    console.log(`🔄 Attempting recovery for service: ${serviceName}`);

    switch (serviceName) {
      case 'auth':
        // Attempt to refresh authentication
        this.authService.refreshToken().subscribe({
          next: () => console.log('✅ Auth service recovered'),
          error: (err) => console.error('❌ Auth recovery failed:', err)
        });
        break;

      case 'websocket':
        // Attempt to reconnect WebSocket
        this.wsService.connect();
        break;

      case 'api':
        // Retry API connection after delay
        setTimeout(() => {
          this.performHealthChecks();
        }, 5000);
        break;
    }
  }

  /**
   * Data Orchestration
   * ==================
   * Unified data retrieval with caching and error handling
   */
  getData<T>(
    endpoint: string, 
    options: { 
      useCache?: boolean; 
      cacheTtl?: number; 
      retries?: number;
      requireAuth?: boolean;
    } = {}
  ): Observable<T> {
    const cacheKey = `data_${endpoint}`;
    const {
      useCache = this.config.enableCaching,
      cacheTtl = 300000, // 5 minutes
      retries = this.config.retryAttempts,
      requireAuth = true
    } = options;

    // Check cache first
    if (useCache && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      if (Date.now() - cached.timestamp < cached.ttl) {
        return of(cached.data);
      }
    }

    // Check authentication requirement
    if (requireAuth && !this.systemState$.value.authenticated) {
      return throwError(new Error('Authentication required'));
    }

    // Make API call with orchestration
    return this.apiService.get<T>(endpoint).pipe(
      timeout(this.config.timeoutMs),
      retry(retries),
      tap((data) => {
        // Cache successful response
        if (useCache) {
          this.cache.set(cacheKey, {
            data,
            timestamp: Date.now(),
            ttl: cacheTtl
          });
        }
      }),
      catchError((error) => {
        this.handleServiceError('api', error);
        return throwError(error);
      }),
      shareReplay(1)
    );
  }

  /**
   * Unified Role Management
   * ======================
   * Centralized permission and role checking
   */
  hasPermission(permission: string): boolean {
    const permissions = this.systemState$.value.permissions;
    return permissions.includes(permission) || permissions.includes('admin');
  }

  hasRole(role: string): boolean {
    const user = this.systemState$.value.user;
    return user?.role === role || user?.role === 'admin';
  }

  requirePermission(permission: string): Observable<boolean> {
    return this.systemState$.pipe(
      map(state => state.permissions.includes(permission) || state.permissions.includes('admin')),
      tap(hasPermission => {
        if (!hasPermission) {
          throw new Error(`Permission required: ${permission}`);
        }
      })
    );
  }

  /**
   * Feature Flag Management
   * ======================
   * Dynamic feature enablement
   */
  isFeatureEnabled(feature: string): boolean {
    return this.systemState$.value.features[feature] || false;
  }

  enableFeature(feature: string): void {
    const currentState = this.systemState$.value;
    this.systemState$.next({
      ...currentState,
      features: {
        ...currentState.features,
        [feature]: true
      }
    });
  }

  /**
   * Public Observables
   * ==================
   * Expose reactive state to components
   */
  get systemState(): Observable<SystemState> {
    return this.systemState$.asObservable();
  }

  get isAuthenticated(): Observable<boolean> {
    return this.systemState$.pipe(
      map(state => state.authenticated),
      distinctUntilChanged()
    );
  }

  get isConnected(): Observable<boolean> {
    return this.systemState$.pipe(
      map(state => state.connected),
      distinctUntilChanged()
    );
  }

  get serviceHealth(): Observable<ServiceHealth[]> {
    return this.systemState$.pipe(
      map(state => state.services),
      distinctUntilChanged()
    );
  }

  /**
   * Utility Methods
   * ===============
   */
  clearCache(): void {
    this.cache.clear();
    console.log('🗑️ Orchestration cache cleared');
  }

  getSystemInfo(): any {
    return {
      version: '1.0.0',
      services: Array.from(this.services.keys()),
      cacheSize: this.cache.size,
      uptime: Date.now(),
      config: this.config
    };
  }

  /**
   * Cleanup
   * =======
   */
  destroy(): void {
    this.systemState$.complete();
    this.clearCache();
    console.log('🧹 Orchestration service destroyed');
  }
}
