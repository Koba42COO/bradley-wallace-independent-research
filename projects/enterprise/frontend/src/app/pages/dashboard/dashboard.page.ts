import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, Subscription } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

interface SystemStatus {
  gateway: {
    status: string;
    version: string;
    services: number;
  };
  services: {
    [key: string]: {
      name: string;
      status: string;
      port?: number;
      health: string;
      requests: number;
    };
  };
  metrics: {
    total_requests: number;
    error_rate: number;
    average_response_time: number;
  };
}

interface KnowledgeStats {
  total_documents: number;
  domains: { [key: string]: number };
  synthesis_types: { [key: string]: number };
}

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.page.html',
  styleUrls: ['./dashboard.page.scss'],
})
export class DashboardPage implements OnInit, OnDestroy {
  systemStatus$: Observable<SystemStatus>;
  knowledgeStats$: Observable<KnowledgeStats>;
  refreshSubscription: Subscription;

  // Dashboard sections
  dashboardSections = [
    {
      title: 'Knowledge Systems',
      icon: 'library',
      items: [
        { name: 'RAG System', route: '/knowledge', status: 'online', description: 'Document retrieval & synthesis' },
        { name: 'Polymath Brain', route: '/polymath', status: 'online', description: 'Advanced reasoning engine' },
        { name: 'Cross-Domain Mapping', route: '/cross-domain', status: 'online', description: 'Interdisciplinary connections' }
      ]
    },
    {
      title: 'AI & ML Systems',
      icon: 'bulb',
      items: [
        { name: 'CUDNT Accelerator', route: '/cudnt', status: 'online', description: 'GPU acceleration framework' },
        { name: 'Quantum Simulator', route: '/quantum', status: 'online', description: 'Quantum computing simulation' },
        { name: 'Knowledge Expansion', route: '/expansion', status: 'online', description: 'Continuous learning system' }
      ]
    },
    {
      title: 'Educational Tools',
      icon: 'school',
      items: [
        { name: 'Learning Pathways', route: '/learning', status: 'online', description: 'Personalized education journeys' },
        { name: 'Coding Trainer', route: '/coding', status: 'online', description: 'Programming fundamentals' },
        { name: 'Data Science Lab', route: '/data-science', status: 'online', description: 'Interactive ML environment' }
      ]
    },
    {
      title: 'Development Tools',
      icon: 'code-working',
      items: [
        { name: 'API Gateway', route: '/api-docs', status: 'online', description: 'Unified service access' },
        { name: 'System Monitor', route: '/monitoring', status: 'online', description: 'Performance analytics' },
        { name: 'Configuration', route: '/config', status: 'online', description: 'System configuration' }
      ]
    }
  ];

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit() {
    // Load initial data
    this.loadSystemStatus();
    this.loadKnowledgeStats();

    // Auto-refresh every 30 seconds
    this.refreshSubscription = interval(30000).subscribe(() => {
      this.loadSystemStatus();
      this.loadKnowledgeStats();
    });
  }

  ngOnDestroy() {
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
    }
  }

  loadSystemStatus() {
    this.systemStatus$ = this.http.get<SystemStatus>('/status').pipe(
      catchError(error => {
        console.error('Failed to load system status:', error);
        return [{
          gateway: { status: 'error', version: 'unknown', services: 0 },
          services: {},
          metrics: { total_requests: 0, error_rate: 0, average_response_time: 0 }
        }];
      })
    );
  }

  loadKnowledgeStats() {
    // Try to get knowledge stats from the knowledge system
    this.knowledgeStats$ = this.http.get<KnowledgeStats>('/knowledge/stats').pipe(
      catchError(error => {
        // Fallback to basic stats
        return [{
          total_documents: 10000,
          domains: { 'python': 2500, 'data_science': 2500, 'algorithms': 2500 },
          synthesis_types: { 'programming_fundamentals': 2500, 'data_science_ml': 2500 }
        }];
      })
    );
  }

  navigateTo(route: string) {
    this.router.navigate([route]);
  }

  getStatusColor(status: string): string {
    switch (status.toLowerCase()) {
      case 'online':
      case 'healthy':
        return 'success';
      case 'offline':
      case 'unhealthy':
        return 'danger';
      case 'maintenance':
        return 'warning';
      default:
        return 'medium';
    }
  }

  getHealthIcon(health: string): string {
    switch (health.toLowerCase()) {
      case 'healthy':
        return 'heart';
      case 'unhealthy':
        return 'heart-dislike';
      case 'maintenance':
        return 'construct';
      default:
        return 'help-circle';
    }
  }

  formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }

  refreshData() {
    this.loadSystemStatus();
    this.loadKnowledgeStats();
  }
}
