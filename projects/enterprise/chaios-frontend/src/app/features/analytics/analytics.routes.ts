import { Routes } from '@angular/router';
import { AuthGuard } from '../../core/guards/auth.guard';

export const analyticsRoutes: Routes = [
  {
    path: '',
    loadComponent: () => import('./analytics.page').then(m => m.AnalyticsPage),
    canActivate: [AuthGuard],
    data: { 
      title: 'Analytics Dashboard',
      description: 'Comprehensive system analytics, performance metrics, and insights'
    }
  }
];
