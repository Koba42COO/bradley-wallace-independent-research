import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/ai-chat',
    pathMatch: 'full',
  },
  {
    path: 'welcome',
    loadComponent: () => import('./pages/welcome/welcome.page').then(m => m.WelcomePage)
  },
  {
    path: 'dashboard',
    redirectTo: '/ai-chat',
    pathMatch: 'full'
  },
  {
    path: 'ai-chat',
    loadChildren: () => import('./features/ai-chat/ai-chat.routes').then(m => m.routes),
  },
  {
    path: 'consciousness',
    loadChildren: () => import('./features/consciousness/consciousness.routes').then(m => m.routes),
  },
  {
    path: 'quantum',
    loadChildren: () => import('./features/quantum/quantum.routes').then(m => m.quantumRoutes),
  },
  {
    path: 'mathematics',
    loadChildren: () => import('./features/math-visualizations/math-visualizations.routes').then(m => m.mathVisualizationRoutes),
  },
  {
    path: 'analytics',
    loadChildren: () => import('./features/analytics/analytics.routes').then(m => m.analyticsRoutes),
  },
  {
    path: '**',
    redirectTo: '/ai-chat',
  },
];