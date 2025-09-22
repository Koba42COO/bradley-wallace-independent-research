import { Routes } from '@angular/router';
import { AuthGuard } from '../../core/guards/auth.guard';

export const mathVisualizationRoutes: Routes = [
  {
    path: '',
    loadComponent: () => import('./math-visualizations.page').then(m => m.MathVisualizationsPage),
    canActivate: [AuthGuard],
    data: { 
      title: 'Mathematical Visualizations',
      description: 'Interactive mathematical and consciousness equation visualizations'
    }
  }
];
