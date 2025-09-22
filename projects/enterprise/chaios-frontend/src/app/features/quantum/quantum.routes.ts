import { Routes } from '@angular/router';
import { AuthGuard } from '../../core/guards/auth.guard';

export const quantumRoutes: Routes = [
  {
    path: '',
    loadComponent: () => import('./quantum.page').then(m => m.QuantumPage),
    canActivate: [AuthGuard],
    data: { 
      title: 'Quantum Simulation',
      description: 'Advanced quantum consciousness simulation and analysis'
    }
  }
];
