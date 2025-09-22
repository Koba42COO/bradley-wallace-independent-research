import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./pages/consciousness.page').then(m => m.ConsciousnessPage)
  }
];

