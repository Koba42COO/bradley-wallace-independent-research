import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./pages/ai-chat.page').then(m => m.AiChatPage)
  }
];

