import { bootstrapApplication } from '@angular/platform-browser';
import { RouteReuseStrategy, provideRouter, withPreloading, PreloadAllModules } from '@angular/router';
import { IonicRouteStrategy, provideIonicAngular } from '@ionic/angular/standalone';
import { provideHttpClient } from '@angular/common/http';
import { importProvidersFrom } from '@angular/core';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { IonicModule } from '@ionic/angular';

import { AppComponent } from './app/app.component';
import { routes } from './app/app.routes';

// Import services
import { ApiService } from './app/core/api.service';
import { ConsciousnessService } from './app/features/consciousness/services/consciousness.service';
import { UXService } from './app/core/ux.service';

bootstrapApplication(AppComponent, {
  providers: [
    { provide: RouteReuseStrategy, useClass: IonicRouteStrategy },
    provideIonicAngular({
      mode: 'ios', // Use iOS mode for consistent styling
      rippleEffect: true,
      animated: true,
      backButtonText: '',
      backButtonIcon: 'chevron-back-outline',
      menuIcon: 'menu-outline',
      menuType: 'overlay',
      platform: {
        // Platform-specific configurations
        desktop: (win) => {
          const userAgent = win.navigator.userAgent.toLowerCase();
          return !(userAgent.indexOf('mobile') > -1 || userAgent.indexOf('tablet') > -1);
        }
      }
    }),
    provideRouter(routes, withPreloading(PreloadAllModules)),
    provideHttpClient(),
    importProvidersFrom(
      BrowserAnimationsModule,
      IonicModule.forRoot({}) // <--- CRITICAL FIX: Provide Ionic controllers
    ),
    
    // Core Services
    ApiService,
    ConsciousnessService,
    UXService,
  ],
}).catch(err => {
  console.error('Error starting chAIos application:', err);
  
  // Show error message to user
  const errorDiv = document.createElement('div');
  errorDiv.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      color: #F1F5F9;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: 'Inter', sans-serif;
      z-index: 10000;
    ">
      <div style="text-align: center; max-width: 500px; padding: 2rem;">
        <h1 style="font-size: 2rem; color: #EF4444; margin-bottom: 1rem;">
          ⚠️ chAIos Initialization Error
        </h1>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem; color: #94A3B8;">
          The consciousness matrix failed to initialize. Please refresh the page or contact support.
        </p>
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
          <code style="color: #F59E0B; font-size: 0.9rem;">${err.message}</code>
        </div>
        <button onclick="window.location.reload()" style="
          background: linear-gradient(45deg, #D4AF37, #8A2BE2);
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 8px;
          font-size: 1rem;
          cursor: pointer;
          font-weight: 600;
        ">
          Refresh Application
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(errorDiv);
});