import { bootstrapApplication } from '@angular/platform-browser';
import { RouteReuseStrategy } from '@angular/router';
import { IonicRouteStrategy, provideIonicAngular } from '@ionic/angular/standalone';
import { provideHttpClient } from '@angular/common/http';

import { SimpleAppComponent } from './app/app.component.simple';

/**
 * Simple Bootstrap for Debugging
 * ==============================
 * Minimal bootstrap to test if the issue is with complex dependencies
 */

console.log('🔍 Starting simple bootstrap...');

bootstrapApplication(SimpleAppComponent, {
  providers: [
    { provide: RouteReuseStrategy, useClass: IonicRouteStrategy },
    provideIonicAngular({
      mode: 'ios',
      rippleEffect: true,
      animated: true
    }),
    provideHttpClient()
  ],
}).then(() => {
  console.log('✅ Simple bootstrap successful!');
}).catch(err => {
  console.error('❌ Simple bootstrap failed:', err);
  
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
          ⚠️ Bootstrap Error
        </h1>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem; color: #94A3B8;">
          Simple component failed to initialize.
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
          Refresh
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(errorDiv);
});
