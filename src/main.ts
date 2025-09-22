import { routes } from './app/app.routes';
import { PLATFORM_PROVIDERS } from './app/platform-providers';

// Import services
import { ApiService } from './app/core/api.service';
import { ConsciousnessService } from './app/features/consciousness/services/consciousness.service';
import { UXService } from './app/core/ux.service';

bootstrapApplication(AppComponent, {
  providers: [
    { provide: RouteReuseStrategy, useClass: IonicRouteStrategy },
    provideIonicAngular({
      // Core Services
      ApiService,
      ConsciousnessService,
      UXService,
      ...PLATFORM_PROVIDERS
    }),
    provideRouter(routes),
    provideHttpClient(),
    provideAnimations(),
    provideServiceWorker('ngsw-worker.js', {
      enabled: environment.production,
      registrationStrategy: 'registerWhenStable:30000',
    }),
  ],
}).catch(err => {
  console.error('Error starting chAIos application:', err);
});
