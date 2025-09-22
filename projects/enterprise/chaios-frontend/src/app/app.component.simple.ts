import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { IonApp, IonRouterOutlet, IonContent } from '@ionic/angular/standalone';

/**
 * Simple App Component for Debugging
 * ==================================
 * Minimal component to test if the issue is with the complex app component
 */

@Component({
  selector: 'app-root',
  template: `
    <ion-app>
      <ion-content>
        <div style="padding: 2rem; text-align: center;">
          <h1 style="color: #D4AF37; font-size: 2rem; margin-bottom: 1rem;">
            🚀 chAIos Platform
          </h1>
          <p style="color: #64748B; font-size: 1.1rem; margin-bottom: 2rem;">
            Debugging Mode - Simple Component Test
          </p>
          <div style="background: #1E293B; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <p style="color: #10B981; margin: 0;">
              ✅ Angular is working
            </p>
            <p style="color: #10B981; margin: 0;">
              ✅ Ionic is working  
            </p>
            <p style="color: #10B981; margin: 0;">
              ✅ TypeScript is working
            </p>
          </div>
          <button 
            style="
              background: linear-gradient(45deg, #D4AF37, #8A2BE2);
              color: white;
              border: none;
              padding: 12px 24px;
              border-radius: 8px;
              font-size: 1rem;
              cursor: pointer;
              font-weight: 600;
            "
            (click)="testClick()">
            Test Click
          </button>
          <div *ngIf="clicked" style="margin-top: 1rem; color: #10B981;">
            🎉 Click event working!
          </div>
        </div>
      </ion-content>
    </ion-app>
  `,
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    IonApp,
    IonRouterOutlet,
    IonContent
  ]
})
export class SimpleAppComponent {
  clicked = false;

  testClick(): void {
    this.clicked = true;
    console.log('🎯 Simple component click test successful!');
  }
}
