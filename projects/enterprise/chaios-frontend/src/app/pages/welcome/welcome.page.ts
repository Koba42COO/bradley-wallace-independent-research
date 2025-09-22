import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { 
  IonContent, 
  IonHeader, 
  IonTitle, 
  IonToolbar,
  IonButton,
  IonIcon,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
  IonGrid,
  IonRow,
  IonCol,
  IonText
} from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import { 
  rocketOutline, 
  sparklesOutline, 
  nuclearOutline,
  chatbubbleOutline,
  analyticsOutline,
  lockClosedOutline
} from 'ionicons/icons';

@Component({
  selector: 'app-welcome',
  template: `
    <ion-header [translucent]="true">
      <ion-toolbar>
        <ion-title>
          <div class="welcome-title">
            <ion-icon name="sparkles-outline"></ion-icon>
            chAIos Platform
          </div>
        </ion-title>
      </ion-toolbar>
    </ion-header>

    <ion-content [fullscreen]="true" class="welcome-content">
      <div class="welcome-container">
        
        <!-- Hero Section -->
        <ion-card class="hero-card">
          <ion-card-header>
            <ion-card-title class="hero-title">
              <ion-icon name="rocket-outline" class="hero-icon"></ion-icon>
              Welcome to chAIos
            </ion-card-title>
          </ion-card-header>
          <ion-card-content>
            <ion-text class="hero-subtitle">
              <h2>Chiral Harmonic Aligned Intelligence Optimisation System</h2>
              <p>Experience the future of consciousness-enhanced AI processing with our revolutionary platform.</p>
            </ion-text>
          </ion-card-content>
        </ion-card>

        <!-- Features Grid -->
        <ion-grid class="features-grid">
          <ion-row>
            <ion-col size="12" size-md="6">
              <ion-card class="feature-card">
                <ion-card-header>
                  <ion-card-title>
                    <ion-icon name="chatbubble-outline"></ion-icon>
                    AI Chat Interface
                  </ion-card-title>
                </ion-card-header>
                <ion-card-content>
                  Multi-provider AI chat with consciousness enhancement
                </ion-card-content>
              </ion-card>
            </ion-col>
            
            <ion-col size="12" size-md="6">
              <ion-card class="feature-card">
                <ion-card-header>
                  <ion-card-title>
                    <ion-icon name="nuclear-outline"></ion-icon>
                    Quantum Processing
                  </ion-card-title>
                </ion-card-header>
                <ion-card-content>
                  Advanced quantum consciousness simulation and processing
                </ion-card-content>
              </ion-card>
            </ion-col>
            
            <ion-col size="12" size-md="6">
              <ion-card class="feature-card">
                <ion-card-header>
                  <ion-card-title>
                    <ion-icon name="analytics-outline"></ion-icon>
                    Consciousness Analytics
                  </ion-card-title>
                </ion-card-header>
                <ion-card-content>
                  Real-time visualization of consciousness metrics and patterns
                </ion-card-content>
              </ion-card>
            </ion-col>
            
            <ion-col size="12" size-md="6">
              <ion-card class="feature-card">
                <ion-card-header>
                  <ion-card-title>
                    <ion-icon name="sparkles-outline"></ion-icon>
                    Golden Ratio Design
                  </ion-card-title>
                </ion-card-header>
                <ion-card-content>
                  Beautiful chiral harmonic interface based on mathematical perfection
                </ion-card-content>
              </ion-card>
            </ion-col>
          </ion-row>
        </ion-grid>

        <!-- Action Buttons -->
        <div class="action-section">
          <ion-button 
            expand="block" 
            size="large" 
            [routerLink]="['/auth/login']"
            class="primary-action">
            <ion-icon name="lock-closed-outline" slot="start"></ion-icon>
            Sign In to chAIos
          </ion-button>
          
          <ion-button 
            expand="block" 
            size="large" 
            fill="outline"
            [routerLink]="['/auth/register']"
            class="secondary-action">
            <ion-icon name="sparkles-outline" slot="start"></ion-icon>
            Create New Account
          </ion-button>
        </div>

        <!-- Status Section -->
        <ion-card class="status-card">
          <ion-card-content>
            <ion-text class="status-text">
              <h3>🚀 Platform Status: OPERATIONAL</h3>
              <p>✅ Backend API: Connected</p>
              <p>✅ Consciousness Engine: Active</p>
              <p>✅ Quantum Processor: Ready</p>
              <p>✅ 25 Curated Tools: Loaded</p>
            </ion-text>
          </ion-card-content>
        </ion-card>

      </div>
    </ion-content>
  `,
  styleUrls: ['./welcome.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    IonContent,
    IonHeader,
    IonTitle,
    IonToolbar,
    IonButton,
    IonIcon,
    IonCard,
    IonCardHeader,
    IonCardTitle,
    IonCardContent,
    IonGrid,
    IonRow,
    IonCol,
    IonText
  ]
})
export class WelcomePage {
  
  constructor() {
    this.initializeIcons();
  }

  private initializeIcons() {
    addIcons({
      'rocket-outline': rocketOutline,
      'sparkles-outline': sparklesOutline,
      'nuclear-outline': nuclearOutline,
      'chatbubble-outline': chatbubbleOutline,
      'analytics-outline': analyticsOutline,
      'lock-closed-outline': lockClosedOutline
    });
  }
}
