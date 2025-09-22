import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';

/**
 * chAIos Error State Component
 * ============================
 * Professional error handling UX following tangtalk standards
 * Provides consistent error feedback with recovery actions
 */

export type ErrorType = 'network' | 'server' | 'validation' | 'permission' | 'notFound' | 'generic';

export interface ErrorStateConfig {
  type: ErrorType;
  title: string;
  message: string;
  icon?: string;
  showRetry?: boolean;
  showSupport?: boolean;
  retryText?: string;
  supportText?: string;
}

@Component({
  selector: 'chaios-error-state',
  standalone: true,
  imports: [CommonModule, IonicModule],
  template: `
    <div class="chaios-error-state" [class]="'error-' + config.type">
      <div class="error-container">
        <!-- Error Icon -->
        <div class="error-icon">
          <ion-icon 
            [name]="config.icon || getDefaultIcon()" 
            size="large"
            [class]="'icon-' + config.type">
          </ion-icon>
        </div>

        <!-- Error Content -->
        <div class="error-content">
          <h2 class="error-title">{{ config.title }}</h2>
          <p class="error-message">{{ config.message }}</p>
          
          <!-- Additional Details -->
          <div class="error-details" *ngIf="details">
            <ion-button 
              fill="clear" 
              size="small" 
              (click)="showDetails = !showDetails"
              class="details-toggle">
              <ion-icon 
                [name]="showDetails ? 'chevron-up' : 'chevron-down'" 
                slot="start">
              </ion-icon>
              {{ showDetails ? 'Hide' : 'Show' }} Details
            </ion-button>
            
            <div class="details-content" *ngIf="showDetails">
              <pre>{{ details }}</pre>
            </div>
          </div>
        </div>

        <!-- Error Actions -->
        <div class="error-actions">
          <ion-button 
            *ngIf="config.showRetry !== false"
            expand="block" 
            color="primary"
            (click)="onRetry()"
            class="retry-button">
            <ion-icon name="refresh-outline" slot="start"></ion-icon>
            {{ config.retryText || 'Try Again' }}
          </ion-button>

          <ion-button 
            *ngIf="config.showSupport"
            expand="block" 
            fill="outline"
            color="medium"
            (click)="onSupport()"
            class="support-button">
            <ion-icon name="help-circle-outline" slot="start"></ion-icon>
            {{ config.supportText || 'Get Help' }}
          </ion-button>

          <ion-button 
            *ngIf="showGoBack"
            expand="block" 
            fill="clear"
            color="medium"
            (click)="onGoBack()"
            class="back-button">
            <ion-icon name="arrow-back-outline" slot="start"></ion-icon>
            Go Back
          </ion-button>
        </div>

        <!-- Error Code -->
        <div class="error-code" *ngIf="errorCode">
          <small>Error Code: {{ errorCode }}</small>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .chaios-error-state {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 300px;
      padding: 2rem;
      text-align: center;
    }

    .error-container {
      max-width: 400px;
      width: 100%;
    }

    .error-icon {
      margin-bottom: 1.5rem;
      opacity: 0.8;
    }

    .error-icon ion-icon {
      font-size: 4rem;
      transition: all 0.3s ease;
    }

    .error-title {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--ion-color-dark);
    }

    .error-message {
      font-size: 1rem;
      line-height: 1.5;
      color: var(--ion-color-medium);
      margin-bottom: 1.5rem;
    }

    .error-actions {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      margin-top: 1.5rem;
    }

    .error-details {
      margin: 1rem 0;
      text-align: left;
    }

    .details-content {
      margin-top: 0.5rem;
      padding: 1rem;
      background: var(--ion-color-light);
      border-radius: 8px;
      border-left: 4px solid var(--ion-color-warning);
    }

    .details-content pre {
      font-size: 0.8rem;
      color: var(--ion-color-dark);
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
    }

    .error-code {
      margin-top: 1rem;
      opacity: 0.6;
    }

    .error-code small {
      font-size: 0.75rem;
      font-family: monospace;
    }

    /* Error Type Specific Styles */
    .error-network .error-icon ion-icon {
      color: var(--ion-color-warning);
    }

    .error-server .error-icon ion-icon {
      color: var(--ion-color-danger);
    }

    .error-validation .error-icon ion-icon {
      color: var(--ion-color-warning);
    }

    .error-permission .error-icon ion-icon {
      color: var(--ion-color-tertiary);
    }

    .error-notFound .error-icon ion-icon {
      color: var(--ion-color-medium);
    }

    .error-generic .error-icon ion-icon {
      color: var(--ion-color-medium);
    }

    /* Animations */
    .chaios-error-state {
      animation: errorFadeIn 0.5s ease-out;
    }

    @keyframes errorFadeIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .error-icon ion-icon {
      animation: errorIconBounce 0.6s ease-out 0.2s both;
    }

    @keyframes errorIconBounce {
      0% {
        transform: scale(0.3);
        opacity: 0;
      }
      50% {
        transform: scale(1.1);
      }
      100% {
        transform: scale(1);
        opacity: 0.8;
      }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .chaios-error-state {
        padding: 1rem;
        min-height: 250px;
      }

      .error-container {
        max-width: 100%;
      }

      .error-title {
        font-size: 1.25rem;
      }

      .error-icon ion-icon {
        font-size: 3rem;
      }
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
      .details-content {
        background: var(--ion-color-dark);
        border-left-color: var(--ion-color-warning);
      }

      .details-content pre {
        color: var(--ion-color-light);
      }
    }

    /* Consciousness Theme */
    .consciousness-theme .error-icon ion-icon {
      color: #D4AF37;
    }

    .consciousness-theme .details-content {
      background: linear-gradient(135deg, 
        rgba(212, 175, 55, 0.1) 0%, 
        rgba(46, 139, 87, 0.05) 100%
      );
      border-left-color: #D4AF37;
    }
  `]
})
export class ErrorStateComponent {
  @Input() config!: ErrorStateConfig;
  @Input() details?: string;
  @Input() errorCode?: string;
  @Input() showGoBack: boolean = false;

  @Output() retry = new EventEmitter<void>();
  @Output() support = new EventEmitter<void>();
  @Output() goBack = new EventEmitter<void>();

  showDetails = false;

  getDefaultIcon(): string {
    const iconMap: Record<ErrorType, string> = {
      network: 'wifi-outline',
      server: 'server-outline',
      validation: 'warning-outline',
      permission: 'lock-closed-outline',
      notFound: 'search-outline',
      generic: 'alert-circle-outline'
    };

    return iconMap[this.config.type] || 'alert-circle-outline';
  }

  onRetry(): void {
    this.retry.emit();
  }

  onSupport(): void {
    this.support.emit();
  }

  onGoBack(): void {
    this.goBack.emit();
  }

  // Static factory methods for common error states
  static networkError(): ErrorStateConfig {
    return {
      type: 'network',
      title: 'Connection Problem',
      message: 'Please check your internet connection and try again.',
      icon: 'wifi-outline',
      showRetry: true
    };
  }

  static serverError(): ErrorStateConfig {
    return {
      type: 'server',
      title: 'Server Error',
      message: 'Something went wrong on our end. Please try again in a moment.',
      icon: 'server-outline',
      showRetry: true,
      showSupport: true
    };
  }

  static notFoundError(): ErrorStateConfig {
    return {
      type: 'notFound',
      title: 'Page Not Found',
      message: 'The page you\'re looking for doesn\'t exist or has been moved.',
      icon: 'search-outline',
      showRetry: false
    };
  }

  static permissionError(): ErrorStateConfig {
    return {
      type: 'permission',
      title: 'Access Denied',
      message: 'You don\'t have permission to access this resource.',
      icon: 'lock-closed-outline',
      showRetry: false,
      showSupport: true
    };
  }

  static validationError(message: string): ErrorStateConfig {
    return {
      type: 'validation',
      title: 'Invalid Input',
      message: message || 'Please check your input and try again.',
      icon: 'warning-outline',
      showRetry: false
    };
  }
}
