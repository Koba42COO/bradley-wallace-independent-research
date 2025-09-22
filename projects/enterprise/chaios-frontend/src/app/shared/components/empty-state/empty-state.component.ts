import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';

/**
 * chAIos Empty State Component
 * ============================
 * Professional empty state UX following tangtalk standards
 * Provides engaging empty states with clear calls to action
 */

export interface EmptyStateConfig {
  icon: string;
  title: string;
  message: string;
  actionText?: string;
  secondaryActionText?: string;
  illustration?: string;
  showAction?: boolean;
  showSecondaryAction?: boolean;
}

@Component({
  selector: 'chaios-empty-state',
  standalone: true,
  imports: [CommonModule, IonicModule],
  template: `
    <div class="chaios-empty-state">
      <div class="empty-container">
        <!-- Illustration or Icon -->
        <div class="empty-visual">
          <div class="empty-icon" *ngIf="!config.illustration">
            <ion-icon [name]="config.icon" size="large"></ion-icon>
          </div>
          
          <div class="empty-illustration" *ngIf="config.illustration">
            <img [src]="config.illustration" [alt]="config.title" />
          </div>
        </div>

        <!-- Content -->
        <div class="empty-content">
          <h2 class="empty-title">{{ config.title }}</h2>
          <p class="empty-message">{{ config.message }}</p>
        </div>

        <!-- Actions -->
        <div class="empty-actions" *ngIf="config.showAction !== false">
          <ion-button 
            *ngIf="config.actionText"
            expand="block" 
            color="primary"
            (click)="onAction()"
            class="primary-action">
            {{ config.actionText }}
          </ion-button>

          <ion-button 
            *ngIf="config.secondaryActionText && config.showSecondaryAction"
            expand="block" 
            fill="outline"
            color="medium"
            (click)="onSecondaryAction()"
            class="secondary-action">
            {{ config.secondaryActionText }}
          </ion-button>
        </div>

        <!-- Additional Content Slot -->
        <div class="empty-extra" *ngIf="showExtraContent">
          <ng-content></ng-content>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .chaios-empty-state {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 300px;
      padding: 2rem;
      text-align: center;
    }

    .empty-container {
      max-width: 400px;
      width: 100%;
    }

    .empty-visual {
      margin-bottom: 2rem;
    }

    .empty-icon {
      opacity: 0.6;
      margin-bottom: 0.5rem;
    }

    .empty-icon ion-icon {
      font-size: 5rem;
      color: var(--ion-color-medium);
      transition: all 0.3s ease;
    }

    .empty-illustration {
      max-width: 200px;
      margin: 0 auto;
    }

    .empty-illustration img {
      width: 100%;
      height: auto;
      opacity: 0.8;
    }

    .empty-content {
      margin-bottom: 2rem;
    }

    .empty-title {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      color: var(--ion-color-dark);
    }

    .empty-message {
      font-size: 1rem;
      line-height: 1.6;
      color: var(--ion-color-medium);
      margin-bottom: 0;
    }

    .empty-actions {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .empty-extra {
      margin-top: 1.5rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--ion-color-light);
    }

    /* Animations */
    .chaios-empty-state {
      animation: emptyFadeIn 0.6s ease-out;
    }

    @keyframes emptyFadeIn {
      from {
        opacity: 0;
        transform: translateY(30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .empty-icon ion-icon {
      animation: emptyIconFloat 3s ease-in-out infinite;
    }

    @keyframes emptyIconFloat {
      0%, 100% {
        transform: translateY(0px);
      }
      50% {
        transform: translateY(-10px);
      }
    }

    .empty-illustration {
      animation: emptyIllustrationScale 0.8s ease-out 0.2s both;
    }

    @keyframes emptyIllustrationScale {
      from {
        transform: scale(0.8);
        opacity: 0;
      }
      to {
        transform: scale(1);
        opacity: 0.8;
      }
    }

    /* Hover Effects */
    .empty-icon:hover ion-icon {
      transform: scale(1.1);
      color: var(--ion-color-primary);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .chaios-empty-state {
        padding: 1.5rem;
        min-height: 250px;
      }

      .empty-container {
        max-width: 100%;
      }

      .empty-title {
        font-size: 1.25rem;
      }

      .empty-icon ion-icon {
        font-size: 4rem;
      }

      .empty-illustration {
        max-width: 150px;
      }
    }

    /* Theme Variations */
    .consciousness-theme .empty-icon ion-icon {
      color: #D4AF37;
    }

    .consciousness-theme .empty-icon:hover ion-icon {
      color: #2E8B57;
    }

    /* Dark Mode */
    @media (prefers-color-scheme: dark) {
      .empty-extra {
        border-top-color: var(--ion-color-dark);
      }
    }

    /* Specific Empty State Styles */
    .empty-search .empty-icon ion-icon {
      color: var(--ion-color-tertiary);
    }

    .empty-data .empty-icon ion-icon {
      color: var(--ion-color-warning);
    }

    .empty-connection .empty-icon ion-icon {
      color: var(--ion-color-danger);
    }

    .empty-welcome .empty-icon ion-icon {
      color: var(--ion-color-success);
    }
  `]
})
export class EmptyStateComponent {
  @Input() config!: EmptyStateConfig;
  @Input() showExtraContent: boolean = false;

  @Output() action = new EventEmitter<void>();
  @Output() secondaryAction = new EventEmitter<void>();

  onAction(): void {
    this.action.emit();
  }

  onSecondaryAction(): void {
    this.secondaryAction.emit();
  }

  // Static factory methods for common empty states
  static noData(): EmptyStateConfig {
    return {
      icon: 'document-outline',
      title: 'No Data Available',
      message: 'There\'s nothing to show here yet. Try refreshing or check back later.',
      actionText: 'Refresh',
      showAction: true
    };
  }

  static noSearchResults(query?: string): EmptyStateConfig {
    return {
      icon: 'search-outline',
      title: 'No Results Found',
      message: query 
        ? `We couldn't find anything matching "${query}". Try adjusting your search terms.`
        : 'No results match your search criteria. Try different keywords.',
      actionText: 'Clear Search',
      showAction: true
    };
  }

  static noConnection(): EmptyStateConfig {
    return {
      icon: 'wifi-outline',
      title: 'No Internet Connection',
      message: 'Please check your connection and try again.',
      actionText: 'Retry',
      showAction: true
    };
  }

  static welcomeState(): EmptyStateConfig {
    return {
      icon: 'rocket-outline',
      title: 'Welcome to chAIos!',
      message: 'Get started by exploring our consciousness-driven AI tools and mathematical visualizations.',
      actionText: 'Explore Features',
      secondaryActionText: 'View Tutorial',
      showAction: true,
      showSecondaryAction: true
    };
  }

  static emptyConversation(): EmptyStateConfig {
    return {
      icon: 'chatbubble-outline',
      title: 'Start a Conversation',
      message: 'Begin your journey with AI-powered consciousness exploration. Ask anything!',
      actionText: 'Send First Message',
      showAction: true
    };
  }

  static emptyNotifications(): EmptyStateConfig {
    return {
      icon: 'notifications-outline',
      title: 'All Caught Up!',
      message: 'You have no new notifications. We\'ll let you know when something important happens.',
      showAction: false
    };
  }

  static emptyFavorites(): EmptyStateConfig {
    return {
      icon: 'heart-outline',
      title: 'No Favorites Yet',
      message: 'Items you mark as favorites will appear here for quick access.',
      actionText: 'Browse Content',
      showAction: true
    };
  }

  static maintenanceMode(): EmptyStateConfig {
    return {
      icon: 'construct-outline',
      title: 'Under Maintenance',
      message: 'We\'re making improvements to serve you better. Please check back soon.',
      showAction: false
    };
  }
}
