import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, Router, NavigationStart, NavigationEnd } from '@angular/router';
import { IonApp, IonRouterOutlet, IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonIcon, IonLabel, IonMenuButton, IonButtons, IonButton, IonBadge } from '@ionic/angular/standalone';
import { addIcons } from 'ionicons';
import {
  homeOutline,
  chatbubbleOutline,
  analyticsOutline,
  nuclearOutline,
  calculatorOutline,
  settingsOutline,
  personOutline,
  logOutOutline,
  menuOutline,
  notificationsOutline,
  helpCircleOutline,
  informationCircleOutline,
  wifiOutline,
  checkmarkCircleOutline,
  warningOutline,
  closeCircleOutline
} from 'ionicons/icons';
import { Subscription } from 'rxjs';
import { UXService } from './core/ux.service';

// Removed auth dependencies for simplified access

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss'],
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    IonApp,
    IonRouterOutlet,
    IonMenu,
    IonHeader,
    IonToolbar,
    IonTitle,
    IonContent,
    IonList,
    IonItem,
    IonIcon,
    IonLabel,
    IonMenuButton,
    IonButtons,
    IonButton,
    IonBadge,
    RouterLink
  ]
})
export class AppComponent implements OnInit, OnDestroy {
  public appPages = [
    { 
      title: 'Dashboard', 
      url: '/dashboard', 
      icon: 'home-outline',
      badge: null,
      badgeColor: 'primary'
    },
    { 
      title: 'AI Chat', 
      url: '/ai-chat', 
      icon: 'chatbubble-outline',
      badge: null,
      badgeColor: 'success'
    },
    { 
      title: 'Consciousness', 
      url: '/consciousness', 
      icon: 'analytics-outline',
      badge: 'New',
      badgeColor: 'tertiary'
    },
    { 
      title: 'Quantum Lab', 
      url: '/quantum', 
      icon: 'nuclear-outline',
      badge: null,
      badgeColor: 'warning'
    },
    { 
      title: 'Mathematics', 
      url: '/mathematics', 
      icon: 'calculator-outline',
      badge: null,
      badgeColor: 'secondary'
    },
    { 
      title: 'Analytics', 
      url: '/analytics', 
      icon: 'analytics-outline',
      badge: null,
      badgeColor: 'primary'
    },
    { 
      title: 'Settings', 
      url: '/settings', 
      icon: 'settings-outline',
      badge: null,
      badgeColor: 'medium'
    }
  ];

  public isAuthenticated = true; // Always authenticated for simplified access
  public currentUser: any = { name: 'chAIos User', email: 'user@chaios.ai' };
  public unreadNotifications = 0;
  public connectionStatus = 'connected';
  public isNavigating = false;
  public selectedMenuItem: string | null = null;

  private subscriptions: Subscription[] = [];

  constructor(
    private router: Router,
    private uxService: UXService
  ) {
    this.initializeApp();
  }

  ngOnInit() {
    this.setupNavigationTracking();
    this.setupUXFeedback();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  private initializeApp() {
    // Add all required icons
    addIcons({
      'home-outline': homeOutline,
      'chatbubble-outline': chatbubbleOutline,
      'analytics-outline': analyticsOutline,
      'nuclear-outline': nuclearOutline,
      'calculator-outline': calculatorOutline,
      'settings-outline': settingsOutline,
      'person-outline': personOutline,
      'log-out-outline': logOutOutline,
      'menu-outline': menuOutline,
      'notifications-outline': notificationsOutline,
      'help-circle-outline': helpCircleOutline,
      'information-circle-outline': informationCircleOutline,
      'wifi-outline': wifiOutline,
      'checkmark-circle-outline': checkmarkCircleOutline,
      'warning-outline': warningOutline,
      'close-circle-outline': closeCircleOutline
    });

    // Initialize global chAIos object
    if (typeof window !== 'undefined') {
      (window as any).chAIos = {
        ...((window as any).chAIos || {}),
        app: {
          version: '1.0.0',
          name: 'chAIos',
          fullName: 'Chiral Harmonic Aligned Intelligence Optimisation System',
          initialized: true,
          startTime: new Date().toISOString()
        }
      };
    }

    console.log('🚀 chAIos Application Initialized');
    console.log('📐 Mathematical Constants:', {
      phi: 1.618034,
      sigma: 0.381966,
      pi: 3.14159265359
    });
  }

  /**
   * Navigation Tracking
   * ===================
   */
  private setupNavigationTracking(): void {
    const navigationSub = this.router.events.subscribe(event => {
      if (event instanceof NavigationStart) {
        this.isNavigating = true;
        this.selectedMenuItem = event.url;
      } else if (event instanceof NavigationEnd) {
        this.isNavigating = false;
        this.selectedMenuItem = null;
      }
    });
    
    this.subscriptions.push(navigationSub);
  }

  /**
   * UX Feedback Setup
   * =================
   */
  private setupUXFeedback(): void {
    // Listen to UX service state for global feedback
    const uxSub = this.uxService.state$.subscribe(state => {
      // Handle global UX states if needed
    });
    
    this.subscriptions.push(uxSub);
  }

  /**
   * Menu Interactions
   * =================
   */
  onMenuItemClick(page: any): void {
    this.selectedMenuItem = page.url;
    
    // Haptic feedback on mobile
    this.uxService.hapticFeedback('light');
    
    // Clear any existing badges when clicked
    if (page.badge) {
      setTimeout(() => {
        page.badge = null;
      }, 1000);
    }
  }

  trackByUrl(index: number, item: any): string {
    return item.url;
  }

  /**
   * Connection Status
   * =================
   */
  getConnectionStatusColor(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'success';
      case 'connecting':
        return 'warning';
      case 'error':
        return 'danger';
      default:
        return 'medium';
    }
  }

  getConnectionStatusText(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'Online';
      case 'connecting':
        return 'Connecting...';
      case 'error':
        return 'Offline';
      default:
        return 'Unknown';
    }
  }

  getConnectionIcon(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'checkmark-circle-outline';
      case 'connecting':
        return 'warning-outline';
      case 'error':
        return 'close-circle-outline';
      default:
        return 'wifi-outline';
    }
  }

  async showConnectionDetails(): Promise<void> {
    const message = this.getConnectionDetailMessage();
    await this.uxService.showAlert({
      header: 'Connection Status',
      message: message,
      buttons: ['OK'],
      cssClass: 'connection-details-alert'
    });
  }

  private getConnectionDetailMessage(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'You are connected to the chAIos consciousness network. All features are available.';
      case 'connecting':
        return 'Establishing connection to the consciousness network. Some features may be limited.';
      case 'error':
        return 'Connection to the consciousness network failed. Please check your internet connection.';
      default:
        return 'Connection status unknown. Please refresh the application.';
    }
  }

  /**
   * User Actions
   * ============
   */
  async logout(): Promise<void> {
    const confirmed = await this.uxService.showConfirm({
      header: 'Logout',
      message: 'Are you sure you want to logout from chAIos?',
      confirmText: 'Logout',
      cancelText: 'Cancel'
    });

    if (confirmed) {
      await this.uxService.showLoading({
        message: 'Logging out...',
        duration: 1500
      });
      
      // Simulate logout process
      setTimeout(() => {
        this.uxService.showSuccess('Successfully logged out');
        window.location.reload();
      }, 1500);
    }
  }
}