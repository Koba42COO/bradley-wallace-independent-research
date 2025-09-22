import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, timer } from 'rxjs';
import { map } from 'rxjs/operators';
import { ToastController, LoadingController, AlertController, ModalController } from '@ionic/angular';
import { Router } from '@angular/router';

/**
 * chAIos UX Service
 * =================
 * Professional UX management following tangtalk standards
 * Centralized user experience orchestration with consistent feedback
 */

export interface UXState {
  isLoading: boolean;
  loadingMessage?: string;
  error?: {
    type: string;
    message: string;
    details?: any;
  };
  toast?: {
    message: string;
    type: 'success' | 'warning' | 'error' | 'info';
    duration?: number;
  };
}

export interface LoadingOptions {
  message?: string;
  duration?: number;
  backdrop?: boolean;
  showSpinner?: boolean;
  spinnerName?: string;
}

export interface ToastOptions {
  message: string;
  type?: 'success' | 'warning' | 'error' | 'info';
  duration?: number;
  position?: 'top' | 'middle' | 'bottom';
  showCloseButton?: boolean;
  closeButtonText?: string;
}

export interface AlertOptions {
  header?: string;
  subHeader?: string;
  message: string;
  buttons?: any[];
  inputs?: any[];
  cssClass?: string;
}

export interface ConfirmOptions {
  header: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  confirmColor?: string;
  cancelColor?: string;
}

@Injectable({
  providedIn: 'root'
})
export class UXService {
  private uxState$ = new BehaviorSubject<UXState>({
    isLoading: false
  });

  private loadingElement: HTMLIonLoadingElement | null = null;
  private toastQueue: ToastOptions[] = [];
  private isProcessingToast = false;

  constructor(
    private toastController: ToastController,
    private loadingController: LoadingController,
    private alertController: AlertController,
    private modalController: ModalController,
    private router: Router
  ) {
    this.initializeUXService();
  }

  /**
   * Initialize UX Service
   * ====================
   */
  private initializeUXService(): void {
    console.log('🎨 Initializing chAIos UX Service');
    
    // Set up global error handling
    this.setupGlobalErrorHandling();
    
    // Set up navigation feedback
    this.setupNavigationFeedback();
    
    console.log('✅ UX Service initialized');
  }

  /**
   * Observable State
   * ================
   */
  get state$(): Observable<UXState> {
    return this.uxState$.asObservable();
  }

  get isLoading$(): Observable<boolean> {
    return this.uxState$.pipe(
      map(state => state.isLoading)
    );
  }

  /**
   * Loading States
   * ==============
   */
  async showLoading(options: LoadingOptions = {}): Promise<void> {
    if (this.loadingElement) {
      await this.hideLoading();
    }

    this.updateState({ 
      isLoading: true, 
      loadingMessage: options.message 
    });

    this.loadingElement = await this.loadingController.create({
      message: options.message || 'Loading...',
      duration: options.duration,
      showBackdrop: options.backdrop !== false,
      spinner: options.spinnerName as any || 'crescent',
      cssClass: 'chaios-loading'
    });

    await this.loadingElement.present();

    // Auto-hide after duration if specified
    if (options.duration) {
      timer(options.duration).subscribe(() => {
        this.hideLoading();
      });
    }
  }

  async hideLoading(): Promise<void> {
    this.updateState({ isLoading: false, loadingMessage: undefined });

    if (this.loadingElement) {
      await this.loadingElement.dismiss();
      this.loadingElement = null;
    }
  }

  /**
   * Toast Notifications
   * ===================
   */
  async showToast(options: ToastOptions): Promise<void> {
    this.toastQueue.push(options);
    
    if (!this.isProcessingToast) {
      await this.processToastQueue();
    }
  }

  private async processToastQueue(): Promise<void> {
    if (this.toastQueue.length === 0) {
      this.isProcessingToast = false;
      return;
    }

    this.isProcessingToast = true;
    const options = this.toastQueue.shift()!;

    const toast = await this.toastController.create({
      message: options.message,
      duration: options.duration || 3000,
      position: options.position || 'bottom',
      color: this.getToastColor(options.type || 'info'),
      buttons: options.showCloseButton ? [
        {
          text: options.closeButtonText || 'Close',
          role: 'cancel'
        }
      ] : undefined,
      cssClass: `chaios-toast toast-${options.type || 'info'}`
    });

    await toast.present();
    
    // Update state
    this.updateState({
      toast: {
        message: options.message,
        type: options.type || 'info',
        duration: options.duration
      }
    });

    // Wait for toast to dismiss, then process next
    toast.onDidDismiss().then(() => {
      this.updateState({ toast: undefined });
      setTimeout(() => this.processToastQueue(), 100);
    });
  }

  private getToastColor(type: string): string {
    const colorMap: Record<string, string> = {
      success: 'success',
      error: 'danger',
      warning: 'warning',
      info: 'primary'
    };
    return colorMap[type] || 'primary';
  }

  /**
   * Quick Toast Methods
   * ===================
   */
  async showSuccess(message: string, duration: number = 3000): Promise<void> {
    await this.showToast({
      message,
      type: 'success',
      duration
    });
  }

  async showError(message: string, duration: number = 4000): Promise<void> {
    await this.showToast({
      message,
      type: 'error',
      duration,
      showCloseButton: true
    });
  }

  async showWarning(message: string, duration: number = 3500): Promise<void> {
    await this.showToast({
      message,
      type: 'warning',
      duration
    });
  }

  async showInfo(message: string, duration: number = 3000): Promise<void> {
    await this.showToast({
      message,
      type: 'info',
      duration
    });
  }

  /**
   * Alert Dialogs
   * =============
   */
  async showAlert(options: AlertOptions): Promise<any> {
    const alert = await this.alertController.create({
      header: options.header,
      subHeader: options.subHeader,
      message: options.message,
      buttons: options.buttons || ['OK'],
      inputs: options.inputs,
      cssClass: `chaios-alert ${options.cssClass || ''}`
    });

    await alert.present();
    return alert.onDidDismiss();
  }

  async showConfirm(options: ConfirmOptions): Promise<boolean> {
    const alert = await this.alertController.create({
      header: options.header,
      message: options.message,
      buttons: [
        {
          text: options.cancelText || 'Cancel',
          role: 'cancel',
          cssClass: 'secondary',
          handler: () => false
        },
        {
          text: options.confirmText || 'Confirm',
          cssClass: 'primary',
          handler: () => true
        }
      ],
      cssClass: 'chaios-confirm'
    });

    await alert.present();
    const result = await alert.onDidDismiss();
    return result.data === true;
  }

  /**
   * Error Handling
   * ==============
   */
  handleError(error: any, context?: string): void {
    console.error(`UX Error ${context ? `(${context})` : ''}:`, error);

    const errorMessage = this.extractErrorMessage(error);
    const errorType = this.determineErrorType(error);

    // Update state
    this.updateState({
      error: {
        type: errorType,
        message: errorMessage,
        details: error
      }
    });

    // Show user-friendly error message
    this.showError(errorMessage);
  }

  private extractErrorMessage(error: any): string {
    if (typeof error === 'string') {
      return error;
    }

    if (error?.error?.message) {
      return error.error.message;
    }

    if (error?.message) {
      return error.message;
    }

    if (error?.status) {
      const statusMessages: Record<number, string> = {
        400: 'Invalid request. Please check your input.',
        401: 'Authentication required. Please log in.',
        403: 'Access denied. You don\'t have permission.',
        404: 'The requested resource was not found.',
        429: 'Too many requests. Please try again later.',
        500: 'Server error. Please try again later.',
        502: 'Service temporarily unavailable.',
        503: 'Service maintenance in progress.'
      };

      return statusMessages[error.status] || `Server error (${error.status})`;
    }

    return 'An unexpected error occurred. Please try again.';
  }

  private determineErrorType(error: any): string {
    if (error?.status) {
      if (error.status >= 400 && error.status < 500) {
        return 'client';
      } else if (error.status >= 500) {
        return 'server';
      }
    }

    if (error?.name === 'NetworkError' || error?.code === 'NETWORK_ERROR') {
      return 'network';
    }

    return 'generic';
  }

  /**
   * Navigation Feedback
   * ===================
   */
  private setupNavigationFeedback(): void {
    // Show loading during navigation
    this.router.events.subscribe((event) => {
      // Implementation depends on router events
      // This is a placeholder for navigation feedback
    });
  }

  /**
   * Global Error Handling
   * =====================
   */
  private setupGlobalErrorHandling(): void {
    // Set up global error handlers
    window.addEventListener('unhandledrejection', (event) => {
      this.handleError(event.reason, 'Unhandled Promise Rejection');
    });

    window.addEventListener('error', (event) => {
      this.handleError(event.error, 'Global Error');
    });
  }

  /**
   * Haptic Feedback
   * ===============
   */
  async hapticFeedback(type: 'light' | 'medium' | 'heavy' = 'light'): Promise<void> {
    try {
      // Use Web Vibration API if available
      if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
        const duration = type === 'light' ? 10 : type === 'medium' ? 20 : 50;
        navigator.vibrate(duration);
      }
      // Note: Capacitor Haptics can be added later when Capacitor is properly installed
    } catch (error) {
      // Haptic feedback not available - fail silently
      console.debug('Haptic feedback not available:', error);
    }
  }

  /**
   * Utility Methods
   * ===============
   */
  private updateState(update: Partial<UXState>): void {
    const currentState = this.uxState$.value;
    this.uxState$.next({ ...currentState, ...update });
  }

  clearError(): void {
    this.updateState({ error: undefined });
  }

  clearToast(): void {
    this.updateState({ toast: undefined });
  }

  /**
   * Cleanup
   * =======
   */
  async cleanup(): Promise<void> {
    await this.hideLoading();
    this.toastQueue = [];
    this.uxState$.complete();
  }
}
