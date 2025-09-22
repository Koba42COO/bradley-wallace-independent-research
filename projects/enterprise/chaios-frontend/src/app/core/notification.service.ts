import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { ToastController, AlertController } from '@ionic/angular';

export interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
  actions?: NotificationAction[];
  data?: any;
}

export interface NotificationAction {
  text: string;
  handler: () => void;
  style?: 'default' | 'cancel' | 'destructive';
}

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  private notificationsSubject = new BehaviorSubject<Notification[]>([]);
  public notifications$ = this.notificationsSubject.asObservable();

  private unreadCountSubject = new BehaviorSubject<number>(0);
  public unreadCount$ = this.unreadCountSubject.asObservable();

  constructor(
    private toastController: ToastController,
    private alertController: AlertController
  ) {
    this.loadNotifications();
  }

  /**
   * Show a toast notification
   */
  async showNotification(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info', duration: number = 3000): Promise<void> {
    const toast = await this.toastController.create({
      message: message,
      duration: duration,
      position: 'top',
      color: this.getToastColor(type),
      buttons: [
        {
          text: 'Dismiss',
          role: 'cancel'
        }
      ]
    });

    await toast.present();
  }

  /**
   * Add a persistent notification
   */
  addNotification(title: string, message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info', actions?: NotificationAction[]): string {
    const notification: Notification = {
      id: this.generateId(),
      title: title,
      message: message,
      type: type,
      timestamp: new Date(),
      read: false,
      actions: actions
    };

    const currentNotifications = this.notificationsSubject.value;
    const updatedNotifications = [notification, ...currentNotifications];
    
    this.notificationsSubject.next(updatedNotifications);
    this.updateUnreadCount();
    this.saveNotifications();

    // Also show as toast for immediate feedback
    this.showNotification(`${title}: ${message}`, type);

    return notification.id;
  }

  /**
   * Mark notification as read
   */
  markAsRead(id: string): void {
    const notifications = this.notificationsSubject.value;
    const notification = notifications.find(n => n.id === id);
    
    if (notification && !notification.read) {
      notification.read = true;
      this.notificationsSubject.next([...notifications]);
      this.updateUnreadCount();
      this.saveNotifications();
    }
  }

  /**
   * Mark all notifications as read
   */
  markAllAsRead(): void {
    const notifications = this.notificationsSubject.value;
    const updatedNotifications = notifications.map(n => ({ ...n, read: true }));
    
    this.notificationsSubject.next(updatedNotifications);
    this.updateUnreadCount();
    this.saveNotifications();
  }

  /**
   * Remove notification
   */
  removeNotification(id: string): void {
    const notifications = this.notificationsSubject.value;
    const updatedNotifications = notifications.filter(n => n.id !== id);
    
    this.notificationsSubject.next(updatedNotifications);
    this.updateUnreadCount();
    this.saveNotifications();
  }

  /**
   * Clear all notifications
   */
  clearAll(): void {
    this.notificationsSubject.next([]);
    this.updateUnreadCount();
    this.saveNotifications();
  }

  /**
   * Show alert dialog
   */
  async showAlert(title: string, message: string, buttons: string[] = ['OK']): Promise<string> {
    return new Promise(async (resolve) => {
      const alert = await this.alertController.create({
        header: title,
        message: message,
        buttons: buttons.map(text => ({
          text: text,
          handler: () => resolve(text)
        }))
      });

      await alert.present();
    });
  }

  /**
   * Show confirmation dialog
   */
  async showConfirmation(title: string, message: string): Promise<boolean> {
    return new Promise(async (resolve) => {
      const alert = await this.alertController.create({
        header: title,
        message: message,
        buttons: [
          {
            text: 'Cancel',
            role: 'cancel',
            handler: () => resolve(false)
          },
          {
            text: 'Confirm',
            handler: () => resolve(true)
          }
        ]
      });

      await alert.present();
    });
  }

  /**
   * Get notifications
   */
  getNotifications(): Notification[] {
    return this.notificationsSubject.value;
  }

  /**
   * Get unread count
   */
  getUnreadCount(): number {
    return this.unreadCountSubject.value;
  }

  // Specialized notifications for chAIos features

  /**
   * Consciousness processing notification
   */
  notifyConsciousnessUpdate(gain: number, correlation: number, processingTime: number): void {
    this.addNotification(
      'Consciousness Processing Complete',
      `Performance gain: ${gain.toFixed(2)}%, Correlation: ${correlation.toFixed(6)}, Time: ${processingTime.toFixed(3)}s`,
      'success'
    );
  }

  /**
   * Quantum simulation notification
   */
  notifyQuantumSimulation(qubits: number, fidelity: number, status: string): void {
    const type = status === 'completed' ? 'success' : status === 'error' ? 'error' : 'info';
    this.addNotification(
      'Quantum Simulation Update',
      `${qubits} qubits, Fidelity: ${(fidelity * 100).toFixed(2)}%, Status: ${status}`,
      type
    );
  }

  /**
   * System alert notification
   */
  notifySystemAlert(level: 'info' | 'warning' | 'error', message: string, component?: string): void {
    const title = component ? `${component} Alert` : 'System Alert';
    this.addNotification(title, message, level);
  }

  /**
   * Chat message notification
   */
  notifyChatMessage(provider: string, preview: string): void {
    this.addNotification(
      `New message from ${provider}`,
      preview.length > 50 ? preview.substring(0, 47) + '...' : preview,
      'info'
    );
  }

  /**
   * Mathematical calculation notification
   */
  notifyMathematicalResult(type: string, result: any): void {
    let message = '';
    switch (type) {
      case 'zeta':
        message = `Zeta zeros found: ${result.zeros_count}, Correlation: ${result.correlation}`;
        break;
      case 'golden_ratio':
        message = `Golden ratio calculation: φ = ${result.phi}`;
        break;
      default:
        message = `Result: ${JSON.stringify(result)}`;
    }

    this.addNotification(
      `Mathematical Calculation: ${type}`,
      message,
      'success'
    );
  }

  /**
   * Performance alert notification
   */
  notifyPerformanceAlert(metric: string, value: number, threshold: number): void {
    const type = value > threshold ? 'warning' : 'error';
    this.addNotification(
      'Performance Alert',
      `${metric}: ${value} (threshold: ${threshold})`,
      type
    );
  }

  // Private helper methods

  private getToastColor(type: string): string {
    switch (type) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'danger';
      default: return 'primary';
    }
  }

  private generateId(): string {
    return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private updateUnreadCount(): void {
    const unreadCount = this.notificationsSubject.value.filter(n => !n.read).length;
    this.unreadCountSubject.next(unreadCount);
  }

  private saveNotifications(): void {
    try {
      const notifications = this.notificationsSubject.value;
      localStorage.setItem('chaios_notifications', JSON.stringify(notifications));
    } catch (error) {
      console.warn('Failed to save notifications to localStorage:', error);
    }
  }

  private loadNotifications(): void {
    try {
      const stored = localStorage.getItem('chaios_notifications');
      if (stored) {
        const notifications: Notification[] = JSON.parse(stored);
        // Convert timestamp strings back to Date objects
        notifications.forEach(n => {
          n.timestamp = new Date(n.timestamp);
        });
        this.notificationsSubject.next(notifications);
        this.updateUnreadCount();
      }
    } catch (error) {
      console.warn('Failed to load notifications from localStorage:', error);
    }
  }
}
