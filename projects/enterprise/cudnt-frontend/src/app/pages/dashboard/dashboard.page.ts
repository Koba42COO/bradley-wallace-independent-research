import { Component, OnInit, OnDestroy } from '@angular/core';
import { CUDNTService, DashboardStats, SystemStatus } from '../../services/cudnt.service';
import { LoadingController, ToastController, AlertController } from '@ionic/angular';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.page.html',
  styleUrls: ['./dashboard.page.scss'],
})
export class DashboardPage implements OnInit, OnDestroy {
  dashboardData: DashboardStats | null = null;
  systemStatus: SystemStatus | null = null;
  isLoading = false;
  userId = 'demo-user'; // In production, get from auth service

  private subscriptions: Subscription[] = [];

  constructor(
    private cudntService: CUDNTService,
    private loadingController: LoadingController,
    private toastController: ToastController,
    private alertController: AlertController
  ) {}

  ngOnInit() {
    this.loadDashboard();
    this.subscribeToRealTimeUpdates();
  }

  ngOnDestroy() {
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  async loadDashboard() {
    const loading = await this.loadingController.create({
      message: 'Loading CUDNT Dashboard...',
      spinner: 'circular'
    });
    await loading.present();

    try {
      // Load dashboard data
      this.dashboardData = await this.cudntService.getDashboardData(this.userId).toPromise();

      // Load system status
      const statusSub = this.cudntService.systemStatus$.subscribe(status => {
        this.systemStatus = status;
      });
      this.subscriptions.push(statusSub);

      await loading.dismiss();
    } catch (error) {
      await loading.dismiss();
      this.showError('Failed to load dashboard data');
    }
  }

  private subscribeToRealTimeUpdates() {
    const realtimeSub = this.cudntService.realtimeUpdates$.subscribe(update => {
      if (update.type === 'status_update') {
        // Update real-time metrics
        if (this.systemStatus) {
          this.systemStatus.activeOptimizations = update.data.activeOptimizations;
          this.systemStatus.performance.consciousnessLevel = update.data.consciousnessLevel;
        }
      }
    });
    this.subscriptions.push(realtimeSub);
  }

  async refreshDashboard(event?: any) {
    try {
      this.dashboardData = await this.cudntService.getDashboardData(this.userId).toPromise();
      this.cudntService.loadSystemStatus();

      if (event) {
        event.target.complete();
      }

      const toast = await this.toastController.create({
        message: 'Dashboard refreshed successfully',
        duration: 2000,
        color: 'success'
      });
      await toast.present();
    } catch (error) {
      if (event) {
        event.target.complete();
      }
      this.showError('Failed to refresh dashboard');
    }
  }

  async showSystemDetails() {
    const alert = await this.alertController.create({
      header: 'CUDNT System Details',
      subHeader: 'Consciousness Mathematics Framework',
      message: `
        <p><strong>Architecture:</strong> ${this.systemStatus?.system.architecture}</p>
        <p><strong>Complexity Reduction:</strong> ${this.systemStatus?.system.complexityReduction}</p>
        <p><strong>K-Loop Production:</strong> ${this.systemStatus?.performance.kLoopProduction}</p>
        <p><strong>Golden Ratio (φ):</strong> ${(1 + Math.sqrt(5)) / 2}</p>
        <p><strong>Consciousness Ratio:</strong> ${79/21}</p>
      `,
      buttons: ['OK']
    });
    await alert.present();
  }

  private async showError(message: string) {
    const toast = await this.toastController.create({
      message,
      duration: 3000,
      color: 'danger'
    });
    await toast.present();
  }

  getPerformanceColor(value: number): string {
    if (value >= 200) return 'success';
    if (value >= 100) return 'warning';
    return 'danger';
  }

  getConsciousnessLevelColor(level: number): string {
    if (level >= 9) return 'success';
    if (level >= 6) return 'warning';
    return 'danger';
  }
}
