import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { CUDNTService, OptimizationResult } from '../../services/cudnt.service';
import { LoadingController, ToastController, AlertController } from '@ionic/angular';

@Component({
  selector: 'app-optimize',
  templateUrl: './optimize.page.html',
  styleUrls: ['./optimize.page.scss'],
})
export class OptimizePage implements OnInit {
  optimizeForm: FormGroup;
  isOptimizing = false;
  optimizationResult: OptimizationResult | null = null;
  testMatrix: number[][] = [];
  matrixSize = 32;

  constructor(
    private fb: FormBuilder,
    private cudntService: CUDNTService,
    private loadingController: LoadingController,
    private toastController: ToastController,
    private alertController: AlertController
  ) {
    this.optimizeForm = this.fb.group({
      matrixSize: [32, [Validators.required, Validators.min(4), Validators.max(1024)]],
      customMatrix: [false],
      matrixData: ['']
    });
  }

  ngOnInit() {
    this.generateTestMatrix();

    // Subscribe to processing status
    this.cudntService.isProcessing$.subscribe(isProcessing => {
      this.isOptimizing = isProcessing;
    });
  }

  generateTestMatrix() {
    this.matrixSize = this.optimizeForm.get('matrixSize')?.value || 32;
    this.testMatrix = this.cudntService.generateTestMatrix(this.matrixSize);

    // Show theoretical performance
    const theoretical = this.cudntService.calculateTheoreticalPerformance(this.matrixSize);
    console.log('Theoretical Performance:', theoretical);
  }

  async optimizeMatrix() {
    if (!this.optimizeForm.valid) {
      const toast = await this.toastController.create({
        message: 'Please check your input parameters',
        duration: 2000,
        color: 'warning'
      });
      await toast.present();
      return;
    }

    const loading = await this.loadingController.create({
      message: 'Optimizing matrix with CUDNT...',
      spinner: 'circular'
    });
    await loading.present();

    try {
      let matrix = this.testMatrix;

      if (this.optimizeForm.get('customMatrix')?.value) {
        // Parse custom matrix data
        const matrixString = this.optimizeForm.get('matrixData')?.value;
        if (matrixString) {
          matrix = JSON.parse(matrixString);
        }
      }

      this.optimizationResult = await this.cudntService.optimizeMatrix(
        matrix,
        undefined,
        'demo-user'
      ).toPromise();

      await loading.dismiss();

      const toast = await this.toastController.create({
        message: `Optimization complete! ${this.optimizationResult.result.performance.speedupFactor.toFixed(1)}x speedup achieved`,
        duration: 3000,
        color: 'success'
      });
      await toast.present();

    } catch (error) {
      await loading.dismiss();

      const toast = await this.toastController.create({
        message: 'Optimization failed. Please try again.',
        duration: 3000,
        color: 'danger'
      });
      await toast.present();
    }
  }

  async showResults() {
    if (!this.optimizationResult) return;

    const result = this.optimizationResult.result;
    const alert = await this.alertController.create({
      header: 'Optimization Results',
      subHeader: `CUDNT Performance: ${result.performance.speedupFactor.toFixed(1)}x Speedup`,
      message: `
        <p><strong>Processing Time:</strong> ${result.performance.processingTime.toFixed(4)}s</p>
        <p><strong>Complexity Reduction:</strong> ${result.performance.complexityReduction}</p>
        <p><strong>Consciousness Level:</strong> ${result.performance.consciousnessLevel}/12</p>
        <p><strong>Improvement:</strong> ${result.performance.improvementPercent.toFixed(1)}%</p>
        <p><strong>Algorithm:</strong> ${result.metadata.algorithm}</p>
        <p><strong>Golden Ratio (φ):</strong> ${result.metadata.phi.toFixed(6)}</p>
      `,
      buttons: ['OK']
    });
    await alert.present();
  }

  getPerformanceColor(value: number): string {
    if (value >= 200) return 'success';
    if (value >= 100) return 'warning';
    return 'danger';
  }
}
