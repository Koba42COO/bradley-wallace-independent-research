import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface SpiralPoint {
  x: number;
  y: number;
  phi: number;
}

@Component({
  selector: 'app-golden-spiral',
  template: `
    <div class="spiral-container">
      <canvas 
        #spiralCanvas 
        class="spiral-canvas"
        width="400" 
        height="400">
      </canvas>
      
      <div class="spiral-controls" *ngIf="spiralData && spiralData.length > 0">
        <div class="phi-display">
          <span class="phi-symbol">φ</span>
          <span class="phi-value">{{ getCurrentPhi() }}</span>
        </div>
        
        <div class="spiral-stats">
          <div class="stat">
            <span class="label">Points:</span>
            <span class="value">{{ spiralData.length }}</span>
          </div>
          <div class="stat">
            <span class="label">Radius:</span>
            <span class="value">{{ getCurrentRadius().toFixed(2) }}</span>
          </div>
        </div>
      </div>
    </div>
  `,
  styleUrls: ['./golden-spiral.component.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class GoldenSpiralComponent implements OnChanges, AfterViewInit {
  @ViewChild('spiralCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() spiralData: SpiralPoint[] = [];
  @Input() phiEvolution: number[] = [];

  private ctx: CanvasRenderingContext2D | null = null;
  private animationFrame: number | null = null;
  private currentFrame = 0;
  private phi = (1 + Math.sqrt(5)) / 2;

  ngAfterViewInit() {
    this.initializeCanvas();
    this.startAnimation();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['spiralData'] || changes['phiEvolution']) {
      this.updateVisualization();
    }
  }

  private initializeCanvas() {
    const canvas = this.canvasRef.nativeElement;
    this.ctx = canvas.getContext('2d');
    
    if (this.ctx) {
      // Set up canvas for high DPI displays
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      
      this.ctx.scale(dpr, dpr);
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
    }
  }

  private startAnimation() {
    const animate = () => {
      this.drawSpiral();
      this.currentFrame++;
      this.animationFrame = requestAnimationFrame(animate);
    };
    
    animate();
  }

  private updateVisualization() {
    if (this.ctx && this.spiralData.length > 0) {
      this.drawSpiral();
    }
  }

  private drawSpiral() {
    if (!this.ctx || !this.spiralData.length) return;

    const canvas = this.canvasRef.nativeElement;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = Math.min(canvas.width, canvas.height) / 8;

    // Clear canvas with dark background
    this.ctx.fillStyle = '#0F172A';
    this.ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw background grid
    this.drawGrid(centerX, centerY, scale);

    // Draw golden rectangles
    this.drawGoldenRectangles(centerX, centerY, scale);

    // Draw the spiral
    this.drawSpiralPath(centerX, centerY, scale);

    // Draw animated points
    this.drawAnimatedPoints(centerX, centerY, scale);

    // Draw phi ratio indicators
    this.drawPhiIndicators(centerX, centerY, scale);
  }

  private drawGrid(centerX: number, centerY: number, scale: number) {
    if (!this.ctx) return;

    this.ctx.strokeStyle = 'rgba(212, 175, 55, 0.1)';
    this.ctx.lineWidth = 0.5;

    // Horizontal and vertical lines
    const gridSize = 50;
    for (let i = -10; i <= 10; i++) {
      // Vertical lines
      this.ctx.beginPath();
      this.ctx.moveTo(centerX + i * gridSize, 0);
      this.ctx.lineTo(centerX + i * gridSize, this.canvasRef.nativeElement.height);
      this.ctx.stroke();

      // Horizontal lines
      this.ctx.beginPath();
      this.ctx.moveTo(0, centerY + i * gridSize);
      this.ctx.lineTo(this.canvasRef.nativeElement.width, centerY + i * gridSize);
      this.ctx.stroke();
    }
  }

  private drawGoldenRectangles(centerX: number, centerY: number, scale: number) {
    if (!this.ctx) return;

    this.ctx.strokeStyle = 'rgba(46, 139, 87, 0.3)';
    this.ctx.lineWidth = 1;

    // Draw nested golden rectangles
    let size = scale * 0.5;
    for (let i = 0; i < 8; i++) {
      this.ctx.strokeRect(
        centerX - size / 2,
        centerY - size / (this.phi * 2),
        size,
        size / this.phi
      );
      size *= this.phi;
    }
  }

  private drawSpiralPath(centerX: number, centerY: number, scale: number) {
    if (!this.ctx || this.spiralData.length < 2) return;

    // Create gradient for the spiral
    const gradient = this.ctx.createLinearGradient(0, 0, centerX * 2, centerY * 2);
    gradient.addColorStop(0, '#D4AF37');
    gradient.addColorStop(0.5, '#8A2BE2');
    gradient.addColorStop(1, '#2E8B57');

    this.ctx.strokeStyle = gradient;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();

    // Draw the spiral path
    for (let i = 0; i < this.spiralData.length; i++) {
      const point = this.spiralData[i];
      const x = centerX + point.x * scale * 0.1;
      const y = centerY + point.y * scale * 0.1;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }

    this.ctx.stroke();
  }

  private drawAnimatedPoints(centerX: number, centerY: number, scale: number) {
    if (!this.ctx) return;

    const animationSpeed = 0.02;
    const numActivePoints = 50;
    const startIndex = Math.floor(this.currentFrame * animationSpeed) % this.spiralData.length;

    for (let i = 0; i < numActivePoints && startIndex + i < this.spiralData.length; i++) {
      const point = this.spiralData[startIndex + i];
      const x = centerX + point.x * scale * 0.1;
      const y = centerY + point.y * scale * 0.1;

      // Fade effect based on age
      const alpha = 1 - (i / numActivePoints);
      
      this.ctx.fillStyle = `rgba(212, 175, 55, ${alpha})`;
      this.ctx.beginPath();
      this.ctx.arc(x, y, 3 * alpha, 0, Math.PI * 2);
      this.ctx.fill();

      // Draw phi value at some points
      if (i % 10 === 0) {
        this.ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
        this.ctx.font = '10px monospace';
        this.ctx.fillText(point.phi.toFixed(2), x + 5, y - 5);
      }
    }
  }

  private drawPhiIndicators(centerX: number, centerY: number, scale: number) {
    if (!this.ctx) return;

    // Draw phi ratio arcs
    this.ctx.strokeStyle = 'rgba(138, 43, 226, 0.5)';
    this.ctx.lineWidth = 1;

    const baseRadius = scale * 0.2;
    for (let i = 1; i <= 5; i++) {
      const radius = baseRadius * Math.pow(this.phi, i);
      this.ctx.beginPath();
      this.ctx.arc(centerX, centerY, radius, 0, Math.PI / 2);
      this.ctx.stroke();
    }

    // Draw center phi symbol
    this.ctx.fillStyle = '#D4AF37';
    this.ctx.font = 'bold 16px serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('φ', centerX, centerY + 5);
  }

  getCurrentPhi(): string {
    if (this.phiEvolution && this.phiEvolution.length > 0) {
      const index = Math.floor(this.currentFrame * 0.01) % this.phiEvolution.length;
      return this.phiEvolution[index].toFixed(6);
    }
    return this.phi.toFixed(6);
  }

  getCurrentRadius(): number {
    if (this.spiralData && this.spiralData.length > 0) {
      const currentIndex = Math.floor(this.currentFrame * 0.02) % this.spiralData.length;
      return this.spiralData[currentIndex].phi;
    }
    return 0;
  }

  ngOnDestroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
  }
}

