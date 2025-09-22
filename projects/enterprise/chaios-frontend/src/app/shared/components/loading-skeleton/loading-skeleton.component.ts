import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

/**
 * chAIos Loading Skeleton Component
 * ================================
 * Professional loading skeleton following tangtalk UX standards
 * Provides visual feedback during content loading
 */

export type SkeletonType = 'text' | 'card' | 'avatar' | 'image' | 'button' | 'list';

@Component({
  selector: 'chaios-loading-skeleton',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div 
      class="chaios-skeleton"
      [class]="'chaios-skeleton--' + type"
      [style.width]="width"
      [style.height]="height"
      [attr.aria-label]="ariaLabel"
      role="status">
      
      <!-- Text Skeleton -->
      <div *ngIf="type === 'text'" class="skeleton-lines">
        <div 
          *ngFor="let line of getLines()" 
          class="skeleton-line"
          [style.width]="line.width">
        </div>
      </div>
      
      <!-- Card Skeleton -->
      <div *ngIf="type === 'card'" class="skeleton-card">
        <div class="skeleton-card-header"></div>
        <div class="skeleton-card-content">
          <div class="skeleton-line" style="width: 100%"></div>
          <div class="skeleton-line" style="width: 80%"></div>
          <div class="skeleton-line" style="width: 60%"></div>
        </div>
      </div>
      
      <!-- Avatar Skeleton -->
      <div *ngIf="type === 'avatar'" class="skeleton-avatar"></div>
      
      <!-- Image Skeleton -->
      <div *ngIf="type === 'image'" class="skeleton-image">
        <div class="skeleton-image-placeholder">
          <svg width="24" height="24" viewBox="0 0 24 24">
            <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
          </svg>
        </div>
      </div>
      
      <!-- Button Skeleton -->
      <div *ngIf="type === 'button'" class="skeleton-button"></div>
      
      <!-- List Skeleton -->
      <div *ngIf="type === 'list'" class="skeleton-list">
        <div *ngFor="let item of getListItems()" class="skeleton-list-item">
          <div class="skeleton-avatar skeleton-list-avatar"></div>
          <div class="skeleton-list-content">
            <div class="skeleton-line" style="width: 70%"></div>
            <div class="skeleton-line" style="width: 40%"></div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .chaios-skeleton {
      display: block;
      background: linear-gradient(90deg, 
        #f0f0f0 25%, 
        #e0e0e0 50%, 
        #f0f0f0 75%
      );
      background-size: 200% 100%;
      animation: skeleton-shimmer 1.5s infinite;
      border-radius: 4px;
      overflow: hidden;
    }

    @keyframes skeleton-shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    /* Text Skeleton */
    .chaios-skeleton--text {
      background: transparent;
    }

    .skeleton-lines {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .skeleton-line {
      height: 16px;
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: skeleton-shimmer 1.5s infinite;
      border-radius: 4px;
    }

    /* Card Skeleton */
    .chaios-skeleton--card {
      background: white;
      border: 1px solid #e0e0e0;
      padding: 16px;
    }

    .skeleton-card-header {
      height: 20px;
      width: 40%;
      margin-bottom: 16px;
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: skeleton-shimmer 1.5s infinite;
      border-radius: 4px;
    }

    .skeleton-card-content {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    /* Avatar Skeleton */
    .chaios-skeleton--avatar {
      width: 40px;
      height: 40px;
      border-radius: 50%;
    }

    .skeleton-avatar {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: skeleton-shimmer 1.5s infinite;
    }

    /* Image Skeleton */
    .chaios-skeleton--image {
      background: #f5f5f5;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .skeleton-image-placeholder svg {
      fill: #ccc;
    }

    /* Button Skeleton */
    .chaios-skeleton--button {
      width: 120px;
      height: 36px;
      border-radius: 18px;
    }

    /* List Skeleton */
    .chaios-skeleton--list {
      background: transparent;
    }

    .skeleton-list-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 0;
      border-bottom: 1px solid #f0f0f0;
    }

    .skeleton-list-avatar {
      width: 40px;
      height: 40px;
      flex-shrink: 0;
    }

    .skeleton-list-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      .chaios-skeleton {
        background: linear-gradient(90deg, #333 25%, #444 50%, #333 75%);
        background-size: 200% 100%;
      }

      .skeleton-line {
        background: linear-gradient(90deg, #333 25%, #444 50%, #333 75%);
        background-size: 200% 100%;
      }

      .skeleton-card-header {
        background: linear-gradient(90deg, #333 25%, #444 50%, #333 75%);
        background-size: 200% 100%;
      }

      .skeleton-avatar {
        background: linear-gradient(90deg, #333 25%, #444 50%, #333 75%);
        background-size: 200% 100%;
      }

      .chaios-skeleton--image {
        background: #2a2a2a;
      }

      .skeleton-image-placeholder svg {
        fill: #666;
      }

      .chaios-skeleton--card {
        background: #1a1a1a;
        border-color: #333;
      }
    }

    /* Consciousness theme */
    .chaios-skeleton.consciousness-theme {
      background: linear-gradient(90deg, 
        rgba(212, 175, 55, 0.1) 25%, 
        rgba(46, 139, 87, 0.1) 50%, 
        rgba(212, 175, 55, 0.1) 75%
      );
      background-size: 200% 100%;
    }
  `]
})
export class LoadingSkeletonComponent {
  @Input() type: SkeletonType = 'text';
  @Input() width: string = '100%';
  @Input() height: string = 'auto';
  @Input() lines: number = 3;
  @Input() listItems: number = 3;
  @Input() ariaLabel: string = 'Loading content';

  getLines(): { width: string }[] {
    const lines = [];
    for (let i = 0; i < this.lines; i++) {
      const widths = ['100%', '80%', '60%', '90%', '70%'];
      lines.push({ width: widths[i % widths.length] });
    }
    return lines;
  }

  getListItems(): number[] {
    return Array.from({ length: this.listItems }, (_, i) => i);
  }
}
