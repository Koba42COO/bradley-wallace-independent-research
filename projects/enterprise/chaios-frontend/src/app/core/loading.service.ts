import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { LoadingController } from '@ionic/angular';

@Injectable({
  providedIn: 'root'
})
export class LoadingService {
  private loadingSubject = new BehaviorSubject<boolean>(false);
  public loading$ = this.loadingSubject.asObservable();
  
  private loadingCount = 0;
  private currentLoader: HTMLIonLoadingElement | null = null;

  constructor(private loadingController: LoadingController) {}

  async show(message: string = 'Processing...', duration?: number): Promise<void> {
    this.loadingCount++;
    
    if (this.loadingCount === 1) {
      this.loadingSubject.next(true);
      
      this.currentLoader = await this.loadingController.create({
        message: message,
        duration: duration,
        spinner: 'crescent',
        cssClass: 'consciousness-loading',
        translucent: true
      });
      
      await this.currentLoader.present();
    }
  }

  async hide(): Promise<void> {
    this.loadingCount = Math.max(0, this.loadingCount - 1);
    
    if (this.loadingCount === 0) {
      this.loadingSubject.next(false);
      
      if (this.currentLoader) {
        await this.currentLoader.dismiss();
        this.currentLoader = null;
      }
    }
  }

  async forceHide(): Promise<void> {
    this.loadingCount = 0;
    this.loadingSubject.next(false);
    
    if (this.currentLoader) {
      await this.currentLoader.dismiss();
      this.currentLoader = null;
    }
  }

  isLoading(): boolean {
    return this.loadingSubject.value;
  }
}

