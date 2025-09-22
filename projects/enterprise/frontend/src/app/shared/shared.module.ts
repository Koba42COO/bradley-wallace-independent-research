import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';
import { ThemeToggleComponent } from './components/theme-toggle/theme-toggle.component';

@NgModule({
  imports: [
    CommonModule,
    IonicModule,
    ThemeToggleComponent
  ],
  exports: [ThemeToggleComponent]
})
export class SharedModule { }
