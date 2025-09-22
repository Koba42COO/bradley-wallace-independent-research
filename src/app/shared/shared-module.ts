import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule } from '@ionic/angular';
import { ThemeToggleComponent } from './components/theme-toggle/theme-toggle.component';

@NgModule({
  declarations: [ThemeToggleComponent],
  imports: [
    CommonModule,
    IonicModule
  ],
  exports: [ThemeToggleComponent]
})
export class SharedModule { }
