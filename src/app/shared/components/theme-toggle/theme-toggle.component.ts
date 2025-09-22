import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-theme-toggle',
  templateUrl: './theme-toggle.component.html',
  styleUrls: ['./theme-toggle.component.scss'],
})
export class ThemeToggleComponent implements OnInit {
  isDarkMode: boolean = false;

  constructor() { }

  ngOnInit() {
    // Use matchMedia to check the user's preferred color scheme
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
    this.isDarkMode = prefersDark.matches;
    this.updateBodyTheme();
  }

  toggleTheme() {
    this.isDarkMode = !this.isDarkMode;
    this.updateBodyTheme();
  }

  private updateBodyTheme() {
    document.body.classList.toggle('dark', this.isDarkMode);
  }
}
