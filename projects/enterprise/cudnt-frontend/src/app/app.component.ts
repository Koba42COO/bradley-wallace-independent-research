import { Component } from '@angular/core';
import { Platform } from '@ionic/angular';
import { StatusBar } from '@capacitor/status-bar';
import { Capacitor } from '@capacitor/core';

@Component({
  selector: 'app-root',
  templateUrl: 'app.component.html',
  styleUrls: ['app.component.scss']
})
export class AppComponent {
  constructor(private platform: Platform) {
    this.initializeApp();
  }

  initializeApp() {
    this.platform.ready().then(() => {
      if (Capacitor.isPluginAvailable('StatusBar')) {
        StatusBar.setBackgroundColor({ color: '#1e3a8a' });
      }
    });
  }
}
