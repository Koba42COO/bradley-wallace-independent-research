import { Provider } from '@angular/core';
import { ISecureStorage } from './core/interfaces/secure-storage.interface';
import { SecureStorageIonic } from './platforms/ionic/services/secure-storage.ionic.service';

export const PLATFORM_PROVIDERS: Provider[] = [
  { provide: 'ISecureStorage', useClass: SecureStorageIonic }
];
