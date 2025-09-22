import { Provider } from '@angular/core';
import { ISecureStorage } from '../core/interfaces/secure-storage.interface';
import { SecureStorageWeb } from '../platforms/web/services/secure-storage.web.service';

export const PLATFORM_PROVIDERS: Provider[] = [
  { provide: 'ISecureStorage', useClass: SecureStorageWeb }
];
