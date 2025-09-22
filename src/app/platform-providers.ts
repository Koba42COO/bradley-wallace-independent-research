// This file is the default and will be replaced by platform-specific provider files at build time.
// By default, it uses the web implementation.
import { Provider } from '@angular/core';
import { ISecureStorage } from './core/interfaces/secure-storage.interface';
import { SecureStorageWeb } from './platforms/web/services/secure-storage.web.service';

export const PLATFORM_PROVIDERS: Provider[] = [
  { provide: 'ISecureStorage', useClass: SecureStorageWeb }
];
