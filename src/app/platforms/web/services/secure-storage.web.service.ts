import { Injectable } from '@angular/core';
import { ISecureStorage } from '../../core/interfaces/secure-storage.interface';

@Injectable()
export class SecureStorageWeb implements ISecureStorage {
  async get(key: string): Promise<string | null> {
    return localStorage.getItem(key);
  }

  async set(key: string, value: string): Promise<void> {
    localStorage.setItem(key, value);
  }

  async remove(key: string): Promise<void> {
    localStorage.removeItem(key);
  }
}
