import { Injectable } from '@angular/core';
import { ISecureStorage } from '../../../core/interfaces/secure-storage.interface';
import { Storage } from '@ionic/storage-angular';

@Injectable()
export class SecureStorageIonic implements ISecureStorage {
  private _storage: Storage | null = null;

  constructor(private storage: Storage) {
    this.init();
  }

  async init() {
    // Create the Ionic storage instance
    const storage = await this.storage.create();
    this._storage = storage;
  }

  public async get(key: string): Promise<string | null> {
    return await this._storage?.get(key) || null;
  }

  public async set(key: string, value: string): Promise<void> {
    await this._storage?.set(key, value);
  }

  public async remove(key: string): Promise<void> {
    await this._storage?.remove(key);
  }
}
