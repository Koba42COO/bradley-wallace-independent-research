import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, interval, map, switchMap, catchError, of } from 'rxjs';

import { ApiService, ConsciousnessRequest, ConsciousnessResponse } from '../../../core/api.service';
import { WebSocketService } from '../../../core/websocket.service';
import { NotificationService } from '../../../core/notification.service';

export interface ConsciousnessMetrics {
  performanceGain: number;
  // ... existing code ...
}
