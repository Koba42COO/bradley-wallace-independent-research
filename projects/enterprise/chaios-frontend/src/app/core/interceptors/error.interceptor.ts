import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { Router } from '@angular/router';

import { NotificationService } from '../services/notification.service';
import { AuthService } from '../services/auth.service';

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const notificationService = inject(NotificationService);
  const authService = inject(AuthService);
  const router = inject(Router);

  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      console.error('🔥 HTTP Error:', error);

      // Handle different error status codes
      switch (error.status) {
        case 401:
          // Unauthorized - token expired or invalid
          console.log('🔐 Unauthorized - logging out user');
          authService.logout();
          notificationService.showNotification('Session expired. Please login again.', 'warning');
          router.navigate(['/auth/login']);
          break;

        case 403:
          // Forbidden - user doesn't have permission
          notificationService.showNotification('Access denied. Insufficient permissions.', 'error');
          break;

        case 404:
          // Not found
          notificationService.showNotification('Resource not found.', 'warning');
          break;

        case 500:
          // Internal server error
          notificationService.showNotification('Server error. Please try again later.', 'error');
          break;

        case 0:
          // Network error
          notificationService.showNotification('Network error. Please check your connection.', 'error');
          break;

        default:
          // Generic error handling
          const errorMessage = error.error?.message || error.message || 'An unexpected error occurred';
          notificationService.showNotification(errorMessage, 'error');
      }

      return throwError(() => error);
    })
  );
};

