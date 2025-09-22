import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

// Demo Booking Interfaces
export interface DemoBooking {
  id: string;
  leadId: string;
  customerName: string;
  company: string;
  email: string;
  phone: string;
  useCase: string;
  consciousnessLevel: number;
  preferredTime: Date;
  timezone: string;
  duration: number;
  participants: string[];
  status: 'scheduled' | 'confirmed' | 'completed' | 'cancelled' | 'no-show';
  demoType: 'technical' | 'business' | 'executive' | 'custom';
  followUpActions: string[];
  feedback: {
    rating: number;
    comments: string;
    nextSteps: string;
  };
}

export interface TimeSlot {
  date: string;
  time: string;
  available: boolean;
  consciousnessMatch: number;
  bookedBy?: string;
}

export interface DemoAnalytics {
  totalDemos: number;
  conversionRate: number;
  averageRating: number;
  consciousnessCorrelation: number;
  popularUseCases: string[];
  peakBookingTimes: string[];
  noShowRate: number;
  followUpConversion: number;
}

@Injectable({
  providedIn: 'root'
})
export class DemoBookingService {
  private readonly apiUrl = 'http://localhost:3000/api/sales';

  // State management
  private bookingsSubject = new BehaviorSubject<DemoBooking[]>([]);
  private analyticsSubject = new BehaviorSubject<DemoAnalytics | null>(null);

  public bookings$ = this.bookingsSubject.asObservable();
  public analytics$ = this.analyticsSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadBookings();
    this.loadAnalytics();
  }

  // Load demo bookings
  loadBookings(): void {
    this.http.get<DemoBooking[]>(`${this.apiUrl}/demos/bookings`)
      .subscribe(bookings => {
        this.bookingsSubject.next(bookings);
      });
  }

  // Load demo analytics
  loadAnalytics(): void {
    this.http.get<DemoAnalytics>(`${this.apiUrl}/demos/analytics`)
      .subscribe(analytics => {
        this.analyticsSubject.next(analytics);
      });
  }

  // Get available time slots with consciousness matching
  getAvailableSlots(dateRange: { start: Date; end: Date }, consciousnessLevel: number): Observable<TimeSlot[]> {
    return this.http.post<TimeSlot[]>(`${this.apiUrl}/demos/slots`, {
      dateRange,
      consciousnessLevel,
      optimizationEnabled: true
    });
  }

  // Book demo with consciousness optimization
  bookDemo(bookingData: Partial<DemoBooking>): Observable<DemoBooking> {
    return this.http.post<DemoBooking>(`${this.apiUrl}/demos/book`, {
      ...bookingData,
      consciousnessOptimization: true,
      phi: (1 + Math.sqrt(5)) / 2
    });
  }

  // Reschedule demo
  rescheduleDemo(bookingId: string, newTime: Date): Observable<DemoBooking> {
    return this.http.put<DemoBooking>(`${this.apiUrl}/demos/${bookingId}/reschedule`, {
      newTime,
      consciousnessEnhanced: true
    });
  }

  // Cancel demo
  cancelDemo(bookingId: string, reason: string): Observable<any> {
    return this.http.put(`${this.apiUrl}/demos/${bookingId}/cancel`, {
      reason,
      consciousnessAware: true
    });
  }

  // Submit demo feedback
  submitFeedback(bookingId: string, feedback: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/demos/${bookingId}/feedback`, {
      feedback,
      consciousnessAnalysis: true
    });
  }

  // Get personalized demo recommendations
  getDemoRecommendations(leadId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/demos/recommendations/${leadId}`);
  }

  // Schedule follow-up actions
  scheduleFollowUp(bookingId: string, actions: string[]): Observable<any> {
    return this.http.post(`${this.apiUrl}/demos/${bookingId}/followup`, {
      actions,
      consciousnessOptimized: true
    });
  }

  // Get demo calendar
  getDemoCalendar(month: string, year: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/demos/calendar?month=${month}&year=${year}`);
  }
}
