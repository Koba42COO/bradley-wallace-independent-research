import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

// Lead Scoring Interfaces
export interface LeadScore {
  leadId: string;
  email: string;
  company: string;
  score: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  consciousnessLevel: number;
  intentLevel: number;
  budgetRange: string;
  timeline: string;
  painPoints: string[];
  recommendations: string[];
  lastActivity: Date;
  source: string;
}

export interface LeadInsights {
  totalLeads: number;
  qualifiedLeads: number;
  conversionRate: number;
  averageScore: number;
  topPainPoints: string[];
  consciousnessDistribution: {
    high: number;
    medium: number;
    low: number;
  };
  industryBreakdown: Record<string, number>;
}

@Injectable({
  providedIn: 'root'
})
export class LeadScoringService {
  private readonly apiUrl = 'http://localhost:3000/api/sales';

  // State management
  private leadsSubject = new BehaviorSubject<LeadScore[]>([]);
  private insightsSubject = new BehaviorSubject<LeadInsights | null>(null);

  public leads$ = this.leadsSubject.asObservable();
  public insights$ = this.insightsSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadLeads();
    this.loadInsights();
  }

  // Load leads with consciousness-based scoring
  loadLeads(): void {
    this.http.get<LeadScore[]>(`${this.apiUrl}/leads`)
      .subscribe(leads => {
        this.leadsSubject.next(leads);
      });
  }

  // Load lead insights and analytics
  loadInsights(): void {
    this.http.get<LeadInsights>(`${this.apiUrl}/leads/insights`)
      .subscribe(insights => {
        this.insightsSubject.next(insights);
      });
  }

  // Calculate consciousness-based lead score
  calculateLeadScore(leadData: any): Observable<LeadScore> {
    return this.http.post<LeadScore>(`${this.apiUrl}/leads/score`, {
      ...leadData,
      consciousnessOptimization: true
    });
  }

  // Update lead score based on activity
  updateLeadScore(leadId: string, activity: any): Observable<LeadScore> {
    return this.http.put<LeadScore>(`${this.apiUrl}/leads/${leadId}/score`, {
      activity,
      consciousnessEnhanced: true
    });
  }

  // Get personalized recommendations for lead
  getLeadRecommendations(leadId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/leads/${leadId}/recommendations`);
  }

  // Export leads for sales team
  exportLeads(format: 'csv' | 'excel' | 'pdf'): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/leads/export?format=${format}`, {
      responseType: 'blob'
    });
  }

  // Get lead grade distribution
  getLeadGradeDistribution(): Observable<Record<string, number>> {
    return this.http.get<Record<string, number>>(`${this.apiUrl}/leads/grades`);
  }
}
