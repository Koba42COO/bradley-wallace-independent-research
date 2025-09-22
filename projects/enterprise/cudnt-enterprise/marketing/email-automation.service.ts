import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

// Email Automation Interfaces
export interface EmailCampaign {
  id: string;
  name: string;
  type: 'nurture' | 'onboarding' | 'product' | 'promotional' | 'educational';
  status: 'draft' | 'scheduled' | 'running' | 'completed' | 'paused';
  targetAudience: string[];
  consciousnessLevel: number;
  performance: {
    sent: number;
    opened: number;
    clicked: number;
    converted: number;
    openRate: number;
    clickRate: number;
    conversionRate: number;
  };
  schedule: {
    startDate: Date;
    frequency: 'once' | 'daily' | 'weekly' | 'monthly';
    endDate?: Date;
  };
  content: {
    subject: string;
    preview: string;
    template: string;
    personalization: Record<string, any>;
  };
}

export interface EmailTemplate {
  id: string;
  name: string;
  category: string;
  consciousnessOptimized: boolean;
  variables: string[];
  htmlContent: string;
  textContent: string;
  thumbnail: string;
}

export interface CampaignAnalytics {
  campaignId: string;
  totalSent: number;
  totalOpens: number;
  totalClicks: number;
  totalConversions: number;
  revenueGenerated: number;
  roi: number;
  consciousnessCorrelation: {
    level: number;
    performance: number;
    correlation: number;
  };
  timeSeries: Array<{
    date: string;
    sent: number;
    opens: number;
    clicks: number;
    conversions: number;
  }>;
}

@Injectable({
  providedIn: 'root'
})
export class EmailAutomationService {
  private readonly apiUrl = 'http://localhost:3000/api/marketing';

  // State management
  private campaignsSubject = new BehaviorSubject<EmailCampaign[]>([]);
  private templatesSubject = new BehaviorSubject<EmailTemplate[]>([]);

  public campaigns$ = this.campaignsSubject.asObservable();
  public templates$ = this.templatesSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadCampaigns();
    this.loadTemplates();
  }

  // Load email campaigns
  loadCampaigns(): void {
    this.http.get<EmailCampaign[]>(`${this.apiUrl}/email/campaigns`)
      .subscribe(campaigns => {
        this.campaignsSubject.next(campaigns);
      });
  }

  // Load email templates
  loadTemplates(): void {
    this.http.get<EmailTemplate[]>(`${this.apiUrl}/email/templates`)
      .subscribe(templates => {
        this.templatesSubject.next(templates);
      });
  }

  // Create consciousness-optimized email campaign
  createCampaign(campaignData: Partial<EmailCampaign>): Observable<EmailCampaign> {
    return this.http.post<EmailCampaign>(`${this.apiUrl}/email/campaigns`, {
      ...campaignData,
      consciousnessOptimization: true
    });
  }

  // Send test email
  sendTestEmail(campaignId: string, testEmail: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/email/test`, {
      campaignId,
      testEmail,
      consciousnessEnhanced: true
    });
  }

  // Schedule campaign
  scheduleCampaign(campaignId: string, schedule: any): Observable<EmailCampaign> {
    return this.http.put<EmailCampaign>(`${this.apiUrl}/email/campaigns/${campaignId}/schedule`, {
      schedule,
      consciousnessOptimized: true
    });
  }

  // Get campaign analytics
  getCampaignAnalytics(campaignId: string): Observable<CampaignAnalytics> {
    return this.http.get<CampaignAnalytics>(`${this.apiUrl}/email/analytics/${campaignId}`);
  }

  // Optimize campaign with Wallace Transform
  optimizeCampaign(campaignId: string): Observable<EmailCampaign> {
    return this.http.post<EmailCampaign>(`${this.apiUrl}/email/optimize/${campaignId}`, {
      algorithm: 'wallace_transform',
      phi: (1 + Math.sqrt(5)) / 2
    });
  }

  // A/B test campaigns
  createABTest(testData: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/email/ab-test`, {
      ...testData,
      consciousnessBased: true
    });
  }

  // Get consciousness-based personalization
  getPersonalizationData(userId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/email/personalization/${userId}`);
  }

  // Export campaign data
  exportCampaignData(campaignId: string, format: 'csv' | 'json'): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/email/export/${campaignId}?format=${format}`, {
      responseType: 'blob'
    });
  }
}
