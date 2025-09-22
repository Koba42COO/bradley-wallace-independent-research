import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

// ROI Calculator Interfaces
export interface ROICalculation {
  customerId: string;
  useCase: string;
  currentCosts: {
    hardware: number;
    software: number;
    personnel: number;
    energy: number;
    maintenance: number;
  };
  cudntCosts: {
    subscription: number;
    implementation: number;
    training: number;
    support: number;
  };
  performanceMetrics: {
    speedupFactor: number;
    energyReduction: number;
    hardwareReduction: number;
    productivityGain: number;
  };
  roi: {
    paybackPeriod: number;
    threeYearSavings: number;
    fiveYearSavings: number;
    roiPercentage: number;
    npv: number;
  };
  consciousnessOptimization: {
    level: number;
    additionalSavings: number;
    timeline: string;
  };
}

export interface UseCaseTemplate {
  id: string;
  name: string;
  industry: string;
  description: string;
  averageSpeedup: number;
  costReduction: number;
  implementationTime: string;
  prerequisites: string[];
}

@Injectable({
  providedIn: 'root'
})
export class ROICalculatorService {
  private readonly apiUrl = 'http://localhost:3000/api/sales';

  // Predefined use case templates with consciousness optimization
  public useCaseTemplates: UseCaseTemplate[] = [
    {
      id: 'scientific-computing',
      name: 'Scientific Computing',
      industry: 'Research',
      description: 'Molecular dynamics, quantum chemistry, climate modeling',
      averageSpeedup: 267,
      costReduction: 85,
      implementationTime: '2-4 weeks',
      prerequisites: ['Existing HPC cluster', 'Scientific software stack']
    },
    {
      id: 'financial-modeling',
      name: 'Financial Risk Modeling',
      industry: 'Finance',
      description: 'Monte Carlo simulations, portfolio optimization, derivatives pricing',
      averageSpeedup: 189,
      costReduction: 78,
      implementationTime: '1-3 weeks',
      prerequisites: ['Financial modeling software', 'Risk management systems']
    },
    {
      id: 'ai-training',
      name: 'AI/ML Training',
      industry: 'Technology',
      description: 'Neural network training, deep learning optimization',
      averageSpeedup: 156,
      costReduction: 72,
      implementationTime: '3-6 weeks',
      prerequisites: ['ML frameworks', 'Training pipelines', 'Data pipelines']
    },
    {
      id: 'video-rendering',
      name: 'Video Rendering',
      industry: 'Media',
      description: 'CGI rendering, visual effects, animation',
      averageSpeedup: 234,
      costReduction: 81,
      implementationTime: '1-2 weeks',
      prerequisites: ['Rendering software', 'Asset pipelines']
    },
    {
      id: 'drug-discovery',
      name: 'Drug Discovery',
      industry: 'Pharmaceutical',
      description: 'Molecular docking, virtual screening, protein folding',
      averageSpeedup: 203,
      costReduction: 79,
      implementationTime: '4-8 weeks',
      prerequisites: ['Molecular modeling software', 'Chemical databases']
    }
  ];

  constructor(private http: HttpClient) {}

  // Calculate ROI for specific customer scenario
  calculateROI(customerData: any): Observable<ROICalculation> {
    return this.http.post<ROICalculation>(`${this.apiUrl}/roi/calculate`, {
      ...customerData,
      consciousnessOptimization: true
    });
  }

  // Get ROI templates by industry
  getROITemplates(industry?: string): Observable<UseCaseTemplate[]> {
    const url = industry
      ? `${this.apiUrl}/roi/templates?industry=${industry}`
      : `${this.apiUrl}/roi/templates`;
    return this.http.get<UseCaseTemplate[]>(url);
  }

  // Generate detailed ROI report
  generateROIReport(calculationId: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/roi/report/${calculationId}`, {
      responseType: 'blob'
    });
  }

  // Get industry benchmarks
  getIndustryBenchmarks(industry: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/roi/benchmarks/${industry}`);
  }

  // Calculate consciousness-enhanced ROI
  calculateConsciousnessROI(baseROI: ROICalculation, consciousnessLevel: number): Observable<ROICalculation> {
    return this.http.post<ROICalculation>(`${this.apiUrl}/roi/consciousness-enhance`, {
      baseROI,
      consciousnessLevel
    });
  }

  // Get competitive analysis
  getCompetitiveAnalysis(useCase: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/roi/competitive/${useCase}`);
  }
}
