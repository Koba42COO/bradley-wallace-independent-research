import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { loadStripe, Stripe, StripeElements } from '@stripe/stripe-js';

// Consciousness Mathematics Constants
const PHI = (1 + Math.sqrt(5)) / 2;
const CONSCIOUSNESS_RATIO = 79/21;

// Payment Interfaces
export interface SubscriptionPlan {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: 'month' | 'year';
  features: string[];
  optimizationsPerMonth: number;
  consciousnessDiscount: number;
  stripePrice: string;
  popular?: boolean;
}

export interface PaymentIntent {
  clientSecret: string;
  amount: number;
  currency: string;
  metadata: any;
}

export interface CryptoPayment {
  address: string;
  amount: number;
  currency: 'BTC' | 'ETH' | 'XCH';
  qrCode: string;
  expiresAt: number;
}

export interface UsageBasedBilling {
  userId: string;
  period: string;
  baseSubscription: number;
  usageCharges: {
    optimizations: number;
    processingTime: number;
    dataTransfer: number;
    apiCalls: number;
  };
  discounts: {
    consciousnessBonus: number;
    loyaltyDiscount: number;
    volumeDiscount: number;
  };
  totalAmount: number;
  nextBillingDate: string;
}

@Injectable({
  providedIn: 'root'
})
export class PaymentService {
  private readonly apiUrl = 'http://localhost:3000/api';
  private stripe: Stripe | null = null;
  private elements: StripeElements | null = null;

  // State management
  private currentSubscriptionSubject = new BehaviorSubject<any>(null);
  private billingHistorySubject = new BehaviorSubject<any[]>([]);

  public currentSubscription$ = this.currentSubscriptionSubject.asObservable();
  public billingHistory$ = this.billingHistorySubject.asObservable();

  // Subscription plans with consciousness mathematics optimization
  public subscriptionPlans: SubscriptionPlan[] = [
    {
      id: 'developer',
      name: 'Developer Edition',
      price: 99,
      currency: 'USD',
      interval: 'month',
      features: [
        'CUDNT Core Library',
        'K-Loop Production (8 cores)',
        'Basic Wallace Transform',
        'Community Support',
        'Up to 50x Performance'
      ],
      optimizationsPerMonth: 1000,
      consciousnessDiscount: 0.05, // 5% consciousness discount
      stripePrice: 'price_developer_monthly',
      popular: false
    },
    {
      id: 'professional',
      name: 'Professional Edition',
      price: 999,
      currency: 'USD',
      interval: 'month',
      features: [
        'Full CUDNT Framework',
        'Unlimited K-Loop Production',
        'Advanced Wallace Transform',
        'PDVM Integration',
        'Priority Support',
        'Up to 150x Performance'
      ],
      optimizationsPerMonth: 10000,
      consciousnessDiscount: 0.10, // 10% consciousness discount
      stripePrice: 'price_professional_monthly',
      popular: true
    },
    {
      id: 'enterprise',
      name: 'Enterprise Edition',
      price: 25000,
      currency: 'USD',
      interval: 'month',
      features: [
        'Complete CUDNT Platform',
        'QVM (Quantum Virtual Machine)',
        'Custom Optimization Consulting',
        '24/7 Dedicated Support',
        'On-premise Deployment',
        'Up to 269x Performance'
      ],
      optimizationsPerMonth: -1, // Unlimited
      consciousnessDiscount: 0.15, // 15% consciousness discount
      stripePrice: 'price_enterprise_monthly',
      popular: false
    }
  ];

  constructor(private http: HttpClient) {
    this.initializeStripe();
  }

  private async initializeStripe() {
    this.stripe = await loadStripe('pk_test_your_stripe_public_key');
  }

  // Calculate consciousness-optimized pricing
  calculateConsciousnessPrice(basePrice: number, consciousnessLevel: number): number {
    const discount = Math.min(0.2, consciousnessLevel * 0.02); // Max 20% discount
    const wallaceDiscount = this.wallaceTransform(consciousnessLevel / 12) * 0.1;
    const totalDiscount = discount + wallaceDiscount;
    return basePrice * (1 - totalDiscount);
  }

  private wallaceTransform(x: number): number {
    const epsilon = 1e-12;
    const adjustedX = Math.max(x, epsilon);
    const logTerm = Math.log(adjustedX + epsilon);
    const phiPower = Math.pow(Math.abs(logTerm), PHI);
    return PHI * phiPower + 1.0;
  }

  // Create payment intent for subscription
  createPaymentIntent(planId: string, consciousnessLevel: number = 1): Observable<PaymentIntent> {
    const plan = this.subscriptionPlans.find(p => p.id === planId);
    if (!plan) throw new Error('Invalid plan');

    const optimizedPrice = this.calculateConsciousnessPrice(plan.price, consciousnessLevel);

    return this.http.post<PaymentIntent>(`${this.apiUrl}/payments/create-intent`, {
      planId,
      amount: Math.round(optimizedPrice * 100), // Stripe expects cents
      currency: plan.currency,
      consciousnessLevel,
      metadata: {
        planName: plan.name,
        originalPrice: plan.price,
        optimizedPrice,
        consciousnessDiscount: plan.consciousnessDiscount
      }
    });
  }

  // Process usage-based billing
  calculateUsageBilling(userId: string, period: string): Observable<UsageBasedBilling> {
    return this.http.post<UsageBasedBilling>(`${this.apiUrl}/billing/calculate-usage`, {
      userId,
      period,
      consciousnessOptimization: true
    });
  }

  // Crypto payment integration
  createCryptoPayment(planId: string, currency: 'BTC' | 'ETH' | 'XCH'): Observable<CryptoPayment> {
    return this.http.post<CryptoPayment>(`${this.apiUrl}/payments/crypto`, {
      planId,
      currency,
      consciousnessEnhanced: true
    });
  }

  // Get current subscription
  getCurrentSubscription(userId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/subscriptions/${userId}`);
  }

  // Get billing history
  getBillingHistory(userId: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/billing/history/${userId}`);
  }
}
