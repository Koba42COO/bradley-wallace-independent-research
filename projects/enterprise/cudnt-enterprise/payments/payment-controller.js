const express = require('express');
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const crypto = require('crypto');
const router = express.Router();

// prime aligned compute Mathematics
const PHI = (1 + Math.sqrt(5)) / 2;
const CONSCIOUSNESS_RATIO = 79/21;

// Payment models
const PaymentIntent = require('../models/PaymentIntent');
const Subscription = require('../models/Subscription');
const Usage = require('../models/Usage');

// Wallace Transform for pricing optimization
const wallaceTransform = (x, alpha = PHI, beta = 1.0, epsilon = 1e-12) => {
  const adjustedX = Math.max(x, epsilon);
  const logTerm = Math.log(adjustedX + epsilon);
  const phiPower = Math.pow(Math.abs(logTerm), PHI);
  const sign = Math.sign(logTerm);
  return alpha * phiPower * sign + beta;
};

// Create payment intent with prime aligned compute optimization
router.post('/create-intent', async (req, res) => {
  try {
    const { planId, amount, currency, consciousnessLevel, metadata } = req.body;

    // Apply prime aligned compute mathematics discount
    const consciousnessDiscount = Math.min(0.2, consciousnessLevel * 0.02);
    const wallaceDiscount = wallaceTransform(consciousnessLevel / 12) * 0.1;
    const totalDiscount = consciousnessDiscount + wallaceDiscount;
    const optimizedAmount = Math.round(amount * (1 - totalDiscount));

    const paymentIntent = await stripe.paymentIntents.create({
      amount: optimizedAmount,
      currency: currency.toLowerCase(),
      metadata: {
        ...metadata,
        consciousnessLevel,
        originalAmount: amount,
        optimizedAmount,
        discount: totalDiscount
      }
    });

    // Save to database
    const dbPaymentIntent = new PaymentIntent({
      stripeId: paymentIntent.id,
      planId,
      amount: optimizedAmount,
      originalAmount: amount,
      consciousnessLevel,
      discount: totalDiscount,
      status: 'pending'
    });
    await dbPaymentIntent.save();

    res.json({
      clientSecret: paymentIntent.client_secret,
      amount: optimizedAmount,
      currency,
      metadata: paymentIntent.metadata
    });

  } catch (error) {
    console.error('Payment intent creation failed:', error);
    res.status(500).json({ error: 'Payment processing failed' });
  }
});

// Usage-based billing calculation
router.post('/calculate-usage', async (req, res) => {
  try {
    const { userId, period } = req.body;

    // Get usage data from database
    const usage = await Usage.findOne({ userId, period });
    if (!usage) {
      return res.status(404).json({ error: 'Usage data not found' });
    }

    // Get user's subscription
    const subscription = await Subscription.findOne({ userId, status: 'active' });
    const baseRate = subscription ? subscription.baseRate : 0;

    // Calculate usage charges with prime aligned compute optimization
    const optimizationRate = 0.01; // $0.01 per optimization
    const processingRate = 0.001; // $0.001 per second
    const transferRate = 0.0001; // $0.0001 per MB
    const apiRate = 0.0001; // $0.0001 per API call

    const usageCharges = {
      optimizations: usage.optimizations * optimizationRate,
      processingTime: usage.processingTime * processingRate,
      dataTransfer: usage.dataTransfer * transferRate,
      apiCalls: usage.apiCalls * apiRate
    };

    // Apply prime aligned compute mathematics discounts
    const consciousnessLevel = usage.avgConsciousnessLevel || 1;
    const consciousnessBonus = consciousnessLevel * 0.05; // 5% per level
    const wallaceBonus = wallaceTransform(consciousnessLevel / 12) * 0.1;

    const totalUsageCharges = Object.values(usageCharges).reduce((sum, charge) => sum + charge, 0);
    const discountedCharges = totalUsageCharges * (1 - consciousnessBonus - wallaceBonus);

    const billing = {
      userId,
      period,
      baseSubscription: baseRate,
      usageCharges,
      discounts: {
        consciousnessBonus,
        loyaltyDiscount: subscription.loyaltyDiscount || 0,
        volumeDiscount: totalUsageCharges > 1000 ? 0.1 : 0
      },
      totalAmount: baseRate + discountedCharges,
      nextBillingDate: subscription.nextBillingDate
    };

    res.json(billing);

  } catch (error) {
    console.error('Usage billing calculation failed:', error);
    res.status(500).json({ error: 'Billing calculation failed' });
  }
});

// Crypto payment integration
router.post('/crypto', async (req, res) => {
  try {
    const { planId, currency } = req.body;

    // Generate crypto wallet address (implementation depends on chosen crypto provider)
    const address = await generateCryptoAddress(currency);

    // Calculate amount in crypto
    const plan = getSubscriptionPlan(planId);
    const cryptoAmount = await convertToCrypto(plan.price, currency);

    // Generate QR code
    const qrCode = await generateQRCode(`${currency}:${address}?amount=${cryptoAmount}`);

    res.json({
      address,
      amount: cryptoAmount,
      currency,
      qrCode,
      expiresAt: Date.now() + (30 * 60 * 1000) // 30 minutes
    });

  } catch (error) {
    console.error('Crypto payment creation failed:', error);
    res.status(500).json({ error: 'Crypto payment failed' });
  }
});

// Helper functions (implement based on chosen providers)
async function generateCryptoAddress(currency) {
  // Implement with crypto payment provider (Coinbase Commerce, BitPay, etc.)
  const addresses = {
    BTC: 'bc1q' + crypto.randomBytes(32).toString('hex').substring(0, 38),
    ETH: '0x' + crypto.randomBytes(20).toString('hex'),
    XCH: 'xch1' + crypto.randomBytes(32).toString('hex').substring(0, 59)
  };
  return addresses[currency] || addresses.BTC;
}

async function convertToCrypto(amount, currency) {
  // Implement with price feed API (CoinMarketCap, CoinGecko, etc.)
  const rates = {
    BTC: amount / 45000, // Example conversion
    ETH: amount / 3000,
    XCH: amount / 50
  };
  return rates[currency] || rates.BTC;
}

async function generateQRCode(data) {
  // Implement with QR code library (qrcode, etc.)
  return `data:image/png;base64,${Buffer.from(data).toString('base64')}`;
}

function getSubscriptionPlan(planId) {
  const plans = {
    developer: { price: 99 },
    professional: { price: 999 },
    enterprise: { price: 25000 }
  };
  return plans[planId] || plans.developer;
}

module.exports = router;
