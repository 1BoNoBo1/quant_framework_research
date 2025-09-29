// 🚀 QFrame Performance Load Tests with K6
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// 📊 Custom Metrics
const errorRate = new Rate('errors');
const apiResponseTime = new Trend('api_response_time');
const tradingLatency = new Trend('trading_latency');
const orderCounter = new Counter('orders_processed');

// 🔧 Configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp-up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 users
    { duration: '2m', target: 25 },   // Ramp-up to 25 users
    { duration: '5m', target: 25 },   // Stay at 25 users
    { duration: '2m', target: 0 },    // Ramp-down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(99)<1500'], // 99% of requests must complete below 1.5s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate
    api_response_time: ['p(95)<1000'], // 95% of API calls under 1s
    trading_latency: ['p(99)<500'],    // 99% of trading operations under 500ms
  },
};

// 🌐 Base URL from environment
const BASE_URL = __ENV.K6_TARGET_URL || 'http://localhost:8000';

// 📋 Test Data
const symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD'];
const strategies = ['mean_reversion', 'momentum', 'arbitrage'];

// 🔐 Authentication Setup (if needed)
function getAuthHeaders() {
  // return { 'Authorization': 'Bearer ' + token };
  return {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };
}

// 🏠 Health Check Test
export function healthCheck() {
  const response = http.get(`${BASE_URL}/health`, {
    headers: getAuthHeaders(),
  });

  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 200ms': (r) => r.timings.duration < 200,
  });

  errorRate.add(response.status !== 200);
  apiResponseTime.add(response.timings.duration);
}

// 📊 Market Data Test
export function marketDataTest() {
  const symbol = symbols[Math.floor(Math.random() * symbols.length)];

  const response = http.get(`${BASE_URL}/api/v1/market-data/${symbol}`, {
    headers: getAuthHeaders(),
  });

  check(response, {
    'market data status is 200': (r) => r.status === 200,
    'market data has OHLCV': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.ohlcv && Array.isArray(data.ohlcv);
      } catch {
        return false;
      }
    },
    'market data response time < 500ms': (r) => r.timings.duration < 500,
  });

  errorRate.add(response.status !== 200);
  apiResponseTime.add(response.timings.duration);
}

// 🤖 Strategy Performance Test
export function strategyTest() {
  const strategy = strategies[Math.floor(Math.random() * strategies.length)];
  const symbol = symbols[Math.floor(Math.random() * symbols.length)];

  const payload = {
    strategy_name: strategy,
    symbol: symbol,
    timeframe: '1h',
    lookback_days: 30
  };

  const response = http.post(`${BASE_URL}/api/v1/strategies/backtest`,
    JSON.stringify(payload),
    { headers: getAuthHeaders() }
  );

  check(response, {
    'strategy backtest status is 200': (r) => r.status === 200,
    'strategy returns results': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.results && data.performance_metrics;
      } catch {
        return false;
      }
    },
    'strategy response time < 2s': (r) => r.timings.duration < 2000,
  });

  errorRate.add(response.status !== 200);
  tradingLatency.add(response.timings.duration);
}

// 🛒 Order Management Test
export function orderTest() {
  const symbol = symbols[Math.floor(Math.random() * symbols.length)];

  // Create order
  const orderPayload = {
    symbol: symbol,
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    type: 'market',
    quantity: 0.01,
    portfolio_id: 'test-portfolio'
  };

  const createResponse = http.post(`${BASE_URL}/api/v1/orders`,
    JSON.stringify(orderPayload),
    { headers: getAuthHeaders() }
  );

  check(createResponse, {
    'order creation status is 201': (r) => r.status === 201,
    'order has valid ID': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.order_id && typeof data.order_id === 'string';
      } catch {
        return false;
      }
    },
    'order creation time < 300ms': (r) => r.timings.duration < 300,
  });

  if (createResponse.status === 201) {
    orderCounter.add(1);

    try {
      const orderData = JSON.parse(createResponse.body);

      // Get order status
      const statusResponse = http.get(`${BASE_URL}/api/v1/orders/${orderData.order_id}`, {
        headers: getAuthHeaders(),
      });

      check(statusResponse, {
        'order status retrieval is 200': (r) => r.status === 200,
        'order status response time < 200ms': (r) => r.timings.duration < 200,
      });

      tradingLatency.add(statusResponse.timings.duration);
    } catch (e) {
      console.error('Failed to parse order response:', e);
    }
  }

  errorRate.add(createResponse.status !== 201);
  tradingLatency.add(createResponse.timings.duration);
}

// 📈 Portfolio Test
export function portfolioTest() {
  const response = http.get(`${BASE_URL}/api/v1/portfolios/test-portfolio`, {
    headers: getAuthHeaders(),
  });

  check(response, {
    'portfolio status is 200': (r) => r.status === 200,
    'portfolio has balance': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.balance !== undefined;
      } catch {
        return false;
      }
    },
    'portfolio response time < 300ms': (r) => r.timings.duration < 300,
  });

  errorRate.add(response.status !== 200);
  apiResponseTime.add(response.timings.duration);
}

// 🎯 Main Test Function
export default function () {
  // Test weight distribution
  const rand = Math.random();

  if (rand < 0.3) {
    healthCheck();
  } else if (rand < 0.5) {
    marketDataTest();
  } else if (rand < 0.7) {
    strategyTest();
  } else if (rand < 0.9) {
    orderTest();
  } else {
    portfolioTest();
  }

  // Random think time between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

// 🧪 Setup Function
export function setup() {
  console.log('🚀 Starting QFrame Performance Tests');
  console.log(`📍 Target URL: ${BASE_URL}`);

  // Verify API is accessible
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`API Health Check Failed: ${healthResponse.status}`);
  }

  console.log('✅ API Health Check Passed');
  return { baseUrl: BASE_URL };
}

// 🎬 Teardown Function
export function teardown(data) {
  console.log('🏁 Performance Tests Completed');
  console.log(`📊 Total Orders Processed: ${orderCounter.value}`);
  console.log(`📈 Average API Response Time: ${apiResponseTime.avg}ms`);
  console.log(`⚡ Average Trading Latency: ${tradingLatency.avg}ms`);
  console.log(`❌ Error Rate: ${(errorRate.value * 100).toFixed(2)}%`);
}