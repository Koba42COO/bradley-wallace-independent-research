# 🎓 MEAN Stack Training Guide - chAIos Platform
## Complete Training for Node.js, Ionic, Angular & MEAN Stack Architecture

---

## 📚 **TABLE OF CONTENTS**

1. [MEAN Stack Overview](#mean-stack-overview)
2. [Architecture Hierarchy](#architecture-hierarchy)
3. [Node.js Fundamentals](#nodejs-fundamentals)
4. [Angular Framework](#angular-framework)
5. [Ionic Mobile Framework](#ionic-mobile-framework)
6. [Express Server](#express-server)
7. [MongoDB Integration](#mongodb-integration)
8. [Project Structure](#project-structure)
9. [Development Workflow](#development-workflow)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## 🏗️ **MEAN STACK OVERVIEW**

### **What is MEAN Stack?**

**MEAN** = **M**ongoDB + **E**xpress.js + **A**ngular + **N**ode.js

```
┌─────────────────────────────────────────────────────────┐
│                    MEAN STACK LAYERS                    │
├─────────────────────────────────────────────────────────┤
│  Frontend (Client)    │  Angular + Ionic Framework     │
├─────────────────────────────────────────────────────────┤
│  Backend (Server)     │  Express.js + Node.js          │
├─────────────────────────────────────────────────────────┤
│  Database            │  MongoDB (+ Mongoose ODM)       │
├─────────────────────────────────────────────────────────┤
│  Runtime             │  Node.js JavaScript Engine     │
└─────────────────────────────────────────────────────────┘
```

### **chAIos MEAN Stack Implementation:**

- **M**ongoDB: prime aligned compute data storage
- **E**xpress: API server with proxy capabilities
- **A**ngular: Frontend framework with TypeScript
- **N**ode.js: JavaScript runtime for server
- **+Ionic**: Mobile-first UI components

---

## 🏛️ **ARCHITECTURE HIERARCHY**

### **1. Project Structure Hierarchy**

```
chaios-frontend/
├── 📁 src/                          # Source code
│   ├── 📁 app/                      # Angular application
│   │   ├── 📁 core/                 # Core services & guards
│   │   │   ├── 📁 services/         # Business logic services
│   │   │   ├── 📁 guards/           # Route protection
│   │   │   └── 📁 interceptors/     # HTTP interceptors
│   │   ├── 📁 shared/               # Shared components
│   │   │   ├── 📁 components/       # Reusable UI components
│   │   │   ├── 📁 pipes/            # Data transformation
│   │   │   └── 📁 directives/       # Custom directives
│   │   ├── 📁 features/             # Feature modules
│   │   │   ├── 📁 ai-chat/          # AI Chat feature
│   │   │   ├── 📁 prime aligned compute/    # prime aligned compute feature
│   │   │   └── 📁 quantum/          # Quantum Lab feature
│   │   └── 📁 pages/                # Page components
│   ├── 📁 theme/                    # SCSS theme system
│   │   ├── 📁 foundation/           # Variables, functions, mixins
│   │   ├── 📁 base/                 # Reset, typography, icons
│   │   ├── 📁 layout/               # Grid, containers, utilities
│   │   ├── 📁 components/           # Component-specific styles
│   │   ├── 📁 pages/                # Page-specific styles
│   │   └── 📁 themes/               # Theme variations
│   └── 📁 assets/                   # Static assets
├── 📄 server.js                     # Express server
├── 📄 server.config.js              # Server configuration
└── 📄 package.json                  # Dependencies & scripts
```

### **2. Service Architecture Hierarchy**

```
┌─────────────────────────────────────────────────────────┐
│                  SERVICE HIERARCHY                      │
├─────────────────────────────────────────────────────────┤
│  Orchestration Layer  │  OrchestrationService          │
│                      │  AuthOrchestratorService        │
├─────────────────────────────────────────────────────────┤
│  Business Logic      │  ConsciousnessService           │
│                      │  QuantumService                 │
│                      │  AnalyticsService               │
├─────────────────────────────────────────────────────────┤
│  Data Services       │  ApiService                     │
│                      │  WebSocketService               │
│                      │  StorageService                 │
├─────────────────────────────────────────────────────────┤
│  Core Services       │  UXService                      │
│                      │  ConfigService                  │
│                      │  ErrorHandlingService           │
└─────────────────────────────────────────────────────────┘
```

---

## 🟢 **NODE.JS FUNDAMENTALS**

### **What is Node.js?**

Node.js is a JavaScript runtime built on Chrome's V8 engine that allows you to run JavaScript on the server.

### **Key Concepts:**

#### **1. Event Loop & Non-blocking I/O**

```javascript
// ❌ Blocking (Synchronous)
const fs = require('fs');
const data = fs.readFileSync('file.txt'); // Blocks until file is read
console.log(data);

// ✅ Non-blocking (Asynchronous)
fs.readFile('file.txt', (err, data) => {
  if (err) throw err;
  console.log(data);
});
console.log('This runs immediately');
```

#### **2. Modules & Exports**

```javascript
// math.js - Creating a module
function add(a, b) {
  return a + b;
}

function multiply(a, b) {
  return a * b;
}

// Export methods
module.exports = { add, multiply };

// app.js - Using the module
const { add, multiply } = require('./math');
console.log(add(2, 3)); // 5
```

#### **3. NPM Package Management**

```bash
# Initialize project
npm init -y

# Install dependencies
npm install express mongoose cors

# Install dev dependencies
npm install --save-dev nodemon typescript

# Run scripts
npm start
npm run dev
npm test
```

### **chAIos Node.js Implementation:**

```javascript
// server.js - Express server setup
const express = require('express');
const cors = require('cors');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const config = require('./server.config');

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'dist/app')));

// API Proxy
app.use('/api', createProxyMiddleware({
  target: config.api.baseUrl,
  changeOrigin: true,
  pathRewrite: { '^/api': '' }
}));

// SPA Fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/app/index.html'));
});

app.listen(config.server.port, () => {
  console.log(`🚀 chAIos Server running on port ${config.server.port}`);
});
```

---

## 🅰️ **ANGULAR FRAMEWORK**

### **What is Angular?**

Angular is a TypeScript-based web application framework for building dynamic single-page applications (SPAs).

### **Core Concepts:**

#### **1. Components**

```typescript
// component.ts - Component class
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['./example.component.scss']
})
export class ExampleComponent implements OnInit {
  title = 'chAIos';
  items: string[] = [];

  ngOnInit(): void {
    this.loadItems();
  }

  loadItems(): void {
    this.items = ['prime aligned compute', 'Quantum', 'Analytics'];
  }

  onItemClick(item: string): void {
    console.log(`Clicked: ${item}`);
  }
}
```

```html
<!-- component.html - Component template -->
<div class="example-container">
  <h1>{{ title }}</h1>
  <ul>
    <li *ngFor="let item of items" 
        (click)="onItemClick(item)">
      {{ item }}
    </li>
  </ul>
</div>
```

#### **2. Services & Dependency Injection**

```typescript
// service.ts - Injectable service
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = '/api';

  constructor(private http: HttpClient) {}

  getData(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/data`);
  }

  createItem(item: any): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/items`, item);
  }
}

// component.ts - Using the service
export class ComponentExample {
  constructor(private dataService: DataService) {}

  loadData(): void {
    this.dataService.getData().subscribe({
      next: (data) => console.log(data),
      error: (err) => console.error(err)
    });
  }
}
```

#### **3. Routing**

```typescript
// app.routes.ts - Route configuration
import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/ai-chat',
    pathMatch: 'full'
  },
  {
    path: 'ai-chat',
    loadChildren: () => import('./features/ai-chat/ai-chat.routes')
      .then(m => m.routes)
  },
  {
    path: 'prime aligned compute',
    loadChildren: () => import('./features/prime aligned compute/prime aligned compute.routes')
      .then(m => m.routes)
  },
  {
    path: '**',
    redirectTo: '/ai-chat'
  }
];
```

#### **4. Reactive Forms**

```typescript
// form.component.ts - Reactive forms
import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-form',
  template: `
    <form [formGroup]="userForm" (ngSubmit)="onSubmit()">
      <input formControlName="name" placeholder="Name">
      <input formControlName="email" placeholder="Email">
      <button type="submit" [disabled]="userForm.invalid">Submit</button>
    </form>
  `
})
export class FormComponent {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {
    this.userForm = this.fb.group({
      name: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]]
    });
  }

  onSubmit(): void {
    if (this.userForm.valid) {
      console.log(this.userForm.value);
    }
  }
}
```

---

## 📱 **IONIC MOBILE FRAMEWORK**

### **What is Ionic?**

Ionic is a mobile app development framework that uses web technologies (HTML, CSS, JavaScript) to build native-quality mobile apps.

### **Key Features:**

#### **1. UI Components**

```html
<!-- Ionic components in templates -->
<ion-header>
  <ion-toolbar color="primary">
    <ion-title>chAIos</ion-title>
    <ion-buttons slot="end">
      <ion-menu-button></ion-menu-button>
    </ion-buttons>
  </ion-toolbar>
</ion-header>

<ion-content>
  <ion-list>
    <ion-item *ngFor="let item of items" button (click)="selectItem(item)">
      <ion-icon name="chatbubble-outline" slot="start"></ion-icon>
      <ion-label>{{ item.title }}</ion-label>
      <ion-badge color="primary" slot="end">{{ item.count }}</ion-badge>
    </ion-item>
  </ion-list>

  <ion-fab vertical="bottom" horizontal="end">
    <ion-fab-button (click)="addNew()">
      <ion-icon name="add"></ion-icon>
    </ion-fab-button>
  </ion-fab>
</ion-content>
```

#### **2. Navigation**

```typescript
// navigation.service.ts - Navigation service
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { NavController } from '@ionic/angular';

@Injectable({
  providedIn: 'root'
})
export class NavigationService {
  constructor(
    private router: Router,
    private navCtrl: NavController
  ) {}

  // Navigate forward
  navigateForward(url: string): void {
    this.navCtrl.navigateForward(url);
  }

  // Navigate back
  navigateBack(): void {
    this.navCtrl.back();
  }

  // Navigate with data
  navigateWithData(url: string, data: any): void {
    this.router.navigate([url], { state: data });
  }
}
```

#### **3. Native Features**

```typescript
// native.service.ts - Native device features
import { Injectable } from '@angular/core';
import { Capacitor } from '@capacitor/core';
import { Camera, CameraResultType } from '@capacitor/camera';
import { Haptics, ImpactStyle } from '@capacitor/haptics';

@Injectable({
  providedIn: 'root'
})
export class NativeService {
  
  async takePicture(): Promise<string | null> {
    if (!Capacitor.isPluginAvailable('Camera')) {
      return null;
    }

    try {
      const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: true,
        resultType: CameraResultType.DataUrl
      });
      
      return image.dataUrl || null;
    } catch (error) {
      console.error('Camera error:', error);
      return null;
    }
  }

  async hapticFeedback(): Promise<void> {
    if (Capacitor.isPluginAvailable('Haptics')) {
      await Haptics.impact({ style: ImpactStyle.Light });
    }
  }
}
```

---

## 🚀 **EXPRESS SERVER**

### **What is Express.js?**

Express.js is a minimal and flexible Node.js web application framework that provides robust features for web and mobile applications.

### **chAIos Express Implementation:**

#### **1. Server Setup**

```javascript
// server.js - Main server file
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const config = require('./server.config');

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginEmbedderPolicy: false
}));

// Performance middleware
app.use(compression());
app.use(cors(config.cors));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Static files
app.use(express.static(path.join(__dirname, 'dist/app'), {
  maxAge: config.staticFiles.maxAge,
  etag: true,
  lastModified: true
}));

// API proxy to backend
app.use('/api', createProxyMiddleware({
  target: config.api.baseUrl,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
  onError: (err, req, res) => {
    console.error('Proxy Error:', err);
    res.status(500).json({ error: 'Backend service unavailable' });
  }
}));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: require('./package.json').version
  });
});

// SPA fallback - serve index.html for all routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/app/index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server Error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: config.server.environment === 'development' ? err.message : undefined
  });
});

// Start server
const server = app.listen(config.server.port, config.server.host, () => {
  console.log(`🚀 chAIos Express Server Started`);
  console.log(`🌐 Server: http://${config.server.host}:${config.server.port}`);
  console.log(`📁 Serving: ${path.join(__dirname, 'dist/app')}`);
  console.log(`🔄 API Proxy: ${config.api.baseUrl}`);
});

module.exports = app;
```

#### **2. Configuration**

```javascript
// server.config.js - Server configuration
module.exports = {
  server: {
    port: process.env.PORT || 4200,
    host: process.env.HOST || '0.0.0.0',
    environment: process.env.NODE_ENV || 'development'
  },

  api: {
    baseUrl: process.env.API_BASE_URL || 'http://localhost:8000',
    timeout: 30000,
    retries: 3
  },

  cors: {
    origin: process.env.NODE_ENV === 'production' 
      ? ['https://chaios-platform.com']
      : true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
  },

  staticFiles: {
    maxAge: process.env.NODE_ENV === 'production' ? '1y' : '0'
  }
};
```

---

## 🍃 **MONGODB INTEGRATION**

### **What is MongoDB?**

MongoDB is a NoSQL document database that stores data in flexible, JSON-like documents.

### **Integration with MEAN Stack:**

#### **1. Mongoose ODM**

```javascript
// models/user.js - Mongoose model
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  prime aligned compute: {
    level: { type: Number, default: 0 },
    insights: [{ type: String }],
    lastActive: { type: Date, default: Date.now }
  },
  preferences: {
    theme: { type: String, default: 'prime aligned compute' },
    notifications: { type: Boolean, default: true }
  }
}, {
  timestamps: true
});

// Virtual for full name
userSchema.virtual('displayName').get(function() {
  return `${this.name} (Level ${this.prime aligned compute.level})`;
});

// Instance method
userSchema.methods.updateConsciousness = function(insights) {
  this.prime aligned compute.insights.push(...insights);
  this.prime aligned compute.level += insights.length;
  this.prime aligned compute.lastActive = new Date();
  return this.save();
};

// Static method
userSchema.statics.findByConsciousnessLevel = function(minLevel) {
  return this.find({ 'prime aligned compute.level': { $gte: minLevel } });
};

module.exports = mongoose.model('User', userSchema);
```

#### **2. Database Operations**

```javascript
// services/database.js - Database service
const mongoose = require('mongoose');
const User = require('../models/user');

class DatabaseService {
  async connect() {
    try {
      await mongoose.connect(process.env.MONGODB_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true
      });
      console.log('📊 Connected to MongoDB');
    } catch (error) {
      console.error('❌ MongoDB connection error:', error);
      throw error;
    }
  }

  async createUser(userData) {
    try {
      const user = new User(userData);
      await user.save();
      return user;
    } catch (error) {
      throw new Error(`Failed to create user: ${error.message}`);
    }
  }

  async getUserById(userId) {
    try {
      return await User.findById(userId);
    } catch (error) {
      throw new Error(`Failed to get user: ${error.message}`);
    }
  }

  async updateUserConsciousness(userId, insights) {
    try {
      const user = await User.findById(userId);
      if (!user) throw new Error('User not found');
      
      return await user.updateConsciousness(insights);
    } catch (error) {
      throw new Error(`Failed to update prime aligned compute: ${error.message}`);
    }
  }

  async getConsciousUsers(minLevel = 10) {
    try {
      return await User.findByConsciousnessLevel(minLevel);
    } catch (error) {
      throw new Error(`Failed to get conscious users: ${error.message}`);
    }
  }
}

module.exports = new DatabaseService();
```

---

## 📁 **PROJECT STRUCTURE BEST PRACTICES**

### **1. Feature-Based Organization**

```
src/app/
├── 📁 core/                    # Singleton services, guards, interceptors
│   ├── 📁 services/
│   │   ├── api.service.ts      # HTTP API communication
│   │   ├── ux.service.ts       # User experience management
│   │   └── orchestration.service.ts  # Service coordination
│   ├── 📁 guards/
│   │   └── auth.guard.ts       # Route protection
│   └── 📁 interceptors/
│       └── error.interceptor.ts # Global error handling
├── 📁 shared/                  # Shared components, pipes, directives
│   ├── 📁 components/
│   │   ├── loading-skeleton/   # Loading states
│   │   ├── error-state/        # Error handling
│   │   └── empty-state/        # Empty content states
│   └── 📁 pipes/
│       └── prime aligned compute.pipe.ts # Data transformation
├── 📁 features/                # Feature modules
│   ├── 📁 ai-chat/
│   │   ├── 📁 components/      # Feature-specific components
│   │   ├── 📁 services/        # Feature-specific services
│   │   ├── 📁 pages/           # Feature pages
│   │   └── ai-chat.routes.ts   # Feature routing
│   └── 📁 prime aligned compute/
│       ├── 📁 components/
│       ├── 📁 services/
│       ├── 📁 pages/
│       └── prime aligned compute.routes.ts
└── 📁 pages/                   # Global pages
    └── welcome/
```

### **2. Service Layer Architecture**

```typescript
// Orchestration Layer - Coordinates multiple services
@Injectable({ providedIn: 'root' })
export class OrchestrationService {
  constructor(
    private apiService: ApiService,
    private uxService: UXService,
    private storageService: StorageService
  ) {}

  async processUserAction(action: UserAction): Promise<void> {
    try {
      await this.uxService.showLoading();
      const result = await this.apiService.executeAction(action);
      await this.storageService.saveResult(result);
      await this.uxService.showSuccess('Action completed');
    } catch (error) {
      this.uxService.handleError(error);
    } finally {
      await this.uxService.hideLoading();
    }
  }
}

// Business Logic Layer - Domain-specific logic
@Injectable({ providedIn: 'root' })
export class ConsciousnessService {
  private consciousnessState$ = new BehaviorSubject<ConsciousnessState>(null);

  constructor(private orchestrator: OrchestrationService) {}

  async enhanceConsciousness(input: string): Promise<ConsciousnessResult> {
    return this.orchestrator.processUserAction({
      type: 'CONSCIOUSNESS_ENHANCEMENT',
      payload: { input }
    });
  }
}

// Data Layer - API communication
@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  executeAction(action: UserAction): Observable<any> {
    return this.http.post('/api/actions', action);
  }
}
```

---

## 🔄 **DEVELOPMENT WORKFLOW**

### **1. Development Commands**

```bash
# Install dependencies
npm install

# Development server (Angular dev server)
npm run dev              # ng serve (port 4200)

# Production server (Express server)
npm start               # node server.js (port 4200)

# Build for production
npm run build:prod      # ng build --configuration production

# Build and serve
npm run build:serve     # Build + start Express server

# Testing
npm test               # Unit tests
npm run e2e           # End-to-end tests

# Linting
npm run lint          # ESLint + Angular linting
```

### **2. Git Workflow**

```bash
# Feature development
git checkout -b feature/prime aligned compute-enhancement
git add .
git commit -m "feat: add prime aligned compute enhancement feature"
git push origin feature/prime aligned compute-enhancement

# Create pull request
# Merge to main after review

# Release workflow
git checkout main
git pull origin main
git tag v1.2.0
git push origin v1.2.0
```

### **3. Docker Deployment**

```bash
# Build Docker image
docker build -t chaios-frontend .

# Run container
docker run -p 4200:4200 chaios-frontend

# Docker Compose (full stack)
docker-compose -f docker-compose.platform.yml up -d
```

---

## ✅ **BEST PRACTICES**

### **1. Code Organization**

```typescript
// ✅ Good: Feature-based organization
src/app/features/prime aligned compute/
├── components/
├── services/
├── pages/
└── prime aligned compute.module.ts

// ❌ Bad: Type-based organization
src/app/
├── components/
├── services/
└── pages/
```

### **2. Service Design**

```typescript
// ✅ Good: Single responsibility
@Injectable()
export class UserService {
  getUser(id: string): Observable<User> { /* ... */ }
  updateUser(user: User): Observable<User> { /* ... */ }
}

@Injectable()
export class AuthService {
  login(credentials: LoginData): Observable<AuthResult> { /* ... */ }
  logout(): Observable<void> { /* ... */ }
}

// ❌ Bad: Multiple responsibilities
@Injectable()
export class UserAuthService {
  // Handles both user data AND authentication
}
```

### **3. Error Handling**

```typescript
// ✅ Good: Comprehensive error handling
async loadData(): Promise<void> {
  try {
    await this.uxService.showLoading();
    const data = await this.apiService.getData().toPromise();
    this.processData(data);
    await this.uxService.showSuccess('Data loaded');
  } catch (error) {
    this.uxService.handleError(error, 'Failed to load data');
  } finally {
    await this.uxService.hideLoading();
  }
}

// ❌ Bad: No error handling
async loadData(): Promise<void> {
  const data = await this.apiService.getData().toPromise();
  this.processData(data);
}
```

### **4. TypeScript Usage**

```typescript
// ✅ Good: Strong typing
interface ConsciousnessState {
  level: number;
  insights: string[];
  lastUpdated: Date;
}

class ConsciousnessService {
  private state: ConsciousnessState = {
    level: 0,
    insights: [],
    lastUpdated: new Date()
  };

  updateState(newState: Partial<ConsciousnessState>): void {
    this.state = { ...this.state, ...newState };
  }
}

// ❌ Bad: Any types
class ConsciousnessService {
  private state: any = {};
  
  updateState(newState: any): void {
    this.state = newState;
  }
}
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **1. Node.js Version Compatibility**

```bash
# Problem: Angular CLI requires specific Node.js version
# Solution: Use Node Version Manager (nvm)

# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Install and use Node.js 18
nvm install 18
nvm use 18

# Verify version
node --version  # Should show v18.x.x
```

#### **2. Dependency Conflicts**

```bash
# Problem: Peer dependency warnings
# Solution: Use legacy peer deps flag

npm install --legacy-peer-deps

# Or force resolution
npm install --force
```

#### **3. Build Errors**

```bash
# Problem: Memory issues during build
# Solution: Increase Node.js memory limit

NODE_OPTIONS="--max-old-space-size=8192" npm run build

# Problem: TypeScript errors
# Solution: Check tsconfig.json and fix type issues
npx tsc --noEmit  # Type check without building
```

#### **4. Server Issues**

```bash
# Problem: Port already in use
# Solution: Kill process or use different port

# Kill process on port 4200
lsof -ti:4200 | xargs kill -9

# Or use different port
PORT=4201 npm start
```

#### **5. CORS Issues**

```javascript
// Problem: CORS errors in development
// Solution: Configure proxy in angular.json or server

// angular.json proxy configuration
{
  "/api/*": {
    "target": "http://localhost:8000",
    "secure": false,
    "changeOrigin": true,
    "logLevel": "debug"
  }
}

// Or in server.js
app.use(cors({
  origin: ['http://localhost:4200', 'http://localhost:8100'],
  credentials: true
}));
```

---

## 📚 **LEARNING RESOURCES**

### **Official Documentation**
- [Node.js Docs](https://nodejs.org/docs/)
- [Angular Docs](https://angular.io/docs)
- [Ionic Docs](https://ionicframework.com/docs)
- [Express.js Docs](https://expressjs.com/)
- [MongoDB Docs](https://docs.mongodb.com/)

### **Tutorials & Courses**
- [Angular University](https://angular-university.io/)
- [Ionic Academy](https://ionicacademy.com/)
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)

### **Tools & Extensions**
- **VS Code Extensions:**
  - Angular Language Service
  - Ionic Extension Pack
  - TypeScript Importer
  - ESLint
  - Prettier

---

## 🎯 **NEXT STEPS**

1. **Practice**: Build small features using each technology
2. **Explore**: Dive deeper into specific areas of interest
3. **Build**: Create your own MEAN stack projects
4. **Contribute**: Contribute to open-source projects
5. **Stay Updated**: Follow technology updates and best practices

---

**🚀 Ready to build amazing applications with the MEAN stack! Start with the chAIos platform and expand your knowledge from there.**
