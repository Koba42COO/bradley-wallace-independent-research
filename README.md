# MEAN Stack Application

A full-stack web application built with the MEAN stack (MongoDB, Express.js, Angular, Node.js).

## 🚀 Technology Stack

- **MongoDB** - NoSQL database
- **Express.js** - Backend web framework
- **Angular** - Frontend framework
- **Node.js** - JavaScript runtime

## 📁 Project Structure

```
.
├── backend/              # Express.js backend
│   ├── models/          # MongoDB models
│   ├── routes/          # API routes
│   ├── middleware/      # Custom middleware
│   ├── server.js        # Entry point
│   └── package.json     # Backend dependencies
│
├── frontend/            # Angular frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/  # Angular components
│   │   │   ├── services/    # Angular services
│   │   │   ├── models/       # TypeScript models
│   │   │   ├── guards/       # Route guards
│   │   │   └── interceptors/ # HTTP interceptors
│   │   └── environments/    # Environment configs
│   └── package.json         # Frontend dependencies
│
└── docker-compose.yml   # Docker orchestration
```

## 🛠️ Prerequisites

- Node.js 18+ and npm 9+
- MongoDB (or use Docker)
- Angular CLI 17+

## 📦 Installation

### Option 1: Manual Setup

#### Backend Setup

```bash
cd backend
npm install
cp .env.example .env
# Edit .env with your configuration
npm run dev
```

Backend will run on `http://localhost:3000`

#### Frontend Setup

```bash
cd frontend
npm install
npm start
```

Frontend will run on `http://localhost:4200`

### Option 2: Docker Setup

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

```env
PORT=3000
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/mean-app
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=7d
CORS_ORIGIN=http://localhost:4200
```

### Frontend Environment

Edit `frontend/src/environments/environment.ts`:

```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:3000/api'
};
```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user

### Users
- `GET /api/users` - Get all users (Admin only)
- `GET /api/users/:id` - Get user by ID
- `PUT /api/users/:id` - Update user
- `DELETE /api/users/:id` - Delete user (Admin only)

### Data
- `GET /api/data` - Get all data items (with pagination, search, filters)
- `GET /api/data/:id` - Get data item by ID
- `POST /api/data` - Create new data item
- `PUT /api/data/:id` - Update data item
- `DELETE /api/data/:id` - Delete data item

### Health
- `GET /api/health` - Health check endpoint

## 🎯 Features

- ✅ User authentication (JWT)
- ✅ User registration and login
- ✅ Protected routes
- ✅ CRUD operations for data
- ✅ Search and filtering
- ✅ Pagination
- ✅ Responsive design
- ✅ Error handling
- ✅ Input validation

## 🧪 Testing

### Backend Tests
```bash
cd backend
npm test
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🚀 Deployment

### Production Build

#### Backend
```bash
cd backend
NODE_ENV=production npm start
```

#### Frontend
```bash
cd frontend
npm run build
# Serve the dist/ folder with a web server
```

### Docker Production

Update `docker-compose.yml` with production settings and run:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📝 Development

### Backend Development
```bash
cd backend
npm run dev  # Uses nodemon for auto-reload
```

### Frontend Development
```bash
cd frontend
npm start  # Angular dev server with hot reload
```

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Helmet.js security headers
- Input validation
- Route protection
- Role-based access control

## 📚 Documentation

- API documentation available at `http://localhost:3000/api/health` when server is running
- Angular components are documented with TypeScript types
- Models include JSDoc comments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License

## 👤 Author

Jeff

---

**Built with ❤️ using the MEAN Stack**
