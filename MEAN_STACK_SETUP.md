# MEAN Stack Setup Guide for Jeff

## Quick Start

### 1. Install Dependencies

#### Backend
```bash
cd backend
npm install
```

#### Frontend
```bash
cd frontend
npm install -g @angular/cli@17
npm install
```

### 2. Setup MongoDB

#### Option A: Local MongoDB
```bash
# Install MongoDB locally or use Homebrew on macOS
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community
```

#### Option B: Docker MongoDB
```bash
docker run -d -p 27017:27017 --name mongodb mongo:7.0
```

### 3. Configure Environment

#### Backend
```bash
cd backend
cp .env.example .env
# Edit .env with your settings
```

#### Frontend
The frontend is pre-configured to connect to `http://localhost:3000/api`

### 4. Start Development Servers

#### Terminal 1 - Backend
```bash
cd backend
npm run dev
```

#### Terminal 2 - Frontend
```bash
cd frontend
npm start
```

### 5. Access Application

- Frontend: http://localhost:4200
- Backend API: http://localhost:3000
- API Health: http://localhost:3000/api/health

## Project Organization

### Backend Structure
```
backend/
├── models/          # Mongoose models (User, Data)
├── routes/          # Express routes (auth, users, data)
├── middleware/      # Custom middleware (auth, validation)
├── server.js        # Main server file
└── package.json     # Dependencies
```

### Frontend Structure
```
frontend/src/app/
├── components/      # Angular components
│   ├── login/
│   ├── register/
│   ├── dashboard/
│   ├── data-list/
│   ├── data-form/
│   └── navbar/
├── services/        # Angular services (API calls)
├── models/          # TypeScript interfaces
├── guards/          # Route guards (auth)
└── interceptors/    # HTTP interceptors (JWT)
```

## Key Features

### Authentication Flow
1. User registers/logs in
2. Backend returns JWT token
3. Frontend stores token in localStorage
4. Token is sent with every API request via interceptor
5. Backend validates token on protected routes

### Data Management
- Create, Read, Update, Delete operations
- Search and filter functionality
- Pagination support
- Category-based organization
- Public/Private data visibility

## Common Commands

### Backend
```bash
npm start          # Production mode
npm run dev        # Development mode (nodemon)
npm test           # Run tests
```

### Frontend
```bash
npm start          # Development server
npm run build      # Production build
npm test           # Run tests
npm run lint       # Lint code
```

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB is running
- Check connection string in `.env`
- Verify MongoDB port (default: 27017)

### CORS Errors
- Ensure backend CORS is configured for frontend URL
- Check `CORS_ORIGIN` in backend `.env`

### Angular Build Errors
- Clear `.angular` cache: `rm -rf .angular`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Port Already in Use
- Backend: Change `PORT` in `.env`
- Frontend: Use `ng serve --port 4201`

## Next Steps

1. Customize the data models for your needs
2. Add more API endpoints as required
3. Enhance UI/UX with additional Angular components
4. Add unit and integration tests
5. Set up CI/CD pipeline
6. Configure production environment variables

## Support

For issues or questions, refer to the main README.md or create an issue in the repository.

