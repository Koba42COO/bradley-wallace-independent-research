# Quick Start Guide - MEAN Stack

## 🚀 Get Started in 5 Minutes

### 1. Install MongoDB
```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0
```

### 2. Setup Backend
```bash
cd backend
npm install
# Create .env file (see backend/.env.example)
npm run dev
```

### 3. Setup Frontend
```bash
cd frontend
npm install
npm start
```

### 4. Access Application
- Frontend: http://localhost:4200
- Backend API: http://localhost:3000
- API Docs: http://localhost:3000/api/health

## 📝 First Steps

1. **Register a new user** at http://localhost:4200/register
2. **Login** with your credentials
3. **Create data** using the Data Management section
4. **Explore** the dashboard and features

## 🔧 Common Commands

```bash
# Backend
cd backend && npm run dev    # Start development server
cd backend && npm start      # Start production server

# Frontend
cd frontend && npm start     # Start Angular dev server
cd frontend && npm run build # Build for production

# Docker
docker-compose up -d         # Start all services
docker-compose down          # Stop all services
docker-compose logs -f       # View logs
```

## 🐛 Troubleshooting

**MongoDB not connecting?**
- Check if MongoDB is running: `brew services list` or `docker ps`
- Verify connection string in `backend/.env`

**Port already in use?**
- Backend: Change `PORT` in `backend/.env`
- Frontend: Use `ng serve --port 4201`

**CORS errors?**
- Ensure `CORS_ORIGIN` in `backend/.env` matches frontend URL

## 📚 More Information

- See `README.md` for full documentation
- See `MEAN_STACK_SETUP.md` for detailed setup
- See `JEFF_MEAN_STACK_UPDATE.md` for update summary
