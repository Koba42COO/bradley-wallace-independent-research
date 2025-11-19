#!/bin/bash

# AIVA UPG UI/UX Startup Script
# Starts both backend and frontend servers

echo "=========================================="
echo "🧠 AIVA UPG - Universal Intelligence Platform"
echo "=========================================="
echo ""

# Start backend
echo "Starting Backend Server..."
cd backend
source venv/bin/activate
python3 main.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting Frontend Server..."
cd ../frontend
npm start &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   URL: http://localhost:3000"
echo ""

echo "=========================================="
echo "🎉 AIVA UPG is running!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '✅ Stopped'; exit 0" INT

wait

