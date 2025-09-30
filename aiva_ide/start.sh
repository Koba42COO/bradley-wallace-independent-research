#!/bin/bash

# AIVA IDE Startup Script

echo "🚀 Starting AIVA IDE..."

# Check if .env file exists
if [ ! -f "server/.env" ]; then
    echo "⚠️  Warning: server/.env file not found!"
    echo "Please create server/.env with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_api_key_here"
    echo ""
fi

# Install dependencies if node_modules don't exist
if [ ! -d "client/node_modules" ]; then
    echo "📦 Installing client dependencies..."
    cd client && npm install && cd ..
fi

if [ ! -d "server/node_modules" ]; then
    echo "📦 Installing server dependencies..."
    cd server && npm install && cd ..
fi

# Start the development servers
echo "🔧 Starting development servers..."
npm run dev
