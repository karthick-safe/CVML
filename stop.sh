#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║        🛑 Stopping CVML CardioChek Plus System 🛑            ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Stop backend
echo "🔴 Stopping Backend..."
pkill -f "uvicorn api.main:app" && echo "   ✅ Backend stopped" || echo "   ⚠️  No backend process found"

# Stop frontend
echo "🔴 Stopping Frontend..."
pkill -f "next dev" && echo "   ✅ Frontend stopped" || echo "   ⚠️  No frontend process found"

echo ""
echo "✅ All services stopped"
echo ""
