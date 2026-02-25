#!/bin/bash

# Port colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Démarrage du projet Portfolio Velocity...${NC}"

# Cleanup ports 8000 and 5173
echo -e "${BLUE}🧹 Nettoyage des ports (8000, 5173)...${NC}"
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :5173 | xargs kill -9 2>/dev/null

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}🛑 Arrêt des serveurs...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# 1. Start Backend
echo -e "${GREEN}⚙️  Lancement du Backend (FastAPI)...${NC}"
python3 main.py > backend.log 2>&1 &
BACKEND_PID=$!

# 2. Wait a bit for backend to initialize
sleep 2

# 3. Start Frontend
echo -e "${GREEN}🎨 Lancement du Frontend (Vite)...${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!

echo -e "\n${BLUE}✅ Tout est prêt !${NC}"
echo -e "🔗 Frontend : ${GREEN}http://localhost:5173${NC}"
echo -e "🔗 API Docs : ${GREEN}http://localhost:8000/docs${NC}"
echo -e "\nAppuyez sur ${RED}Ctrl+C${NC} pour tout arrêter."

# Wait for both processes
wait
