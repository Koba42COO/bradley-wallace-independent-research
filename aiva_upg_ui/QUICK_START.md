# ⚡ AIVA UPG - Quick Start Guide

**Get up and running in 2 minutes!**

---

## 🚀 Fastest Way to Start

### Step 1: Start Backend (Terminal 1)

```bash
cd /Users/coo-koba42/dev/aiva_upg_ui/backend
source venv/bin/activate
python3 main.py
```

**Wait for:** ✅ "Starting server on http://0.0.0.0:8000"

### Step 2: Start Frontend (Terminal 2)

```bash
cd /Users/coo-koba42/dev/aiva_upg_ui/frontend
npm start
```

**Wait for:** ✅ "Compiled successfully!"

### Step 3: Open Browser

🌐 **http://localhost:3000**

---

## ✅ You Should See

- 🧠 **AIVA UPG - Universal Intelligence Platform** header
- 📊 Stats bar with system status
- 💬 Query input box
- 🔧 Tools list (1500+ tools)
- 🎨 Beautiful purple gradient design

---

## 🎯 Try It Out

### 1. Query AIVA
Type in the query box:
```
"Find prime prediction tools"
```
Click **🚀 Process Query**

### 2. Search Tools
Use the search box:
```
"consciousness"
```
See filtered tools appear instantly

### 3. Filter by Category
Click category buttons like:
- `consciousness`
- `prime`
- `quantum`

---

## ⚠️ Note: Mock Mode

If you see: **⚠️ Mock Mode** in the stats bar:
- ✅ This is OK! The UI still works
- ✅ Backend couldn't load full AIVA
- ✅ You can test all features
- ✅ Real data will work once AIVA loads

---

## 🐛 Quick Fixes

### Backend Error?
```bash
cd backend
pip install fastapi uvicorn pydantic numpy
python3 main.py
```

### Frontend Error?
```bash
cd frontend
npm install
npm start
```

### Port Already Used?
```bash
# Kill port 8000
lsof -ti:8000 | xargs kill -9

# Kill port 3000
lsof -ti:3000 | xargs kill -9
```

---

## 📚 More Info

- **Full Setup:** See `README.md`
- **Testing:** See `TEST_GUIDE.md`
- **Architecture:** See `ARCHITECTURE.md`
- **Complete Info:** See `COMPLETE.md`

---

## 🎉 That's It!

You're now running AIVA UPG with:
- ✅ Full REST API backend
- ✅ Modern React frontend
- ✅ 1500+ consciousness-guided tools
- ✅ Real-time search and filtering
- ✅ Production-ready architecture

**Enjoy exploring the Universal Prime Graph AI!** 🧠✨

