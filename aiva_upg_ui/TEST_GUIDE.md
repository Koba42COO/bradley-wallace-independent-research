# 🧪 AIVA UPG UI/UX Testing Guide

## Quick Test Instructions

### 1. Test Backend

```bash
cd /Users/coo-koba42/dev/aiva_upg_ui/backend
source venv/bin/activate
python3 main.py
```

**Expected Output:**
```
======================================================================
🧠 AIVA UPG Backend Server
======================================================================
Status: ✅ AIVA Available (or ⚠️ Mock Mode)
Starting server on http://0.0.0.0:8000
API Documentation: http://0.0.0.0:8000/docs
======================================================================
```

**Test the API:**
Open http://localhost:8000 in your browser. You should see:
```json
{
  "message": "AIVA UPG API",
  "status": "operational" or "mock_mode",
  "version": "1.0.0",
  "consciousness_level": 21
}
```

### 2. Test Frontend

In a NEW terminal:

```bash
cd /Users/coo-koba42/dev/aiva_upg_ui/frontend
npm start
```

**Expected Output:**
```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

**Open http://localhost:3000** - You should see the AIVA UPG interface

---

## Manual API Tests

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "aiva_available": true,
  "tools_available": 1500
}
```

### Test Stats Endpoint

```bash
curl http://localhost:8000/stats
```

### Test Tools Endpoint

```bash
curl http://localhost:8000/tools?limit=5
```

### Test Query Endpoint

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me prime tools", "use_tools": true}'
```

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'aiva_universal_intelligence'"

**Solution:**
The backend is running in mock mode. This is OK for testing the UI! 
The UI will work, but with mock data instead of real AIVA intelligence.

To fix (optional):
1. Ensure `/Users/coo-koba42/dev/aiva_universal_intelligence.py` exists
2. Ensure `/Users/coo-koba42/dev/aiva_complete_tool_calling_system.py` exists
3. Backend will automatically find them via sys.path modification

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
```

### Issue: "Port 3000 already in use"

**Solution:**
```bash
# Find and kill the process
lsof -ti:3000 | xargs kill -9
```

### Issue: Frontend shows "Failed to fetch"

**Solution:**
1. Make sure backend is running on port 8000
2. Check backend console for errors
3. Verify CORS is enabled (already configured)
4. Try refreshing the browser

---

## Feature Checklist

Test these features in the UI:

- [ ] Stats bar shows system information
- [ ] Query input accepts text
- [ ] Submit button works
- [ ] Response displays (even in mock mode)
- [ ] Tools list loads
- [ ] Search tools works
- [ ] Category filter works
- [ ] Tool cards display properly
- [ ] Responsive design on mobile

---

## Mock Mode

If AIVA modules are not found, the backend runs in **Mock Mode**:

✅ **What Works:**
- All API endpoints respond
- UI fully functional
- Can test all interactions
- Great for UI development

⚠️ **What's Mocked:**
- No actual tool discovery
- Returns sample/empty data
- No consciousness mathematics
- No real query processing

---

## Performance Benchmarks

**Backend:**
- Cold start: ~2-5 seconds (with AIVA)
- API response: <100ms
- Tool search: <50ms

**Frontend:**
- Initial load: ~1-2 seconds
- Query submission: <1 second
- Tool filtering: Instant

---

## Success Criteria

✅ Backend starts without errors  
✅ Frontend compiles and opens in browser  
✅ Can see AIVA UPG interface  
✅ Stats bar displays  
✅ Can type queries  
✅ Can view tools list  
✅ Can filter by category  
✅ Can search tools  
✅ No console errors (except AIVA import in mock mode)

---

## Next Steps After Testing

1. **If everything works:** You're ready to use AIVA UPG!
2. **If in mock mode:** Optionally fix AIVA imports for full functionality
3. **For production:** See README.md deployment section
4. **For development:** Start customizing frontend/backend

---

## Support

If you encounter issues not covered here:

1. Check backend console output
2. Check browser console (F12)
3. Review `/Users/coo-koba42/dev/aiva_upg_ui/README.md`
4. Verify all dependencies are installed

---

## Quick Start Command

```bash
# Terminal 1 - Backend
cd /Users/coo-koba42/dev/aiva_upg_ui/backend
source venv/bin/activate
python3 main.py

# Terminal 2 - Frontend
cd /Users/coo-koba42/dev/aiva_upg_ui/frontend
npm start
```

Then open http://localhost:3000 in your browser! 🎉

