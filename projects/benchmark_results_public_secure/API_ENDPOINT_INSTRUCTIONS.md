# AIVA Benchmark Results API

## Endpoint

Serve `public_api_secure.json` as API endpoint.

## Usage

```bash
# Serve via Python
python3 -m http.server 8000

# Or via Node.js
npx serve .

# Or upload to GitHub Pages, Netlify, etc.
```

## Response Format

```json
{
  "data": {
    "model": { ... },
    "benchmarks": [ ... ]
  },
  "metadata": {
    "ip_protected": true,
    "obfuscation_level": "high"
  }
}
```

## CORS Headers

If serving via API, include:
- Access-Control-Allow-Origin: *
- Content-Type: application/json

---
Generated: 2025-11-10T13:43:55.955619
