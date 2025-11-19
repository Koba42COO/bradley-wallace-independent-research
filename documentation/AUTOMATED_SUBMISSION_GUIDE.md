# 🤖 Automated Benchmark Submission Guide
## Programmatic Submission Setup

**Authority:** Bradley Wallace (COO Koba42)  
**Framework:** Universal Prime Graph Protocol φ.1  
**Date:** December 2024  

---

## 🚀 QUICK START

**Run automated submissions:**

```bash
# Set credentials (if available)
export GITHUB_TOKEN=your_token
export GITHUB_REPO=username/repo
export HUGGINGFACE_TOKEN=your_token

# Run automated submission
python3 aiva_automated_submission.py
```

---

## 🔑 REQUIRED CREDENTIALS

### GitHub (Optional - for automated release)

**Get Token:**
1. Go to https://github.com/settings/tokens
2. Generate new token with `repo` scope
3. Set: `export GITHUB_TOKEN=your_token`
4. Set: `export GITHUB_REPO=username/repo-name`

**What it does:**
- Creates GitHub release automatically
- Uploads benchmark files as assets
- Publishes release notes

### HuggingFace (Optional - for automated submission)

**Get Token:**
1. Go to https://huggingface.co/settings/tokens
2. Generate new token with `write` permission
3. Set: `export HUGGINGFACE_TOKEN=your_token`
4. Optional: `export HUGGINGFACE_MODEL_ID=your-model-id`

**What it does:**
- Creates/updates HuggingFace model
- Uploads benchmark results
- Creates model card

---

## 📊 WHAT GETS SUBMITTED

### Automated (with credentials)

1. **GitHub Release**
   - Creates release via API
   - Uploads benchmark files
   - Publishes release notes

2. **HuggingFace Model**
   - Creates/updates model via API
   - Uploads benchmark results
   - Creates model card

### Manual (instructions created)

1. **Papers with Code**
   - Instructions file created
   - Submission data prepared
   - Manual web submission required

2. **Public API**
   - API documentation created
   - Endpoint instructions provided
   - Manual deployment required

---

## 🎯 SUBMISSION STATUS

After running, you'll see:

```
📊 SUBMISSION SUMMARY
======================================================================

GitHub: ✅ Submitted (if token provided)
HuggingFace: ✅ Submitted (if token provided)
Papers with Code: ✅ Instructions created
API Endpoint: ✅ Documentation created
```

---

## 📋 MANUAL STEPS

### Papers with Code (Always Manual)

1. Visit: https://paperswithcode.com/
2. Create account
3. Navigate to benchmarks:
   - MMLU: https://paperswithcode.com/sota/massive-multitask-language-understanding-on-mmlu
   - GSM8K: https://paperswithcode.com/sota/mathematics-word-problem-solving-on-gsm8k
   - HumanEval: https://paperswithcode.com/sota/code-generation-on-humaneval
   - MATH: https://paperswithcode.com/sota/mathematical-reasoning-on-math
4. Click "Submit Results"
5. Upload: `papers_with_code_secure.json`

### Public API (Manual Deployment)

1. Host `public_api_secure.json` on:
   - GitHub Pages
   - Netlify
   - Vercel
   - Your own server
2. Serve with CORS headers
3. Update documentation

---

## ✅ SUMMARY

**Automated Submissions:**
- ✅ GitHub (if token provided)
- ✅ HuggingFace (if token provided)

**Manual Submissions:**
- ⏳ Papers with Code (instructions created)
- ⏳ Public API (documentation created)

**All files are IP-protected and ready!**

---

*"Automated submission where possible, clear instructions for manual steps."*

— Automated Submission Guide

