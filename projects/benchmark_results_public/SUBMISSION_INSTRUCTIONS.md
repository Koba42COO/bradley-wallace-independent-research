# 📤 AIVA Benchmark Results - Submission Instructions

## 🎯 Where to Submit Results

### 1. Papers with Code

**URL:** https://paperswithcode.com/

**Steps:**
1. Create account on Papers with Code
2. Navigate to benchmark pages (MMLU, GSM8K, HumanEval, MATH)
3. Click 'Submit Results'
4. Upload `papers_with_code.json`
5. Add model details and methodology

**Files:** `benchmark_results_public/papers_with_code.json`

### 2. HuggingFace Leaderboards

**URL:** https://huggingface.co/spaces

**Steps:**
1. Create HuggingFace account
2. Navigate to benchmark leaderboards
3. Submit results using `huggingface_leaderboard.json`
4. Add model card with details

**Files:** `benchmark_results_public/huggingface_leaderboard.json`

### 3. GitHub Release

**Steps:**
1. Create new GitHub release
2. Use `github_release_notes.md` as release notes
3. Attach benchmark results files
4. Tag release with version number

**Files:** `benchmark_results_public/github_release_notes.md`

### 4. Public API / Website

**Steps:**
1. Host `index.html` on GitHub Pages or web server
2. Serve `public_api.json` as API endpoint
3. Update with new results as available

**Files:**
- `benchmark_results_public/index.html`
- `benchmark_results_public/public_api.json`

## 📋 Submission Checklist

- [ ] Papers with Code submission
- [ ] HuggingFace leaderboard submission
- [ ] GitHub release created
- [ ] Public results page hosted
- [ ] Documentation updated
