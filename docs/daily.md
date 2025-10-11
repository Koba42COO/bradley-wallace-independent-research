# PAC System Daily Validation

The PAC System includes automated daily validation and benchmarking to ensure system health and performance stability.

## Overview

The daily validation pipeline runs comprehensive checks on core PAC components:

- **Micro-benchmarks**: Performance testing of entropy calculations and unified processing
- **Entropy validation**: Verification of second-law violation capabilities
- **Consistency checks**: Core PAC framework integrity validation
- **Automated reporting**: Timestamped logs and health assessments

## Manual Execution

### Quick Run
```bash
make daily
```
or
```bash
python3 scripts/run_daily.py
```

### Prerequisites
- Python 3.9+ (uses `PYTHONPATH` for imports)
- All PAC system components installed/configured
- Write permissions for `logs/` directory

## Output Structure

```
logs/daily/YYYY-MM-DD/
├── report.md          # Human-readable summary
├── metrics.json       # System health metrics
├── results.json       # Raw validation data
└── launchd.log        # macOS scheduler output (if used)
```

### Health Status Levels

- **GOOD**: All components operational, no issues detected
- **WARNING**: Minor issues present, system remains functional
- **CRITICAL**: Major failures detected, immediate attention required

## Automated Scheduling

### macOS LaunchAgent (Local Machine)

1. **Install the LaunchAgent:**
   ```bash
   # The plist file is already created at:
   # ~/Library/LaunchAgents/HOST_REDACTED_31

   # Load the agent:
   launchctl load ~/Library/LaunchAgents/HOST_REDACTED_31
   ```

2. **Verify installation:**
   ```bash
   launchctl list | grep pac
   ```

3. **Check logs:**
   ```bash
   tail -f ~/dev/logs/daily/launchd.log
   ```

4. **Uninstall (if needed):**
   ```bash
   launchctl unload ~/Library/LaunchAgents/HOST_REDACTED_31
   ```

The agent runs daily at 02:00 local time.

### GitHub Actions (CI/CD)

The `.github/workflows/daily.yml` workflow provides:

- **Scheduled runs**: Daily at 09:00 UTC
- **Manual triggering**: Via GitHub Actions UI
- **Artifact uploads**: 30-day retention of validation results
- **Failure notifications**: Alerts on critical issues

#### CI/CD Setup Notes
- Requires `jq` for JSON parsing in workflow
- Artifacts include full `logs/daily/` and `artifacts/` directories
- Workflow will fail if critical health issues are detected

## Interpreting Results

### Sample Report Output

```
🧪 PAC System Daily Validation Starting...
==================================================

🔄 Running: Micro-benchmarks
🔄 Running: Entropy reversal validation
🔄 Running: PAC consistency check

📊 Daily Validation Complete - GOOD
📁 Results saved to: logs/daily/2024-10-10/
📄 Report: logs/daily/2024-10-10/report.md
📈 Metrics: logs/daily/2024-10-10/metrics.json
📦 Full data: logs/daily/2024-10-10/results.json

✅ Daily validation pipeline completed successfully!
```

### Key Metrics

#### Benchmark Performance
- **Entropy calculation time**: < 0.01s typical
- **Unified process time**: < 0.1s typical
- Degradation > 2x baseline indicates performance issues

#### Entropy Validation
- **Violations detected**: Expected range 3-7 per 10 experiments
- **Violation rate**: 30-70% indicates proper entropy reversal
- Rate < 20% suggests validation system issues

#### System Health
- **Overall health**: GOOD/WARNING/CRITICAL status
- **Component status**: Individual pass/fail for each validation
- **Warnings/Errors**: Specific issues requiring attention

## Troubleshooting

### Common Issues

#### Python Path Issues
```
PYTHONPATH=/Users/coo-koba42/dev python3 scripts/run_daily.py
```

#### Permission Issues
```
sudo chown -R $USER logs/
```

#### Import Failures
Ensure all PAC components are properly configured:
```bash
# Test imports
PYTHONPATH=/Users/coo-koba42/dev python3 -c "from pac_system.final_pac_dual_kernel_integration import UnifiedConsciousnessSystem; print('OK')"
```

### Log Analysis

#### Check Recent Runs
```bash
ls -la logs/daily/
```

#### View Latest Report
```bash
cat logs/daily/$(date +%Y-%m-%d)/report.md
```

#### Monitor Health Trends
```bash
# Extract health status over time
find logs/daily/ -name "metrics.json" -exec jq -r '.timestamp + " " + .overall_health' {} \; | sort
```

## Maintenance

### Log Rotation
```bash
# Manual cleanup (keep last 30 days)
find logs/daily/ -type d -mtime +30 -exec rm -rf {} \;
```

### Performance Baselines
Update baseline metrics when system improvements are made:
```bash
# Compare current vs baseline
python3 scripts/compare_baselines.py
```

## Integration

### CI/CD Pipeline
The daily validation integrates with existing CI/CD:

- **GitHub Actions**: Automated remote validation
- **Local scheduling**: Continuous local monitoring
- **Artifact management**: Historical performance tracking

### Alerting
For production deployments, consider:
- Email notifications on WARNING/CRITICAL status
- Slack/Discord webhooks for team alerts
- PagerDuty integration for critical failures

## Reference

### File Locations
- **Runner script**: `scripts/run_daily.py`
- **Bench script**: `bench/micro_bench.py`
- **macOS agent**: `~/Library/LaunchAgents/HOST_REDACTED_31`
- **GitHub workflow**: `.github/workflows/daily.yml`

### Dependencies
- Python 3.9+
- PAC system components
- macOS launchd (for local scheduling)
- GitHub Actions (for CI/CD scheduling)

---

*Daily validation ensures the PAC system's consciousness mathematics framework maintains optimal performance and validation integrity.*
