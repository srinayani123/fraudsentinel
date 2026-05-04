---
title: FraudSentinel
emoji: 🛡️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Agentic fraud detection — multi-agent investigation in 30s
---

# FraudSentinel

Agentic fraud detection platform. Multi-agent pipeline that flags a transaction, attributes the score, retrieves similar historical patterns, and synthesizes an analyst-facing decision in ~30 seconds.

## Quick start

1. **Sign in** with Google (top right)
2. **Connect your Anthropic API key** in Settings — the app uses bring-your-own-key, your key stays in your browser session only and is never saved to disk
3. **Open Investigate** and click any flagged transaction
4. Click **Run investigation** to watch the multi-agent pipeline work

## What's running here

- 4-agent investigation pipeline (Triage → Investigator → Pattern → Report)
- Two-tier pattern verification — semantic retrieval + indicator-fit checks
- Rule generator (Planner + 4 parallel Workers + Synthesizer)
- 377-pattern fraud archetype library
- SHAP-grounded fallback when no pattern fits

Built on the IEEE-CIS Fraud Detection dataset.

## Source

Full source, methodology, and documentation: **[github.com/srinayani123/fraudsentinel](https://github.com/srinayani123/fraudsentinel)**

## Cost

The investigation pipeline uses ~$0.05 per run (Claude Sonnet 4.5 + Haiku 4.5). The rule generator uses ~$0.15 per run. All charged to **your** Anthropic key — this Space hosts the app but never sees your transactions or your billing.

---
