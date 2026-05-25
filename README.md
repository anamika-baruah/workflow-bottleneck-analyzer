# 🚀 AI Workflow Intelligence System

> **Turn raw workflow logs into actionable business decisions using 
> AI-driven process intelligence.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Engine-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)]()

---

## 📌 Overview

Most businesses run on complex, multi-step workflows — sales pipelines,
support queues, HR onboarding — but as volume scales, it becomes
impossible to know where time is being lost, which paths are deviating
from standard, and what fixing the worst offenders is actually worth.

The **AI Workflow Intelligence System** is an operations intelligence
and decision-support platform that automatically ingests raw workflow
event logs and surfaces the insights that matter most:

- *Which step is creating the most delay?*
- *How many distinct workflow paths exist — and what does each one cost?*
- *Is the bottleneck caused by workload or just waiting in a queue?*
- *What is the recoverable operational waste if we fix the worst path?*

Rather than building another metrics dashboard, this system acts as a
**strategic analytical layer** — translating raw operational data into
structured, boardroom-ready decisions. It moves teams from reactive
firefighting to proactive, evidence-based optimization.

---

## ✨ What Makes This Different?

Most analytics tools stop at visualization. This system goes further:

| Typical Dashboard | AI Workflow Intelligence System |
|---|---|
| Shows *what* happened | Explains *why* it happened |
| Displays metrics | Prescribes specific actions |
| Reports the past | Predicts future risks |
| Treats all cases the same | Discovers hidden workflow variants |
| Requires manual interpretation | Quantifies recoverable waste in hours |

- 🔍 **Not just visualization** — diagnoses root causes behind every delay
- 🎯 **Prescribes exactly what to do** — Problem → Cause → Action → Impact
- 💰 **Quantifies business impact** — recoverable hours, SLA risk, cycle cost
- 🔀 **Discovers hidden process variants** — finds every path your cases 
  actually follow, not just the one they're supposed to
- ⚠️ **Predicts risks before they happen** — using statistical variability 
  analysis across task patterns

👉 *Acts like a mini operations consultant, not just an analytics tool.*

---

## ❗ Problem Statement

In any modern enterprise, operational efficiency quietly erodes due to:

- **Invisible Bottlenecks** — A single approval step can silently consume
  60%+ of total process time without anyone noticing until it impacts revenue.
- **Zero Visibility into Idle Time** — No easy way to distinguish between
  active human work and time a task spends sitting in a queue.
- **Process Deviations Going Undetected** — Tasks being skipped, repeated
  unnecessarily (rework loops), or executed out of sequence create hidden
  quality and compliance risks.
- **No Cost Attached to Deviations** — Teams know deviations exist but
  can't answer: *how many hours are we losing to this specific path?*
- **No Predictive Layer** — Teams only discover a process is failing
  *after* SLAs are already breached, not before.

These aren't edge cases. They're the silent drains on efficiency in every
operations, sales, and support team — and they're largely invisible
without the right tooling.

---

## 💡 Solution

A high-fidelity diagnostic and decision engine for any business process:

- **Pinpoints the Exact Bottleneck** — Identifies which task is the
  primary constraint, with precise time and percentage-of-workflow data.
- **Mines Process Variants** — Discovers every unique sequence of steps
  cases actually follow, classifies each as conformant, rework loop, skip,
  or extended, and scores them on cycle time, SLA breach rate, and wait ratio.
- **Quantifies Recoverable Waste** — Calculates total hours that could be
  reclaimed if deviant paths were standardized to the optimal sequence.
- **Decodes Root Causes** — Determines whether delay comes from task
  complexity or queue time, which lead to completely different fixes.
- **Automation Roadmap** — Identifies candidates for RPA or rule-based
  automation with estimated time savings per recommendation.
- **Risk Prediction** — Flags tasks with high statistical variability as
  future failure points before they breach SLAs, using a transparent
  0–100 risk score.

---

## 🎯 Why This Project Matters

Operational inefficiency is a massive, hidden cost center. Research
consistently shows organizations lose 20–30% of revenue annually to
broken or inefficient processes. The challenge isn't a lack of data —
it's the lack of a system to interpret that data and translate it into
action.

What separates this system from a standard analytics project is the
**Process Variant Mining engine** — a capability that tools like
Celonis, ProcessMaker, and Signavio charge enterprise prices to provide.
This project builds a version from scratch, on a simple CSV, with full
explainability at every step.

👉 *Simulates real-world operational intelligence used in modern 
RevOps, BPM, and process excellence teams.*

---

## ⚙️ Features

### Core Analytics
- ⚡ **Bottleneck Detection** — Identifies the primary process constraint
  ranked by total time consumed and case volume, with impact percentage.
- ⏱️ **SLA Compliance Analysis** — Monitors performance against defined
  SLAs and surfaces breach patterns by task and case.
- ⏳ **Waiting vs. Processing Time Breakdown** — Statistical separation
  of active work time from idle queue time.
- ⚠️ **Exception Flow Detection** — Automated discovery of rework loops
  and sequence deviations.
- 📊 **Workflow Health Score (0–100)** — Executive-ready composite metric
  with AI-driven explainability detailing which factors drive the score.
- ⚙️ **Automation Opportunities** — Targeted recommendations with
  estimated ROI and time-reduction impact per suggestion.
- 🎯 **Risk Prediction** — Transparent 0–100 risk scores per task with
  plain-language explanations of contributing risk factors.

### Process Variant Mining (New)
- 🔀 **Variant Discovery** — Automatically groups cases by the sequence
  of steps they actually followed and labels each unique path as a variant.
- 🏷️ **Variant Classification** — Tags each variant as Conformant,
  Rework Loop, Skip, or Extended with a drift indicator showing how many
  steps it deviates from the standard path.
- 💸 **Waste Quantification** — Calculates recoverable hours per variant:
  how much time would be reclaimed if every deviant case followed the
  optimal path.
- 📉 **Variant Scoring** — Scores every path on avg cycle time, SLA
  breach rate, and wait ratio so you can prioritize which deviation to
  fix first.
- 🧠 **Business Insight Generator** — Produces three runtime-computed
  insights: conformance rate with extra hours per deviant case, total
  recoverable waste hours from rework, and SLA risk concentration by
  variant type.
- ⚡ **Smart Warning Engine** — Identifies the highest-risk deviant path
  using a normalized combined risk score and generates a forward-looking
  estimate of hours recovered if that path is fixed.

---

## 🧠 How It Works

**Simple View:**
```
Upload Workflow CSV  →  System Analyzes  →  Surface Variants + Bottlenecks →  Quantifies Waste  →  You Take Action
```

**Detailed Pipeline:**

```
📂 Data Ingestion          Upload raw CSV event logs (case_id, task, start_time, end_time, user)
        ↓
🔧 Preprocessing           Timestamp cleaning, duration calculation, waiting-time simulation
        ↓
🔀 Variant Mining        Fingerprint each case → group into variants → classify + score
        ↓
🔬 Discovery & Analysis    SLA compliance, bottleneck severity, variability scoring, exception detection
        ↓
🧠 Strategic Synthesis     Waste Quantification, recovery estimates, risk prediction
        ↓
📊 Streamlit Dashboard     Multi-tab executive interface with interactive Plotly visualizations
```

Each stage is modular and independently testable, making the system easy to extend with new analysis engines or data sources.

---

## 🖥️ Dashboard Preview

> *Dashboard screenshots are stored in the `assets/` directory.*

![Dashboard Overview](assets/overview.png)
*Workflow Health Score panel with executive-ready Smart Insights and risk flags.*

![Bottleneck Analysis](assets/bottleneck.png)
*Waiting vs. Processing time breakdown with step-level drill-down.*

![Smart Insights](assets/insights.png)
*Detailed AI-driven insights with Problem-Cause-Action-Impact chains.*


---

## 📊 Example Insights

**🔴 Bottleneck Alert**
> `Manager Approval` consumes an average of **707 minutes** per case —
> accounting for **63% of total workflow time**. Primary driver: queue
> wait time, not task complexity.

**🔀 Variant Intelligence**
> **3 distinct workflow paths detected** across 100 cases.
> Variant B (Rework Loop) affects 23 cases and adds **8.4 extra hours**
> per case vs the standard path — **193 total hours of recoverable
> operational waste.**

**⚡ Smart Warning**
> Variant B is your highest-risk deviant path. SLA breach rate: **67%**.
> If optimized to match the best-performing variant, this could reduce
> cycle time by **~11.2 hours per case** — ~258 hours recovered across
> 23 cases.

**⚠️ Risk Prediction**
> `Lead Reviewed` — **High Risk Score: 82/100**
> Extreme duration variability (σ = 340 mins) combined with high
> execution volume creates a statistically likely SLA breach point
> within the next operational cycle.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core application logic |
| Data Engine | Pandas | Vectorized event log processing |
| Visualization | Plotly | Interactive, high-fidelity charts |
| Interface | Streamlit | Multi-tab analytics dashboard |
| Analytics | Custom Statistical Engines | Variant mining, risk scoring, waste quantification |

---

## 📂 Project Structure

```
ai-workflow-intelligence/
│
├── dashboard/
│   └── app.py                  # Main Streamlit application (all tabs)
│
├── src/
│   ├── preprocessing.py        # Timestamp parsing, duration calc, SLA tagging
│   ├── bottleneck_detector.py  # Bottleneck ranking and impact quantification
│   ├── risk_predictor.py       # Statistical variability and risk scoring
│   ├── health_analyzer.py      # Composite 0–100 Workflow Health Score
│   ├── automation_engine.py    # Automation opportunity matching with ROI
│   ├── insight_engine.py       # Problem → Cause → Action → Impact chains
│   ├── context_analyzer.py     # Industry classification, task type detection
│   └── variant_miner.py        # Process variant mining, classification,
│                               # waste quantification, insight generation
│
├── tests/
│   └── test_variant_miner.py   # pytest unit tests for variant mining logic
│
├── data/                       # Input CSV workflow logs
├── assets/                     # Dashboard screenshots
└── requirements.txt            # Project dependencies
```

---

## ▶️ How to Run

**1. Clone the Repository**
```bash
git clone https://github.com/anamika-baruah/ai-workflow-intelligence.git
cd ai-workflow-intelligence
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the Dashboard**
```bash
streamlit run dashboard/app.py
```

**4. Upload Your Data**

| Column | Description |
|---|---|
| `case_id` | Unique identifier per workflow instance |
| `task` | Name of the process step |
| `start_time` | Task start timestamp |
| `end_time` | Task completion timestamp |
| `user` | Agent or system that ran the task |

> 💡 No data? Generate a sample dataset:
> ```bash
> python src/generate_dataset.py
> ```

---

## 🎯 Use Cases

- **Sales Pipeline Optimization** — Identify where leads stall, detect
  which pipeline variants are causing the most SLA breaches, and
  quantify the revenue impact of fixing the worst path.
- **Customer Support Operations** — Separate active resolution time from
  idle queue time; discover which ticket-handling variants are deviating
  from standard and by how many steps.
- **Business Process Governance** — Ensure workflows comply with stated
  SLAs; surface rework loops and skipped steps before they become
  systemic compliance issues.

---

## 👩‍💻 Author

**Anamika Baruah**  
*Final Year MCA Student*

Passionate about building intelligent systems at the intersection of
data analytics, AI, and real-world business problem solving. This
project reflects a conviction that the highest-value AI applications
don't just surface data — they help people make better decisions faster.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anamika-baruah/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/anamika-baruah)

---

*Built independently as part of an applied operational analytics
research project — exploring how process intelligence tooling can be
made accessible without enterprise-scale infrastructure.*

