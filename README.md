# DriftX-Forensics
### Cross-Stage Drift Detection and Root-Cause Localization for MLOps

DriftX-Forensics is an MLOps observability and forensic investigation platform designed to analyze model-performance changes, investigate potential sources of drift across pipeline stages, and evaluate remediation recommendations using safety-oriented decision gates.

## 1. Project Overview

Machine-learning systems can degrade when input data, feature processing, or operating conditions change. Detecting a change is not always enough; practitioners also need evidence to investigate its likely origin and assess whether a proposed model update is safe.

DriftX-Forensics brings these activities into one Streamlit application, with incident investigation, monitoring, analytics, data-integrity checks, and incident history.

## 2. Project Objectives

- Investigate potential drift across multiple ML pipeline stages.
- Analyze evidence and identify potential root causes.
- Compare champion and challenger model metrics.
- Generate confidence-aware remediation recommendations.
- Evaluate model updates using configurable safety-gate rules.
- Maintain incident records for investigation and review.

## 3. Key Features

- **Forensics:** Investigate drift evidence and potential origins.
- **Incident Workbench:** Examine incidents and remediation options.
- **Incident History:** Review recorded incident details.
- **System Health:** Inspect available system-health information.
- **Incident Monitor:** View incident-monitoring information.
- **Incident Analytics:** Explore available incident summaries.
- **Data Integrity:** Inspect data-quality and integrity indicators.
- **Safety-Gate Evaluation:** Assess available model metrics against configured decision rules.
- **Reporting:** Generate investigation reports where supported by the application.

Feature availability and results depend on the configured data and implemented evaluation logic.

## 4. System Architecture

The platform follows a modular architecture:

1. **Input and Evidence Collection:** Load available datasets, metrics, and incident evidence.
2. **Drift Investigation:** Analyze evidence and evaluate potential cross-stage origins.
3. **Root-Cause Assessment:** Identify the most likely source based on available evidence.
4. **Incident Orchestration:** Coordinate investigation and remediation evaluation.
5. **Safety Evaluation:** Assess champion and challenger metrics against configured rules.
6. **Reporting and Persistence:** Store supported incident records and prepare reports.
7. **Streamlit Dashboard:** Present investigation, monitoring, analytics, and integrity information.

### Workflow

Evidence Collection → Drift Investigation → Root-Cause Assessment → Remediation Recommendation → Safety-Gate Evaluation → Incident Report

A predicted origin is an investigative assessment and should not be interpreted as proof of causation.

## 5. Technology Stack

- **Programming Language:** Python
- **User Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn, SciPy
- **Configuration:** PyYAML
- **Report Generation:** ReportLab, where implemented
- **Testing:** Python unittest

## 6. Project Structure

```text
DriftX-Forensics/
├── app/
│   ├── dashboard.py
│   └── pages/
├── src/
│   └── driftx/
│       └── evaluation/
├── results/
├── scripts/
├── tests/
├── .streamlit/
├── requirements.txt
├── runtime.txt
└── README.md
```

## 7. Installation and Setup

Python 3.11 is the intended deployment runtime.

### Step 1: Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### Step 2: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Running the Application

Run the following command from the project root:

```powershell
python -m streamlit run app/dashboard.py
```

Open the local URL provided by Streamlit. The default address is:

`http://localhost:8501`

## 9. Running Tests

Run the regression tests using:

```powershell
$env:PYTHONPATH = "."
python -m unittest discover -s tests -v
```

A passing test suite verifies only the behaviors covered by the tests. Run the tests after modifying the investigation engine, incident orchestration, or safety-gate logic.

## 10. Research Evaluation

The project includes evaluation scripts and benchmark artifacts under the `scripts/` and `results/` directories.

A reproducible evaluation should document:

- Dataset and scenario definitions.
- Baseline and proposed methods.
- Evaluation metrics and their definitions.
- Experimental configuration and random seeds.
- Individual run results and aggregate results.
- Limitations and failure cases.

Only report numerical results that can be reproduced from the actual benchmark artifacts. Repeating fixed scenarios should not be described as independent randomized evaluation unless the experimental conditions genuinely vary.

## 11. Safety-Gate Design

The safety gate evaluates available champion and challenger metrics using configured acceptance rules. It is intended to support controlled review rather than independently authorize production deployment.

A passing gate or remediation recommendation does not guarantee model safety. Production deployment requires appropriate validation, monitoring, rollback planning, and human authorization.

## 12. Limitations

- Root-cause outputs may not establish causal relationships.
- Results depend on input-data quality, available metrics, and configured thresholds.
- Synthetic or fixed benchmark scenarios may not represent real production environments.
- Automated tests cover only explicitly tested behaviors.
- Public deployment and external integrations must be verified separately.

## 13. Future Enhancements

- Evaluate the system using independently generated and real-world drift scenarios.
- Compare root-cause localization against documented ground truth.
- Add confidence calibration and uncertainty analysis.
- Evaluate false-positive and false-negative rates.
- Improve reproducibility through versioned datasets and experiment configurations.
- Explore deployment monitoring and controlled rollback integration.

## 14. Project Status

DriftX-Forensics is a research-oriented prototype for MLOps drift investigation and safety-aware remediation assessment.

Local execution and automated checks should be verified in the target environment. A public deployment should be considered available only after deployment succeeds and accessibility is confirmed.

## 15. Responsible Reporting

Distinguish between observed drift, predicted root cause, recommended remediation, and authorized production action. Correlation and a safety-gate result should not be interpreted as proof of causation or a guarantee of model safety.
