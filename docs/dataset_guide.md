# Dataset Guide — Case Western Reserve University (CWRU) Bearing Data Center

## 1. Selected Dataset Overview

**BearingSense** is designed around the internationally recognized benchmark dataset published by the **Case Western Reserve University (CWRU) Bearing Data Center**:

- **Official Source Portal**: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)
- **Bearing Specifications Page**: [CWRU Bearing Information](https://engineering.case.edu/bearingdatacenter/bearing-information)
- **Apparatus & Test Procedures**: [CWRU Apparatus & Procedures](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures)
- **Normal Baseline Data**: [CWRU Normal Baseline Data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
- **12k Drive End Fault Data**: [CWRU 12k Drive End Bearing Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)
- **Reference Benchmark Study**: Smith, W. A., & Randall, R. B. (2015). *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study*. Mechanical Systems and Signal Processing, 64–65, 100–131.

### Physical Experimental Setup
- **Test Stand**: A 2 HP Reliance Electric induction motor driving a shaft coupled via a torque transducer/encoder to a dynamometer load.
- **Test Bearings**:
  - **Drive End (DE)**: SKF 6205-2RS JEM deep-groove ball bearing (9 balls, ball diameter $0.3126\text{ in}$, pitch diameter $1.537\text{ in}$, contact angle $0^\circ$).
  - **Fan End (FE)**: SKF 6203-2RS JEM deep-groove ball bearing (8 balls, ball diameter $0.2656\text{ in}$, pitch diameter $1.122\text{ in}$, contact angle $0^\circ$).
- **Seeded Faults**: Single-point localized defects were introduced separately on the **Inner Raceway**, **Outer Raceway** (centered at 6 o'clock in the load zone), and **Rolling Element (Ball)** using Electro-Discharge Machining (EDM), with fault diameters of $0.007\text{ in}$, $0.014\text{ in}$, and $0.021\text{ in}$.
- **Operating Loads & Speeds**: Recorded across four motor loads: $0\text{ HP}$ ($1797\text{ RPM}$), $1\text{ HP}$ ($1772\text{ RPM}$), $2\text{ HP}$ ($1750\text{ RPM}$), and $3\text{ HP}$ ($1730\text{ RPM}$).
- **Sampling Frequency**: $12,000\text{ Hz}$.

---

## 2. Recommended CWRU Files to Download for Full Multi-Load Evaluation

To evaluate BearingSense across all four motor loads ($0, 1, 2, 3\text{ HP}$) so that each load serves as an independent physical run in Stratified Group K-Fold Cross-Validation, download the following standard 12 kHz Drive End `.mat` files from the CWRU portal and place them in `data/raw/`:

| Fault Condition Class | Fault Diameter | 0 HP (1797 RPM) | 1 HP (1772 RPM) | 2 HP (1750 RPM) | 3 HP (1730 RPM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | None ($0.000\text{ in}$) | `97.mat` | `98.mat` | `99.mat` | `100.mat` |
| **Inner_Race_Fault** | $0.007\text{ in}$ (`IR007`) | `105.mat` | `106.mat` | `107.mat` | `108.mat` |
| **Ball_Fault** | $0.007\text{ in}$ (`B007`) | `118.mat` | `119.mat` | `120.mat` | `121.mat` |
| **Outer_Race_Fault** | $0.007\text{ in}$ (`OR007@6`) | `130.mat` | `131.mat` | `132.mat` | `133.mat` |
| **Inner_Race_Fault** | $0.014\text{ in}$ (`IR014`) | `169.mat` | `170.mat` | `171.mat` | `172.mat` |
| **Ball_Fault** | $0.014\text{ in}$ (`B014`) | `185.mat` | `186.mat` | `187.mat` | `188.mat` |
| **Outer_Race_Fault** | $0.014\text{ in}$ (`OR014@6`) | `197.mat` | `198.mat` | `199.mat` | `200.mat` |
| **Inner_Race_Fault** | $0.021\text{ in}$ (`IR021`) | `209.mat` | `210.mat` | `211.mat` | `212.mat` |
| **Ball_Fault** | $0.021\text{ in}$ (`B021`) | `222.mat` | `223.mat` | `224.mat` | `225.mat` |
| **Outer_Race_Fault** | $0.021\text{ in}$ (`OR021@6`) | `234.mat` | `235.mat` | `236.mat` | `237.mat` |

`src/bearingsense/data_loader.py` includes a built-in `CWRU_FILE_CATALOG` dictionary covering all 40 file IDs listed above (`97.mat` through `237.mat`).
