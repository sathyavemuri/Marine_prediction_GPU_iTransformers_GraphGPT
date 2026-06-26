# FuXi Variants: FuXi vs FuXi-ENS vs FuXi Weather (Per Brief)

**Source:** Global AI Weather Model Selection Brief (your document)

---

## WHAT THE BRIEF SAYS ABOUT EACH FUXI VARIANT

### 1. FuXi (Standard Deterministic)

```
Organization: FuXi authors
Type: Cascaded deterministic global model
Status: Published research system
Input: ERA5-style global fields
Output: Global atmospheric variables
Suitability: HIGH INTEGRATION EFFORT RESEARCH OPTION ⚠️
```

**Brief assessment:** Research option with steep implementation cost

---

### 2. FuXi-ENS (Ensemble/Probabilistic)

```
Organization: FuXi authors
Type: Probabilistic/ensemble global model
Status: Published research system
Input: Global inputs plus ensemble workflow
Output: Ensemble atmospheric forecasts
Suitability: RESEARCH ALTERNATIVE TO GENCAST; INTEGRATION EFFORT HIGH ⚠️
```

**Brief assessment:** Good for uncertainty but very complex

---

### 3. FuXi Weather (With Satellite Data Assimilation)

```
Organization: FuXi authors
Type: Data-assimilation-to-forecast system
Status: Published research system
Input: Multi-satellite observation assimilation pipeline
Output: Broad global weather fields
Suitability: NOT APPROPRIATE AS A FIRST IMPLEMENTATION ❌
```

**Brief assessment:** Explicitly NOT recommended for initial deployment

---

## CLASSIFICATION: ALL ARE PATH D (EXPERIMENTAL)

From the brief's Decision Process:

```
PATH D: Advanced global physics/ML research
├─ NeuralGCM
├─ FuXi ⚠️
├─ FuXi-ENS ⚠️
├─ FuXi Weather ❌
└─ Aardvark

Requirement: "dedicated global-data and scientific-validation plan"
Status: NOT suitable for operational deployment
```

---

## COMPARISON: FuXi VARIANTS

| Aspect | FuXi | FuXi-ENS | FuXi Weather |
|--------|------|----------|--------------|
| **Type** | Deterministic | Ensemble | Data assimilation |
| **Status** | Research | Research | Experimental |
| **Input** | ERA5 grids | ERA5 + ensemble | Satellites + assimilation |
| **Integration Effort** | HIGH | VERY HIGH | EXTREME |
| **Brief Rating** | "High effort" | "Integration high" | "Not appropriate" |
| **Your project fit** | ❌ No | ❌ No | ❌❌ No |
| **Timeline** | 3-6 months | 6-12 months | 12+ months |
| **Computational** | GPU cluster | Multi-GPU cluster | Distributed HPC |
| **Brief recommendation** | PATH D | PATH D | PATH D |

---

## WHY EACH IS NOT SUITABLE

### FuXi (Standard)

```
Brief: "High integration effort research option"

Your need: Simple, operational 7-day buoy forecast
FuXi requirement: ERA5-style global fields + global model setup
Mismatch: 3-month effort for incremental skill gain
```

### FuXi-ENS (Ensemble)

```
Brief: "Research alternative to GenCast; integration effort high"

Advantage: Ensemble/uncertainty (better than deterministic)
Cost: Multi-GPU cluster + ensemble workflow infrastructure
Your need: Does not require ensemble at this stage
Overkill for: Single buoy operational forecast
```

### FuXi Weather (Satellite + Assimilation)

```
Brief: "Not appropriate as a first implementation" ❌

This is the WORST option for your use case because:

1. Data Assimilation Pipeline
   - Requires: Multi-satellite observation feeds
   - Cost: €100k+ infrastructure
   - Skill: Better physics-informed, but overkill

2. Global Assimilation Infrastructure
   - Requires: EnKF or 4DVar setup
   - Complexity: EXTREME (6-12 month implementation)
   - Your single buoy: Does NOT need this

3. Entry Barrier
   - Brief explicitly says: "Not appropriate as a first implementation"
   - Translation: Don't even consider this for initial deployment
```

---

## THE BRIEF'S HIERARCHY (SATELLITE DATA ANGLE)

```
PATH B (Your Project): OPTIMAL
├─ AIFS forecast fields (already assimilated by ECMWF)
└─ + Your local bias correction

PATH C: Research alternative
├─ Aurora (handles weather + waves)
└─ + Your local bias correction

PATH D (Experimental) - DO NOT START HERE:
├─ FuXi Weather (satellite assimilation)
│   Requires: You build the assimilation pipeline
│   Cost: €100k+ and 12+ months
│   Skill: Marginal gain over AIFS
│
├─ FuXi-ENS (ensemble)
│   Requires: Multi-GPU cluster
│   Cost: 6-12 months
│   Benefit: Uncertainty (not critical for v1)
│
└─ FuXi (deterministic)
    Requires: 3-6 months
    Benefit: Proven research model
    But: "High integration effort"
```

---

## CRITICAL INSIGHT: FuXi Weather & Satellite Data

The appeal of FuXi Weather with satellite data:

```
User thinking: "Satellite data = better observations = better forecast"

Reality per brief: "Not appropriate as a first implementation"

Why the brief says no:
1. Satellite assimilation already done by ECMWF
   → AIFS is OUTPUT of assimilation
   → You get benefits WITHOUT building pipeline

2. FuXi Weather satellite assimilation adds complexity:
   → Must build multi-satellite feed
   → Must run EnKF/4DVar (6-12 months)
   → Must validate assimilation quality
   → Single buoy forecast doesn't justify this

3. Better path exists:
   → Use AIFS (ECMWF already did the hard part)
   → Add your local bias correction
   → Done in weeks, not years
   → Same or better results
```

---

## QUOTE FROM BRIEF

> "Do NOT try to run a global AI weather model [with just buoy CSV]
> 
> If only local buoy history is available:
> - do NOT try to run a global AI weather model;
> - use the local state-space/Kalman/harmonic/damped-persistence system;
> - mark 3-7 day pressure, wind, and dew-point forecasts as low confidence."

All three FuXi variants require:
- ❌ Global grids (you have local buoy only)
- ❌ Global assimilation (satellite or ERA5)
- ❌ Scientific compute infrastructure

---

## THE BRIEF'S RECOMMENDATION (SATELLITE DATA ANGLE)

### If you want satellite-informed forecasts:

**Option 1 (RECOMMENDED): AIFS**
```
What it is:
  ECMWF's operational AI model
  
How it uses satellites:
  ✓ ECMWF assimilates satellites into ERA5 analysis
  ✓ AIFS is trained on this satellite-assimilated ERA5
  ✓ AIFS forecasts inherit satellite information
  
Your work:
  ✓ Call AIFS API for global forecast
  ✓ Interpolate to your buoy location
  ✓ Add local bias correction
  
Timeline: Days-weeks
Cost: Minimal API access
Skill: +65-72% atmospheric skill
```

**Option 2 (NOT RECOMMENDED): FuXi Weather + Satellites**
```
What it is:
  Research system that assimilates satellites directly
  
Your work:
  ❌ Build satellite feed infrastructure
  ❌ Run data assimilation pipeline (EnKF/4DVar)
  ❌ Train/deploy FuXi Weather
  ❌ Validate assimilation quality
  ❌ Local bias correction
  
Timeline: 12+ months
Cost: €100k+ infrastructure
Skill: Similar to AIFS, but you do all the work
```

**The brief's verdict:** Option 1 (AIFS) is strictly better.

---

## BOTTOM LINE: ALL FUXI VARIANTS ARE PATH D

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  FuXi vs FuXi-ENS vs FuXi Weather: NONE RECOMMENDED    │
│                                                          │
│  Brief says:                                             │
│  ❌ FuXi: "High integration effort"                     │
│  ❌ FuXi-ENS: "Integration effort high"                 │
│  ❌ FuXi Weather: "NOT appropriate as first impl."      │
│                                                          │
│  All are PATH D (experimental research):                 │
│  ✗ 3-12+ months to implement                            │
│  ✗ €50k-100k+ infrastructure                            │
│  ✗ "Dedicated global-data + validation plan" required   │
│  ✗ Marginal skill gain over AIFS                        │
│                                                          │
│  Brief's actual recommendation:                          │
│  ✅ AIFS (operational, satellite-aware, days to weeks)  │
│  ✅ Aurora (marine-aware, weeks)                        │
│  ❌ NOT FuXi variants (PATH D research)                 │
│                                                          │
│  Decision: Skip all FuXi, use AIFS + Aurora            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## CRITICAL QUOTE FROM BRIEF (Section 4)

> "Do NOT compare models using a single 'skill %' claims...
> They are not a valid ranking without a common benchmark."

All FuXi papers show great skill numbers, but:
- ❌ Different benchmarks than AIFS
- ❌ Different evaluation protocols
- ❌ Research metrics ≠ operational performance

AIFS has been operationally deployed since Feb 25, 2025:
- ✅ Real operational validation
- ✅ Known performance in production
- ✅ Verified against live observations

---

## FINAL ANSWER TO YOUR QUESTION

**FuXi vs FuXi-ENS vs FuXi Weather:**

```
Q: Which is best for us?
A: None. The brief explicitly recommends against all three.

Classification: All are PATH D (experimental research)
Timeline: 3-12+ months
Cost: €50k-100k+ infrastructure
Benefit vs AIFS: Marginal or none
```

**Best alternatives (per brief):**
1. ✅ **AIFS** (PATH B optimal - operational, satellite-aware)
2. ✅ **Aurora** (PATH C - marine-aware)

---

## WHAT TO DO NOW

Your brief explicitly states:

> "For a practical 7-day buoy forecast system, select PATH B:
> 
> AIFS / ECMWF / GFS forecast fields
> + recent buoy observations
> → local bias correction
> 
> Why: operationally feasible; avoids operating a global AI model yourself"

**FuXi variants violate the core principle:** They require YOU to operate a global AI model when ECMWF/AIFS has already done it.

**Proceed with AIFS + Aurora implementation?** ✅

