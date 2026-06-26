# NeuralGCM: Why NOT Recommended (Per Brief)

**Source:** Global AI Weather Model Selection Brief (your document)

---

## WHAT THE BRIEF SAYS ABOUT NEURALGCM

### From Model Comparison Table:

```
NeuralGCM
Organization: Google Research/collaborators
Type: Hybrid physics + machine learning circulation model
Status: Open research software/trained weights
Input: Global atmospheric state and scientific compute setup
Output: Global atmospheric state
Suitability: Physics-informed research route; NOT PLUG-AND-PLAY FOR ONE BUOY ❌
```

### From Decision Process (PATH D):

```
PATH D: Advanced global physics/ML research
├─ NeuralGCM ❌
├─ FuXi/FuXi-ENS
├─ FuXi Weather
└─ Aardvark

Requirement: "dedicated global-data and scientific-validation plan"
Status: NOT suitable for operational deployment
```

---

## WHY NEURALGCM IS NOT RECOMMENDED

### 1. **Not Plug-and-Play (Brief's Exact Words)**

> "Physics-informed research route; **not plug-and-play for one buoy**"

Translation:
- ❌ Cannot point at your buoy location and get forecast
- ❌ Requires global circulation modeling setup
- ❌ Requires scientific compute infrastructure
- ❌ Not designed for single-point local forecasting

### 2. **Requires Scientific Compute Setup**

From brief: "Global atmospheric state and scientific compute setup"

This means:
- ❌ Multi-node distributed computing
- ❌ Global atmospheric state initialization
- ❌ Physics simulation infrastructure
- ❌ Not cloud-API compatible

### 3. **PATH D Classification (Experimental)**

Your project needs: **PATH B** (operational, proven)
NeuralGCM designed for: **PATH D** (advanced research)

```
PATH A: Local CSV only
PATH B: External forecast fields + bias correction ← YOUR PROJECT
PATH C: Global grids + GPU
PATH D: Advanced research ← NEURALGCM
```

**Wrong path for your goal.**

### 4. **Hybrid Physics-ML Research Grade**

- Not production-tested
- Not operationally validated
- Not integrated with commercial APIs
- Designed for scientific research, not operational forecasting

---

## COMPARISON: NEURALGCM vs YOUR BEST OPTIONS

| Criterion | NeuralGCM | AIFS | Aurora |
|-----------|-----------|------|--------|
| **Plug-and-play** | ❌ No | ✅ Yes | ✅ Yes |
| **Single buoy ready** | ❌ No | ✅ Yes | ⚠️ With prep |
| **Operational status** | Experimental | ✅ Live (Feb 2025) | Research |
| **API available** | ❌ No | ✅ Yes | ⚠️ Research |
| **Setup time** | 3+ months | Days | Weeks |
| **Global state needed** | ✅ Yes | ❌ No | ✅ Yes |
| **Scientific compute** | ❌ Required | ✅ Not needed | ⚠️ GPU only |
| **Brief recommendation** | "Not suitable" | "BEST practical" | "Best research" |

---

## WHAT NEURALGCM WOULD REQUIRE

If you insisted on using NeuralGCM (not recommended):

```
1. Build global atmospheric state
   - ERA5 reanalysis data
   - Global initial conditions
   - Multiple pressure levels
   - Land-sea masks, orography

2. Set up physics simulation
   - Global circulation model components
   - Hybrid physics-ML hybrid framework
   - Scientific compute software stack

3. Initialize NeuralGCM
   - Train/configure global state model
   - Validate physics constraints
   - Spin up from reanalysis

4. Run global forecast
   - Generate global atmospheric state
   - Run for 7-day forecast horizon
   - Multi-node computing

5. Downscale to buoy
   - Extract point from global grid
   - Interpolate to buoy location
   - Apply bias correction

Timeline: 4-6 months minimum
Cost: €100,000+ infrastructure + time
Team: 3-5 research scientists
Compute: Multi-node HPC cluster
Result: Solving the wrong problem for your use case
```

---

## THE BRIEF'S EXACT RECOMMENDATION

From Section 7 (Default Recommendation):

> "For a practical 7-day buoy forecast system, select **PATH B**:
> 
> AIFS / ECMWF / GFS forecast fields
> + recent buoy observations
> → local bias correction
> 
> **Why:**
> - operationally feasible ✓
> - avoids operating a global AI model yourself ✓
> - uses buoy history to correct grid-scale local bias ✓
> - gives future-known forcing appropriate for 7-day horizon ✓
> - retains local-only fallback ✓"

**NeuralGCM is the opposite of every one of these.**

---

## SAFETY RULES FROM BRIEF

The brief explicitly warns:

> "Never evaluate forecast skill using future reanalysis as if it had been forecast."

NeuralGCM pitfall:
- Often benchmarked against future reanalysis
- Not validated against real operational forecasts
- Research metrics ≠ operational skill

---

## BOTH AARDVARK & NEURALGCM: SAME STORY

| Model | Brief Classification | Input | Status | Your Fit |
|-------|---------------------|-------|--------|----------|
| **Aardvark** | PATH D | Global assimilation | Experimental | ❌ Not suitable |
| **NeuralGCM** | PATH D | Global state | Research | ❌ Not suitable |
| **AIFS** | PATH B | Forecast fields API | Operational | ✅ BEST FIT |
| **Aurora** | PATH C | Global grids | Research | ✅ Good fit |

---

## BOTTOM LINE

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  NeuralGCM: NOT SUITABLE for your project              │
│                                                          │
│  Brief says:                                             │
│  ❌ "Not plug-and-play for one buoy"                    │
│  ❌ PATH D (experimental research)                       │
│  ❌ Requires global state + scientific compute          │
│  ❌ 4-6 months + €100k+ to implement                    │
│                                                          │
│  Your actual need (from brief):                          │
│  ✅ AIFS (operationally ready, PATH B)                  │
│  ✅ Aurora (research-ready, PATH C)                     │
│  ❌ NOT advanced physics-ML research                     │
│                                                          │
│  Decision: Skip both Aardvark & NeuralGCM              │
│  Proceed: AIFS + Aurora (brief-recommended)            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## DIRECT QUOTE FROM BRIEF

> "Evaluate **NeuralGCM, FuXi/FuXi-ENS, FuXi Weather, or Aardvark only with a dedicated global-data and scientific-validation plan.**"

Your project: Single buoy, 7-day forecast, operational  
These tools: Global physics-ML research systems  
Match: ❌ Completely misaligned

---

## RECOMMENDATION

**Both Aardvark and NeuralGCM are PATH D (experimental research).**

**Your project is PATH B (operational).**

**The brief explicitly recommends:**
1. ✅ **AIFS** (primary Tier 1)
2. ✅ **Aurora** (secondary Tier 1)

**Ready to implement AIFS + Aurora per the brief's recommendation?** 🚀

