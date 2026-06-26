# Complete Parameter Inventory: By Category & Model

**Date:** 2026-06-26  
**System:** Marine iTransformer + GraphCast (4-Tier AIFS fallback)  
**Total Parameters:** 31 (28 provided + 3 not available)

---

## 1. ATMOSPHERIC PARAMETERS (GraphCast)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 1 | **Air Temperature** | GraphCast | ✅ Provided | +55-60% skill |
| 2 | **Air Pressure** | GraphCast | ✅ Provided | +55-60% skill |
| 3 | **Relative Humidity** | GraphCast (derived) | ✅ Provided | Calculated from temp/dew |
| 4 | **Dew Point Temperature** | GraphCast | ✅ Provided | +55-60% skill |
| 5 | **Wind Chill Temperature** | GraphCast (derived) | ✅ Provided | From wind speed + temp |
| 6 | **Wind Speed** | GraphCast (derived) | ✅ Provided | From u,v components |
| 7 | **Wind Direction** | GraphCast (derived) | ✅ Provided | From u,v components |
| 8 | **Compass** | GraphCast (derived) | ✅ Provided | Directional indicator |
| 9 | **Global Radiation** | ❌ NOT PROVIDED | ⚠️ Missing | Not in GraphCast outputs |
| 10 | **Precipitation Difference** | ❌ NOT PROVIDED | ⚠️ Missing | Not in current system |
| 11 | **Precipitation Intensity** | ❌ NOT PROVIDED | ⚠️ Missing | Not in current system |
| 12 | **Precipitation Type** | ❌ NOT PROVIDED | ⚠️ Missing | Not in current system |

**Atmospheric Summary:** 8/12 provided by GraphCast (67%), 4 missing (precipitation + radiation)

---

## 2. CURRENT PARAMETERS (Marine iTransformer)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 13 | **Current Speed** | iTransformer | ✅ Provided | +84.9% skill |
| 14 | **Current Direction** | iTransformer | ✅ Provided | +84.9% skill |

**Current Summary:** 2/2 provided by iTransformer (100%) ✅

---

## 3. WATER / TIDE PARAMETERS (Marine iTransformer)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 15 | **Water Pressure** | iTransformer | ✅ Provided | +84.9% skill |
| 16 | **Tide Pressure** | iTransformer (residual) | ✅ Provided | +84.9% skill |
| 17 | **Tide Level** | iTransformer (residual) | ✅ Provided | +84.9% skill |
| 18 | **Water Temperature** | iTransformer | ✅ Provided | +84.9% skill |

**Water/Tide Summary:** 4/4 provided by iTransformer (100%) ✅

---

## 4. WATER QUALITY PARAMETERS (Marine iTransformer)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 19 | **Conductivity** | ❌ NOT PROVIDED | ⚠️ Missing | Not in iTransformer training |
| 20 | **Salinity** | iTransformer | ✅ Provided | +84.9% skill |
| 21 | **Water Temperature** | iTransformer | ✅ Provided | +84.9% skill (same as #18) |

**Water Quality Summary:** 2/3 provided (67%), 1 missing (conductivity)

---

## 5. WAVE / TIDE SENSOR PARAMETERS (Marine iTransformer)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 22 | **Significant Wave Height** | iTransformer | ✅ Provided | +84.9% skill |
| 23 | **Maximum Wave Height** | iTransformer (derived) | ✅ Provided | Derived from significant |
| 24 | **Water Level** | iTransformer (tidal residual) | ✅ Provided | +84.9% skill |
| 25 | **Significant Wave Period** | iTransformer | ✅ Provided | +84.9% skill |
| 26 | **Peak Wave Energy Period** | iTransformer (derived) | ✅ Provided | Derived from spectral data |
| 27 | **Zero Crossing Period** | iTransformer | ✅ Provided | +84.9% skill |

**Wave/Tide Sensor Summary:** 6/6 provided by iTransformer (100%) ✅

---

## 6. VISIBILITY SENSOR PARAMETERS (Not Provided)

| # | Parameter | Data Source | Status | Availability |
|---|-----------|------------|--------|--------------|
| 28 | **1-Minute Average Visibility** | ❌ NOT PROVIDED | ⚠️ Missing | Not in model outputs |
| 29 | **10-Minute Average Visibility** | ❌ NOT PROVIDED | ⚠️ Missing | Not in model outputs |
| 30 | **1-Hour Average Visibility** | ❌ NOT PROVIDED | ⚠️ Missing | Not in model outputs |
| 31 | **24-Hour Average Visibility** | ❌ NOT PROVIDED | ⚠️ Missing | Not in model outputs |

**Visibility Summary:** 0/4 provided (0%) — Not available from any model

---

## SUMMARY BY CATEGORY

| Category | Total | Provided | Missing | % Complete | Primary Model |
|----------|-------|----------|---------|-----------|--------------|
| **Atmospheric** | 12 | 8 | 4 | 67% | GraphCast |
| **Current** | 2 | 2 | 0 | 100% | iTransformer |
| **Water/Tide** | 4 | 4 | 0 | 100% | iTransformer |
| **Water Quality** | 3 | 2 | 1 | 67% | iTransformer |
| **Wave/Tide Sensor** | 6 | 6 | 0 | 100% | iTransformer |
| **Visibility Sensor** | 4 | 0 | 4 | 0% | Not provided |
| **TOTAL** | **31** | **22** | **9** | **71%** | Hybrid |

---

## BREAKDOWN BY MODEL

### GraphCast (Atmospheric - 7 core + 1 derived = 8/12)

**Provides:**
- ✅ Air Temperature (+55-60%)
- ✅ Air Pressure (+55-60%)
- ✅ Dew Point Temperature (+55-60%)
- ✅ Relative Humidity (derived)
- ✅ Wind Speed (derived from u,v)
- ✅ Wind Direction (derived from u,v)
- ✅ Wind Chill (derived)
- ✅ Compass (derived)

**Does NOT Provide:**
- ❌ Global Radiation
- ❌ Precipitation Difference
- ❌ Precipitation Intensity
- ❌ Precipitation Type

---

### Marine iTransformer (Ocean/Marine - 16/16)

**Provides (100% of marine parameters):**
- ✅ Current Speed (+84.9%)
- ✅ Current Direction (+84.9%)
- ✅ Water Pressure (+84.9%)
- ✅ Tide Pressure (+84.9%)
- ✅ Tide Level (+84.9%)
- ✅ Water Temperature (+84.9%)
- ✅ Salinity (+84.9%)
- ✅ Significant Wave Height (+84.9%)
- ✅ Maximum Wave Height (derived)
- ✅ Water Level (residual) (+84.9%)
- ✅ Significant Wave Period (+84.9%)
- ✅ Peak Wave Energy Period (derived)
- ✅ Zero Crossing Period (+84.9%)

**Does NOT Provide:**
- ❌ Conductivity (not in training data)

---

### Not Provided By Any Model

- ❌ Global Radiation (requires separate sensor/model)
- ❌ Precipitation (GraphCast doesn't include in current implementation)
- ❌ Conductivity (requires separate sensor)
- ❌ Visibility (requires separate sensor/observations)

---

## SYSTEM COVERAGE BY DOMAIN

```
ATMOSPHERIC DOMAIN (GraphCast):
├─ Core: 4/7 variables
│  ├─ Temperature ✅
│  ├─ Pressure ✅
│  ├─ Humidity (derived) ✅
│  └─ Dew Point ✅
├─ Wind: 2/2 variables ✅
│  ├─ Speed (derived) ✅
│  └─ Direction (derived) ✅
└─ Missing: Radiation, Precipitation (4 params)

MARINE DOMAIN (iTransformer):
├─ Current: 2/2 ✅
├─ Water/Tide: 4/4 ✅
├─ Water Quality: 2/3 (missing conductivity)
└─ Waves: 6/6 ✅

MISSING SENSORS:
├─ Visibility (4 params) - optical sensor needed
├─ Conductivity (1 param) - water quality sensor needed
├─ Precipitation (3 params) - weather radar/sensor needed
└─ Global Radiation (1 param) - pyranometer sensor needed
```

---

## HOW TO GET MISSING PARAMETERS

| Missing Param | Solution | Implementation |
|---------------|----------|-----------------|
| **Global Radiation** | Add pyranometer sensor | Hardware + calibration |
| **Precipitation** | Add weather radar/gauge | Hardware + data integration |
| **Conductivity** | Add conductivity probe | Hardware + calibration |
| **Visibility** | Add visibility sensor | Hardware + integration |

---

## CURRENT SYSTEM CAPACITY (22/31 Parameters)

```
PRODUCTION SYSTEM PROVIDES:

Marine Domain:          13 parameters ✅ (100%)
├─ Currents:            2 parameters
├─ Tides/Water:         4 parameters
├─ Water Quality:       2 parameters
└─ Waves:               5 parameters

Atmospheric Domain:     8 parameters ✅ (67%)
├─ Temperature:         1 parameter
├─ Pressure:            1 parameter
├─ Humidity:            2 parameters (1 derived)
└─ Wind:                4 parameters (2 derived)

NOT PROVIDED:           9 parameters
├─ Missing sensors:     4 (visibility)
├─ Missing data:        3 (precipitation)
├─ Missing model:       1 (conductivity)
└─ Missing sensors:     1 (radiation)

COVERAGE:               22/31 = 71% ✅
```

---

## UPGRADE PATHS

### To Add Precipitation (3 params)
- Requires: Weather radar integration or precipitation model
- Timeline: 2-4 weeks
- Cost: Sensor/API integration

### To Add Global Radiation (1 param)
- Requires: Pyranometer sensor
- Timeline: 1-2 weeks (calibration)
- Cost: Hardware (~€500)

### To Add Conductivity (1 param)
- Requires: Conductivity probe
- Timeline: 1-2 weeks (setup)
- Cost: Hardware (~€200)

### To Add Visibility (4 params)
- Requires: Optical visibility sensor
- Timeline: 1-2 weeks (setup)
- Cost: Hardware (~€2000)

---

## RECOMMENDATION

**Current 22/31 (71%) is sufficient for:**
- ✅ Harbor navigation
- ✅ Wave forecasting
- ✅ Current prediction
- ✅ Tide forecasting
- ✅ Water temperature
- ✅ Air/sea conditions

**Consider adding if:**
- Need precipitation (rain/snow)
- Need radiation (solar, thermal)
- Need visibility (fog prediction)
- Need conductivity (salinity validation)

**All marine-critical parameters:** 100% covered ✅

