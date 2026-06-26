# CSV Field to Parameter Mapping Table

**Source CSV:** marine_data_120days_1min.csv  
**Status:** All 31 parameters present in your original data!

---

| # | Category | Corresponding Parameter | CSV Field Name | Model Forecast | CSV Status |
|---|----------|------------------------|-----------------|-----------------|-----------|
| 1 | **Atmospheric** | Air Temperature | air_temp_c | GraphCast | ✅ Present |
| 2 | **Atmospheric** | Air Pressure | air_pressure_hpa | GraphCast | ✅ Present |
| 3 | **Atmospheric** | Relative Humidity | relative_humidity_pct | GraphCast (Derived) | ✅ Present |
| 4 | **Atmospheric** | Dew Point Temperature | dew_point_c | GraphCast | ✅ Present |
| 5 | **Atmospheric** | Wind Chill Temperature | wind_chill_c | GraphCast (Derived) | ✅ Present |
| 6 | **Atmospheric** | Wind Speed | wind_speed_ms | GraphCast (Derived) | ✅ Present |
| 7 | **Atmospheric** | Wind Direction | wind_direction_deg | GraphCast (Derived) | ✅ Present |
| 8 | **Atmospheric** | Compass | compass_deg | GraphCast (Derived) | ✅ Present |
| 9 | **Atmospheric** | Global Radiation | global_radiation_wm2 | ⚠️ Not Forecasted | ✅ Present |
| 10 | **Atmospheric** | Precipitation Difference | precip_diff_mm | ⚠️ Not Forecasted | ✅ Present |
| 11 | **Atmospheric** | Precipitation Intensity | precip_intensity_mmh | ⚠️ Not Forecasted | ✅ Present |
| 12 | **Atmospheric** | Precipitation Type | precip_type | ⚠️ Not Forecasted | ✅ Present |
| 13 | **Current** | Current Speed | current_speed_ms | iTransformer | ✅ Present |
| 14 | **Current** | Current Direction | current_direction_deg | iTransformer | ✅ Present |
| 15 | **Water / Tide** | Water Pressure | water_pressure_dbar | iTransformer | ✅ Present |
| 16 | **Water / Tide** | Tide Pressure | tide_pressure_dbar | iTransformer | ✅ Present |
| 17 | **Water / Tide** | Tide Level | tidal_level_m | iTransformer | ✅ Present |
| 18 | **Water / Tide** | Water Temperature | water_temp_c | iTransformer | ✅ Present |
| 19 | **Water Quality** | Conductivity | conductivity_mscm | ⚠️ Not Forecasted | ✅ Present |
| 20 | **Water Quality** | Salinity | salinity_psu | iTransformer | ✅ Present |
| 21 | **Water Quality** | Water Temperature | water_temp_quality_c | iTransformer | ✅ Present |
| 22 | **Wave / Tide Sensor** | Significant Wave Height | significant_wave_height_m | iTransformer | ✅ Present |
| 23 | **Wave / Tide Sensor** | Maximum Wave Height | max_wave_height_m | iTransformer | ✅ Present |
| 24 | **Wave / Tide Sensor** | Water Level | water_level_m | iTransformer | ✅ Present |
| 25 | **Wave / Tide Sensor** | Significant Wave Period | significant_wave_period_s | iTransformer | ✅ Present |
| 26 | **Wave / Tide Sensor** | Peak Wave Energy Period | peak_wave_period_s | iTransformer | ✅ Present |
| 27 | **Wave / Tide Sensor** | Zero Crossing Period | zero_crossing_period_s | iTransformer | ✅ Present |
| 28 | **Visibility Sensor** | 1-Minute Average Visibility | visibility_1min_km | ⚠️ Not Forecasted | ✅ Present |
| 29 | **Visibility Sensor** | 10-Minute Average Visibility | visibility_10min_km | ⚠️ Not Forecasted | ✅ Present |
| 30 | **Visibility Sensor** | 1-Hour Average Visibility | visibility_1hr_km | ⚠️ Not Forecasted | ✅ Present |
| 31 | **Visibility Sensor** | 24-Hour Average Visibility | visibility_24hr_km | ⚠️ Not Forecasted | ✅ Present |

---

## Summary

```
TOTAL CSV FIELDS:          31/31 (100%) ✅
├─ Actively Forecasted:    22/31 (71%)
├─ Present but NOT Forecasted: 9/31 (29%)
│  ├─ Global Radiation (1)
│  ├─ Precipitation (3)
│  ├─ Conductivity (1)
│  └─ Visibility (4)
└─ Missing from CSV:       0/31 (0%)

STATUS: YOUR CSV HAS EVERYTHING! ✅✅✅
```

---

## Why Parameters Are Not Forecasted (Even Though in CSV)

### Global Radiation ⚠️
| Issue | Reason | Solution |
|-------|--------|----------|
| **CSV Status** | ✅ Present in data | Training data available |
| **Model Status** | ❌ Not forecasted | GraphCast doesn't output radiation |
| **Why** | GraphCast trained on ERA5 (no radiation) | Would need separate model |
| **Solution** | Add solar radiation model or sensor API | 2-4 weeks |

### Precipitation (3 params) ⚠️
| Issue | Reason | Solution |
|-------|--------|----------|
| **CSV Status** | ✅ Present in data | Training data available |
| **Model Status** | ❌ Not forecasted | GraphCast doesn't include precipitation |
| **Why** | GraphCast not trained on precipitation outputs | ERA5 has it, but not GraphCast |
| **Solution** | Add rainfall prediction model | 2-4 weeks |

### Conductivity ⚠️
| Issue | Reason | Solution |
|-------|--------|----------|
| **CSV Status** | ✅ Present in data | Training data available |
| **Model Status** | ❌ Not forecasted | iTransformer not trained with it |
| **Why** | iTransformer focused on marine dynamics, not water quality | Could retrain with it |
| **Solution** | Retrain iTransformer including conductivity | 2-4 weeks |

### Visibility (4 params) ⚠️
| Issue | Reason | Solution |
|-------|--------|----------|
| **CSV Status** | ✅ Present in data | Training data available |
| **Model Status** | ❌ Not forecasted | Neither model outputs visibility |
| **Why** | Visibility requires optical/atmospheric model | Not in GraphCast or iTransformer |
| **Solution** | Add visibility prediction model | 2-4 weeks |

---

## Key Finding

### The Real Issue:

**It's NOT that these parameters are missing from your CSV.**

**It's that the models (GraphCast + iTransformer) don't forecast them.**

```
Your CSV Data:              ✅ 31/31 complete
GraphCast Output:           8/31 parameters
iTransformer Output:        14/31 parameters
Combined Forecast:          22/31 parameters

Missing 9 parameters = Models don't output them
          (NOT missing from CSV!)
```

---

## What This Means

### You Could Add These Forecasts:

| Parameter | Effort | Timeline | Result |
|-----------|--------|----------|--------|
| **Conductivity** | ⭐ Low | 2 weeks | Retrain iTransformer with conductivity |
| **Global Radiation** | ⭐⭐ Medium | 2-4 weeks | Add solar model or API |
| **Precipitation** | ⭐⭐⭐ High | 3-4 weeks | Add rainfall model |
| **Visibility** | ⭐⭐⭐ High | 3-4 weeks | Add optical/atmospheric model |

### Recommended Next Steps:

1. ✅ **Deploy current system** (22/31 forecasted, working)
2. ⭐ **Add Conductivity** (easiest, good water quality indicator)
3. ⭐ **Add Precipitation** (operational use, storm prediction)
4. ⭐ **Add Visibility** (marine safety, fog prediction)

---

## Bottom Line

**Nothing is "missing" from your CSV.** Your data is complete with all 31 parameters.

**The limitation is in the models.** GraphCast and iTransformer don't forecast all 31 parameters—only 22 of them.

**You have 9 parameters "waiting" in your CSV for models to forecast them.**

