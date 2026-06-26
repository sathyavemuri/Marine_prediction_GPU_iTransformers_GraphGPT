"""
=============================================================================
MARINE PARAMETER SYNTHETIC DATA GENERATOR
=============================================================================
Generates physically-realistic hourly marine data for ship mooring/docking.

Covers 16 parameters across 4 groups:
  - WAVE: Significant Wave Height, Wave Period
  - WIND: Wind Speed, Wind Direction
  - WATER: Tidal Level, Current Speed, Sea Surface Temp, Salinity
  - ATMOSPHERIC: Air Pressure, Humidity, Dew Point, Precipitation,
                 Solar Radiation, Visibility

Physical realism built in:
  - Diurnal cycles (temperature, solar, humidity)
  - Tidal harmonics (M2, S2, K1, O1 constituents -> ~12.42h, 12h, 23.93h, 25.82h)
  - Lunar/spring-neap modulation (~14.76 days)
  - Weather systems (passing fronts -> coupled pressure/wind/wave events)
  - Cross-parameter coupling (wind->waves, pressure->wind, temp<->dewpoint)
  - Realistic noise and autocorrelation
=============================================================================
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
N_DAYS = 75                       # 75 days: 70 for train/history + 5 to validate
FREQ_HOURS = 1                    # hourly resolution
START_DATE = "2026-04-01 00:00"

n_hours = N_DAYS * 24
t = np.arange(n_hours)            # hour index
hours_of_day = t % 24
days = t / 24.0

timestamps = pd.date_range(start=START_DATE, periods=n_hours, freq=f"{FREQ_HOURS}h")


def ou_noise(n, theta=0.15, sigma=1.0, x0=0.0):
    """Ornstein-Uhlenbeck process: autocorrelated noise (realistic for geophysical series)."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + theta * (0.0 - x[i-1]) + sigma * np.random.randn()
    return x


# ----------------------------------------------------------------------------
# WEATHER SYSTEMS: passing fronts drive coupled pressure/wind/wave events
# ----------------------------------------------------------------------------
# Generate a few storm/front events as Gaussian bumps over the period
def weather_envelope(n, n_events=6, base=0.0):
    env = np.zeros(n)
    centers = np.sort(np.random.uniform(0, n, n_events))
    for c in centers:
        width = np.random.uniform(18, 60)        # 18-60 hour systems
        amp = np.random.uniform(0.5, 1.0)
        env += amp * np.exp(-0.5 * ((np.arange(n) - c) / width) ** 2)
    env = env / (env.max() + 1e-9)               # normalize 0..1
    return env, centers

storm_env, storm_centers = weather_envelope(n_hours)

# ----------------------------------------------------------------------------
# 1. AIR PRESSURE (hPa) -- drops during storms, drives wind
# ----------------------------------------------------------------------------
pressure_base = 1013.0
pressure = (
    pressure_base
    + 6.0 * np.sin(2 * np.pi * days / 6.0)        # slow synoptic variation (~6 day)
    - 22.0 * storm_env                            # deep lows during storms
    + 1.5 * ou_noise(n_hours, theta=0.05, sigma=0.4)
)

# ----------------------------------------------------------------------------
# 2. WIND SPEED (m/s) -- inversely tied to pressure, enhanced during storms
# ----------------------------------------------------------------------------
pressure_gradient = -(np.gradient(pressure))      # falling pressure -> wind
wind_speed = (
    5.0                                            # baseline breeze
    + 14.0 * storm_env                             # storm winds
    + 8.0 * np.clip(pressure_gradient, 0, None)    # gradient wind
    + 2.0 * np.sin(2 * np.pi * hours_of_day / 24 + 1.0)  # diurnal sea-breeze
    + 1.5 * np.abs(ou_noise(n_hours, theta=0.1, sigma=0.5))
)
wind_speed = np.clip(wind_speed, 0.2, None)

# ----------------------------------------------------------------------------
# 3. WIND DIRECTION (degrees) -- rotates with passing systems
# ----------------------------------------------------------------------------
wind_dir_base = 220.0    # prevailing SW
wind_direction = (
    wind_dir_base
    + 60.0 * np.sin(2 * np.pi * days / 7.0)        # synoptic rotation
    + 90.0 * storm_env * np.sin(2 * np.pi * days / 2.0)  # frontal veering
    + 15.0 * ou_noise(n_hours, theta=0.1, sigma=0.5)
) % 360.0

# ----------------------------------------------------------------------------
# 4. SIGNIFICANT WAVE HEIGHT (m) -- wind-driven with fetch/lag
# ----------------------------------------------------------------------------
# Waves lag wind (sea state builds over hours). Use smoothed wind.
def smooth(x, w=8):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

wind_smoothed = smooth(wind_speed, 10)
swh = (
    0.3
    + 0.055 * wind_smoothed ** 1.3                 # wind-wave growth
    + 0.9 * storm_env                              # swell from storms
    + 0.12 * np.abs(ou_noise(n_hours, theta=0.1, sigma=0.4))
)
swh = np.clip(swh, 0.1, None)

# ----------------------------------------------------------------------------
# 5. WAVE PERIOD (s) -- correlated with wave height (longer waves = bigger)
# ----------------------------------------------------------------------------
wave_period = (
    3.5
    + 1.8 * np.sqrt(swh)                           # dispersion relation flavor
    + 0.5 * np.sin(2 * np.pi * days / 5.0)
    + 0.3 * ou_noise(n_hours, theta=0.15, sigma=0.3)
)
wave_period = np.clip(wave_period, 2.0, None)

# ----------------------------------------------------------------------------
# 6. TIDAL LEVEL (m) -- harmonic constituents + spring/neap modulation
# ----------------------------------------------------------------------------
# Major constituents (periods in hours)
M2 = 12.42; S2 = 12.00; K1 = 23.93; O1 = 25.82
tide = (
    1.20 * np.cos(2 * np.pi * t / M2)              # principal lunar semidiurnal
    + 0.45 * np.cos(2 * np.pi * t / S2 + 0.5)      # principal solar semidiurnal
    + 0.35 * np.cos(2 * np.pi * t / K1 + 1.0)      # lunisolar diurnal
    + 0.25 * np.cos(2 * np.pi * t / O1 + 2.0)      # principal lunar diurnal
)
# Spring-neap: M2+S2 beat ~14.76 days already emerges; add storm surge residual
storm_surge = 0.4 * storm_env + 0.05 * ou_noise(n_hours, theta=0.05, sigma=0.3)
tidal_level = tide + storm_surge

# ----------------------------------------------------------------------------
# 7. CURRENT SPEED (m/s) -- driven by tidal flow (derivative of tide) + wind
# ----------------------------------------------------------------------------
tidal_flow = np.abs(np.gradient(tide)) * 1.6       # current peaks at mid-tide
current_speed = (
    0.10
    + tidal_flow
    + 0.012 * wind_speed                           # wind-driven component
    + 0.03 * np.abs(ou_noise(n_hours, theta=0.15, sigma=0.3))
)
current_speed = np.clip(current_speed, 0.02, None)

# ----------------------------------------------------------------------------
# 8. SEA SURFACE TEMPERATURE (°C) -- slow seasonal + weak diurnal
# ----------------------------------------------------------------------------
sst = (
    12.0
    + 2.5 * np.sin(2 * np.pi * days / 75.0 - 1.5)  # seasonal warming trend
    + 0.4 * np.sin(2 * np.pi * hours_of_day / 24 - 2.0)  # weak diurnal
    - 0.8 * storm_env                              # storm mixing cools surface
    + 0.2 * ou_noise(n_hours, theta=0.05, sigma=0.3)
)

# ----------------------------------------------------------------------------
# 9. SALINITY (PSU) -- stable with small freshwater/rain dilution
# ----------------------------------------------------------------------------
salinity = (
    34.5
    + 0.3 * np.sin(2 * np.pi * days / 30.0)
    - 0.4 * storm_env                              # rain dilution during storms
    + 0.1 * ou_noise(n_hours, theta=0.05, sigma=0.3)
)

# ----------------------------------------------------------------------------
# 10. CONDUCTIVITY (mS/cm) -- function of salinity & temperature
# ----------------------------------------------------------------------------
conductivity = (
    2.9 * salinity / (1 + 0.02 * (20 - sst))       # rises with salinity & temp
    + 0.2 * ou_noise(n_hours, theta=0.1, sigma=0.2)
)

# ----------------------------------------------------------------------------
# 11. AIR TEMPERATURE (°C) -- strong diurnal + synoptic
# ----------------------------------------------------------------------------
air_temp = (
    13.0
    + 2.0 * np.sin(2 * np.pi * days / 75.0 - 1.5)  # seasonal
    + 5.0 * np.sin(2 * np.pi * (hours_of_day - 9) / 24)  # diurnal peak ~3pm
    - 1.5 * storm_env
    + 0.5 * ou_noise(n_hours, theta=0.1, sigma=0.4)
)

# ----------------------------------------------------------------------------
# 12. RELATIVE HUMIDITY (%) -- inverse diurnal to temp, rises in storms
# ----------------------------------------------------------------------------
rel_humidity = (
    72.0
    - 5.0 * np.sin(2 * np.pi * (hours_of_day - 9) / 24)  # drops midday
    + 18.0 * storm_env                             # high in storms
    + 3.0 * ou_noise(n_hours, theta=0.1, sigma=0.5)
)
rel_humidity = np.clip(rel_humidity, 30, 100)

# ----------------------------------------------------------------------------
# 13. DEW POINT (°C) -- derived from temp & humidity (Magnus formula)
# ----------------------------------------------------------------------------
a, b = 17.27, 237.7
gamma = (a * air_temp / (b + air_temp)) + np.log(rel_humidity / 100.0)
dew_point = (b * gamma) / (a - gamma)

# ----------------------------------------------------------------------------
# 14. PRECIPITATION INTENSITY (mm/h) -- bursty, tied to storms & humidity
# ----------------------------------------------------------------------------
rain_prob = storm_env * (rel_humidity / 100.0)
rain_mask = (np.random.rand(n_hours) < rain_prob * 0.6).astype(float)
precipitation = rain_mask * (storm_env * 8.0 + np.abs(np.random.randn(n_hours)) * 1.5)
precipitation = np.clip(precipitation, 0, None)

# ----------------------------------------------------------------------------
# 15. GLOBAL SOLAR RADIATION (W/m²) -- strong diurnal, zero at night, cloud-cut
# ----------------------------------------------------------------------------
solar_clear = np.clip(np.sin(np.pi * (hours_of_day - 6) / 12), 0, None) * 800.0
cloud_factor = 1.0 - 0.7 * storm_env              # clouds reduce radiation
solar_radiation = solar_clear * cloud_factor + 5 * np.abs(np.random.randn(n_hours))
solar_radiation = np.clip(solar_radiation, 0, None)

# ----------------------------------------------------------------------------
# 16. VISIBILITY (km) -- reduced by rain, high humidity (fog), spray
# ----------------------------------------------------------------------------
visibility = (
    20.0
    - 12.0 * storm_env                             # storms cut visibility
    - 0.08 * np.clip(rel_humidity - 90, 0, None) * 10  # fog at high RH
    - precipitation * 0.8                          # rain reduces it
    + 1.0 * ou_noise(n_hours, theta=0.2, sigma=0.4)
)
visibility = np.clip(visibility, 0.2, 25.0)

# ----------------------------------------------------------------------------
# ASSEMBLE DATAFRAME
# ----------------------------------------------------------------------------
df = pd.DataFrame({
    "timestamp": timestamps,
    "significant_wave_height_m": np.round(swh, 3),
    "wave_period_s": np.round(wave_period, 3),
    "wind_speed_ms": np.round(wind_speed, 3),
    "wind_direction_deg": np.round(wind_direction, 1),
    "tidal_level_m": np.round(tidal_level, 3),
    "current_speed_ms": np.round(current_speed, 3),
    "sea_surface_temp_c": np.round(sst, 3),
    "salinity_psu": np.round(salinity, 3),
    "conductivity_mscm": np.round(conductivity, 3),
    "air_pressure_hpa": np.round(pressure, 2),
    "air_temp_c": np.round(air_temp, 3),
    "relative_humidity_pct": np.round(rel_humidity, 2),
    "dew_point_c": np.round(dew_point, 3),
    "precipitation_mmh": np.round(precipitation, 3),
    "solar_radiation_wm2": np.round(solar_radiation, 2),
    "visibility_km": np.round(visibility, 3),
})

# Save
out_path = "/home/claude/marine_data_75days.csv"
df.to_csv(out_path, index=False)

print(f"Generated marine dataset: {df.shape[0]} hourly rows x {df.shape[1]-1} parameters")
print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"Total days: {N_DAYS} (70 days history + 5 days to forecast/validate)")
print(f"\nFile: {out_path}")
print("\n" + "="*78)
print("STATISTICAL SUMMARY")
print("="*78)
print(df.describe().T[["mean", "min", "max", "std"]].round(2).to_string())
