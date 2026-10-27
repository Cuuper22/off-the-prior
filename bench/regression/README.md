# Regression sets for the Timestral fine-tune check

Fixed sets of forecasting windows. They are used to check that a fine-tuned Timestral still forecasts
normal series as well as the released model. They are not a leaderboard.

| set | windows | what it is |
|---|---|---|
| `real` | 412 | last windows of GIFT-Eval and fev-bench series |
| `real_extra` | 80 | last windows from further open-license sources |
| `analogues` | 69 | real series standing in for the synthetic failure families, tagged |
| `in_prior` | 500 | synthetic trend plus season plus AR noise series with oracle quantiles |

Files: `<set>.npz` (windows from sources whose license allows redistribution), `<set>_restricted.npz`
(windows from other sources; git-ignored, rebuilt locally), `manifest.csv` (one row per window: set, sample id,
source, revision, source file, file hash, dataset, term, domain, frequency, series id, window start, origin,
T, H, m, redistributable, family_analogue, reason), `sources.csv` (every raw file: name, URL, license, license
evidence, date fetched, bytes, SHA-256).

Load a set with `otp.realdata.load_union("bench/regression", "<set>")` (real sets; windows missing on this machine
are skipped) or `otp.generators.base.load_samples("bench/regression/in_prior.npz")`. Both return lists of
`otp.generators.base.Sample`. Rebuild everything with
`python scripts/build_regression_sets.py --cache-dir <dir>`; fetch only with `python scripts/fetch_realdata.py`.

## License rule

Windows are committed only when the source states a license that allows redistribution (public domain or US
Government work, CC0, CC BY, Apache-2.0, MIT). The Hub card of `Salesforce/GiftEval` states apache-2.0; its
per-dataset upstream terms were not separately verified, and `restaurant` (competition data) is treated as
restricted. `autogluon/fev_datasets` states no license and says research use only, so all windows from it are
restricted. SILSO monthly sunspots (CC BY-NC) and the R datasets (lynx, lh) are restricted. The Transnet BW part
of the grid-frequency record has no open license and is not used. `sources.csv` has the license and the
check behind every row. Restricted counts: real 372, real_extra 0, analogues 5.

## Window rule

- Real set: the last window of the series. H is GIFT-Eval's prediction length for the dataset frequency (hourly,
  10-minute and 15-minute 48; daily 30; weekly 8; monthly 12; the M4 datasets use the M4 lengths: hourly 48, daily 14,
  weekly 13, monthly 18). Pairs with the term `medium` use 10 times that length. The cap is 720.
- Context `T = min(2048, L - H)`; a series needs `T >= 256` for the real and extra sets. The analogue subset
  allows shorter contexts for short natural records (lynx, hormone series); `T` is in the manifest.
- Only the first variate of multivariate series is used. For `fred_md_2025` every column is one series.
- Series are visited in a seeded random order (seed 20261026). A series is skipped when its future window has a
  missing value, more than 5% of its context is missing, or its context is constant. Missing values inside an
  accepted context are filled by forward then backward fill (`n_filled`).
- `m` is a calendar period used for MASE scaling and seasonal naive: 24 hourly, 96 for 15-minute, 144 for 10-minute,
  7 daily, 5 for business-day series, 52 weekly, 12 monthly; 1 where the manifest says so.
- Live sources (FRED, USGS, NOAA, Seattle open data, OWID) are cut at 2026-09-30; `sources.csv` has the hash of the
  exact bytes used. Event windows in the analogue subset start at a fixed origin date given in the manifest.
- `oracle_q` is NaN for every real window. Score these windows against the classical models and seasonal naive.
- `hierarchical_sales/D` of GIFT-Eval was tried and has no usable series (all series have missing days inside the
  last 30 steps).

Overlap with Timestral's training data: Timestral was pretrained mostly on synthetic data, and the mix is not fully
documented. GIFT-Eval, fev-bench and the other public series may overlap it in unknown ways. The sets are used only
to detect regressions between Timestral and its fine-tuned versions, not as a leaderboard claim. Medium-term pairs
reuse series of the short-term pairs with their own seeded order, so windows may overlap.

## Real set pairs

| dataset | term | domain | freq | windows | T min | T max | H | m | redistributable |
|---|---|---|---|---|---|---|---|---|---|
| electricity/H | short | energy | H | 16 | 2048 | 2048 | 48 | 24 | yes |
| electricity/D | short | energy | D | 16 | 1431 | 1431 | 30 | 7 | yes |
| solar/H | short | energy | H | 16 | 2048 | 2048 | 48 | 24 | no |
| solar/10T | short | energy | 10T | 16 | 2048 | 2048 | 48 | 144 | no |
| entsoe/1H | short | energy | H | 6 | 2048 | 2048 | 48 | 24 | no |
| ercot/1H | short | energy | H | 8 | 2048 | 2048 | 48 | 24 | no |
| LOOP_SEATTLE/H | short | transport | H | 16 | 2048 | 2048 | 48 | 24 | no |
| M_DENSE/H | short | transport | H | 16 | 2048 | 2048 | 48 | 24 | no |
| SZ_TAXI/15T | short | transport | 15T | 16 | 2048 | 2048 | 48 | 96 | no |
| restaurant | short | sales | D | 16 | 266 | 447 | 30 | 7 | no |
| favorita_stores/1D | short | sales | D | 16 | 1658 | 1658 | 30 | 7 | no |
| rossmann/1D | short | sales | D | 16 | 912 | 912 | 30 | 7 | no |
| rohlik_sales/1D | short | sales | D | 16 | 338 | 1372 | 30 | 7 | no |
| m5/1W | short | sales | W | 16 | 259 | 272 | 8 | 52 | no |
| m4_hourly | short | econ_fin | H | 16 | 700 | 960 | 48 | 24 | no |
| m4_daily | short | econ_fin | D | 16 | 280 | 2048 | 14 | 7 | no |
| m4_weekly | short | econ_fin | W | 16 | 379 | 2048 | 13 | 52 | no |
| m4_monthly | short | econ_fin | M | 16 | 272 | 554 | 18 | 12 | no |
| fred_md_2025 | short | econ_fin | M | 16 | 786 | 786 | 12 | 12 | no |
| bitbrains_fast_storage/H | short | web_cloudops | H | 16 | 673 | 673 | 48 | 24 | no |
| redset/1H | short | web_cloudops | H | 16 | 1418 | 2048 | 48 | 24 | no |
| kdd_cup_2018_with_missing/H | short | nature | H | 16 | 2048 | 2048 | 48 | 24 | no |
| uk_covid_utla/1D | short | healthcare | D | 16 | 686 | 697 | 30 | 7 | no |
| hospital_admissions/1D | short | healthcare | D | 8 | 1699 | 1701 | 30 | 7 | no |
| electricity/H | medium | energy | H | 8 | 2048 | 2048 | 480 | 24 | yes |
| solar/H | medium | energy | H | 8 | 2048 | 2048 | 480 | 24 | no |
| solar/10T | medium | energy | 10T | 8 | 2048 | 2048 | 480 | 144 | no |
| LOOP_SEATTLE/H | medium | transport | H | 8 | 2048 | 2048 | 480 | 24 | no |
| M_DENSE/H | medium | transport | H | 8 | 2048 | 2048 | 480 | 24 | no |
| SZ_TAXI/15T | medium | transport | 15T | 8 | 2048 | 2048 | 480 | 96 | no |
| entsoe/1H | medium | energy | H | 6 | 2048 | 2048 | 480 | 24 | no |

## Extra real set (further open-license sources)

| dataset | domain | freq | windows | T min | T max | H | m | redistributable |
|---|---|---|---|---|---|---|---|---|
| fred_daily | finance | B | 8 | 2048 | 2048 | 30 | 5 | yes |
| fred_daily | energy | B | 3 | 2048 | 2048 | 30 | 5 | yes |
| ghcn_daily_TMAX | weather | D | 6 | 2048 | 2048 | 30 | 7 | yes |
| ghcn_daily_PRCP | weather | D | 3 | 2048 | 2048 | 30 | 7 | yes |
| usgs_daily_discharge | ecology_hydrology | D | 6 | 2048 | 2048 | 30 | 7 | yes |
| owid_covid_new_cases | epidemiology | D | 10 | 1202 | 1644 | 30 | 7 | yes |
| monash_web_traffic_daily | web_traffic | D | 16 | 773 | 773 | 30 | 7 | yes |
| seattle_fremont_bridge | transport | H | 1 | 2048 | 2048 | 48 | 24 | yes |
| obf_control_activity_15min | physiology | 15T | 8 | 1396 | 2048 | 48 | 96 | yes |
| monash_sunspot_daily | astronomy | D | 1 | 2048 | 2048 | 30 | 7 | yes |
| monash_pedestrian_counts | sensors_transport | H | 12 | 2048 | 2048 | 48 | 24 | yes |
| monash_bitcoin | finance | D | 6 | 2048 | 2048 | 30 | 7 | yes |

## Analogue subset

Real series that stand in for the synthetic failure families (`family_analogue` and `reason` in the manifest):

| family_analogue | dataset | windows | T min | T max | H | m | redistributable |
|---|---|---|---|---|---|---|---|
| delay_feedback | lynx | 1 | 104 | 104 | 10 | 1 | no |
| delay_feedback | lh | 1 | 36 | 36 | 12 | 1 | no |
| delay_feedback | monash_sunspot_daily | 3 | 2048 | 2048 | 30 | 7 | yes |
| delay_feedback | silso_monthly_sunspots | 3 | 2048 | 2048 | 12 | 12 | no |
| delay_feedback | nino34_monthly | 4 | 1526 | 1867 | 12 | 12 | yes |
| coupled_oscillators | fingrid_frequency_1s | 6 | 2048 | 2048 | 300 | 1 | yes |
| coupled_oscillators | obf_control_activity_15min | 8 | 1396 | 2048 | 48 | 96 | yes |
| heavy_tails | fred_daily | 8 | 2048 | 2048 | 30 | 5 | yes |
| heavy_tails | usgs_quakes_daily | 2 | 2048 | 2048 | 30 | 7 | yes |
| heavy_tails | usgs_daily_discharge | 3 | 2048 | 2048 | 30 | 7 | yes |
| regime_switches | fred_monthly_breaks | 9 | 337 | 1213 | 12 | 12 | yes |
| regime_switches | fred_energy_breaks | 4 | 2048 | 2048 | 30/60 | 5 | yes |
| regime_switches | seattle_fremont_bridge | 3 | 2048 | 2048 | 168/480 | 24 | yes |
| regime_switches | owid_covid_new_cases | 3 | 696 | 696 | 60 | 7 | yes |
| coupled_oscillators | monash_pedestrian_counts | 5 | 2048 | 2048 | 48 | 24 | yes |
| heavy_tails | monash_bitcoin | 2 | 2048 | 2048 | 30 | 7 | yes |
| regime_switches | monash_pedestrian_counts | 4 | 2048 | 2048 | 480 | 24 | yes |

- `analogues/lynx/lynx_annual`: Canadian lynx trappings 1821-1934: predator-prey cycle of about 10 years with delayed density dependence
- `analogues/lh/lh_hormone`: luteinizing hormone in blood samples every 10 minutes: pulsatile secretion under hormonal feedback
- `analogues/monash_sunspot_daily/T1@2008-12-01`: daily sunspot number; origin at solar minimum between cycles 23 and 24 (December 2008); solar dynamo cycle of about 11 years
- `analogues/monash_sunspot_daily/T1@2014-04-01`: daily sunspot number; origin at maximum of solar cycle 24 (smoothed maximum in April 2014); solar dynamo cycle of about 11 years
- `analogues/monash_sunspot_daily/T1`: daily sunspot number; origin at last window of the record (May 2020, close to the minimum before cycle 25); solar dynamo cycle of about 11 years
- `analogues/silso_monthly_sunspots/silso_monthly@2008-12-01`: monthly mean sunspot number; origin at solar minimum between cycles 23 and 24
- `analogues/silso_monthly_sunspots/silso_monthly@2014-04-01`: monthly mean sunspot number; origin at maximum of cycle 24
- `analogues/silso_monthly_sunspots/silso_monthly@2019-12-01`: monthly mean sunspot number; origin at solar minimum between cycles 24 and 25
- `analogues/nino34_monthly/nino34@1997-03-01`: Nino 3.4 anomaly, ENSO is a delayed oscillator; onset of the 1997-98 El Nino
- `analogues/nino34_monthly/nino34@2015-03-01`: Nino 3.4 anomaly, ENSO is a delayed oscillator; onset of the 2015-16 El Nino
- `analogues/nino34_monthly/nino34@2020-08-01`: Nino 3.4 anomaly, ENSO is a delayed oscillator; onset of the 2020-23 La Nina
- `analogues/nino34_monthly/nino34`: Nino 3.4 anomaly, ENSO is a delayed oscillator; last window of the record
- `analogues/fingrid_frequency_1s/fingrid@2024-01-08 03:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/fingrid_frequency_1s/fingrid@2024-01-15 15:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/fingrid_frequency_1s/fingrid@2024-02-05 03:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/fingrid_frequency_1s/fingrid@2024-02-19 15:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/fingrid_frequency_1s/fingrid@2024-03-04 03:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/fingrid_frequency_1s/fingrid@2024-03-18 15:00`: Nordic grid frequency in Hz at 1 s: many synchronous generators coupled through the network, inter-area oscillation modes
- `analogues/obf_control_activity_15min/control_1`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_10`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_11`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_12`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_13`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_14`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_15`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/obf_control_activity_15min/control_16`: wrist activity of a healthy control, 15-minute bins: circadian rhythm with ultradian components
- `analogues/fred_daily/DEXUSEU`: daily log return of US dollars per euro: fat-tailed returns with volatility clustering
- `analogues/fred_daily/DEXJPUS`: daily log return of yen per US dollar: fat-tailed returns with volatility clustering
- `analogues/fred_daily/DEXUSUK`: daily log return of US dollars per pound: fat-tailed returns with volatility clustering
- `analogues/fred_daily/DEXCAUS`: daily log return of Canadian dollars per US dollar: fat-tailed returns with volatility clustering
- `analogues/fred_daily/DEXINUS`: daily log return of rupees per US dollar: fat-tailed returns with volatility clustering
- `analogues/fred_daily/DCOILWTICO`: daily log return of WTI crude oil price: fat-tailed returns
- `analogues/fred_daily/DCOILBRENTEU`: daily log return of Brent crude oil price: fat-tailed returns
- `analogues/fred_daily/DHHNGSP`: daily log return of Henry Hub natural gas price: fat-tailed returns
- `analogues/usgs_quakes_daily/quake_daily_count_M4.5`: daily count of M4.5+ earthquakes worldwide: aftershock clusters, heavy-tailed counts
- `analogues/usgs_quakes_daily/quake_daily_max_magnitude`: daily maximum magnitude of M4.5+ earthquakes worldwide (4.4 on days without one): magnitudes follow a power-law tail
- `analogues/usgs_daily_discharge/01646500`: daily discharge, Potomac River near Washington DC: flood peaks with a heavy upper tail
- `analogues/usgs_daily_discharge/12113000`: daily discharge, Green River, Washington: flood peaks with a heavy upper tail
- `analogues/usgs_daily_discharge/06191500`: daily discharge, Yellowstone River at Corwin Springs: flood peaks with a heavy upper tail
- `analogues/fred_monthly_breaks/INDPRO@2020-02-01`: INDPRO: COVID-19 shutdown of US industry from March 2020
- `analogues/fred_monthly_breaks/INDPRO@2008-09-01`: INDPRO: global financial crisis after September 2008
- `analogues/fred_monthly_breaks/PAYEMS@2020-02-01`: PAYEMS: COVID-19 job losses from March 2020
- `analogues/fred_monthly_breaks/UNRATE@2020-02-01`: UNRATE: COVID-19 unemployment jump in April 2020
- `analogues/fred_monthly_breaks/RSAFS@2020-02-01`: RSAFS: COVID-19 retail sales collapse and rebound
- `analogues/fred_monthly_breaks/CPIAUCSL@2021-02-01`: CPIAUCSL: inflation surge from spring 2021
- `analogues/fred_monthly_breaks/FEDFUNDS@2022-02-01`: FEDFUNDS: start of the Federal Reserve tightening cycle in March 2022
- `analogues/fred_monthly_breaks/HOUST@2007-06-01`: HOUST: US housing starts during the housing bust
- `analogues/fred_monthly_breaks/M2SL@2020-03-01`: M2SL: jump in the money supply from March 2020
- `analogues/fred_energy_breaks/DCOILWTICO@2020-04-01`: DCOILWTICO: WTI front-month price fell below zero on 2020-04-20
- `analogues/fred_energy_breaks/DHHNGSP@2021-02-01`: DHHNGSP: Henry Hub gas spot price spike during Winter Storm Uri, 13-17 February 2021
- `analogues/fred_energy_breaks/DCOILBRENTEU@2014-09-30`: DCOILBRENTEU: oil price collapse from mid-2014
- `analogues/fred_energy_breaks/DCOILWTICO@2022-02-15`: DCOILWTICO: oil price spike after the invasion of Ukraine on 2022-02-24
- `analogues/seattle_fremont_bridge/fremont_total@2020-03-09`: hourly bicycle and scooter crossings, Fremont Bridge Seattle: Washington stay-home order of 2020-03-23 inside the horizon
- `analogues/seattle_fremont_bridge/fremont_total@2019-12-18`: hourly bicycle and scooter crossings, Fremont Bridge Seattle: Christmas week
- `analogues/seattle_fremont_bridge/fremont_total@2019-11-25`: hourly bicycle and scooter crossings, Fremont Bridge Seattle: Thanksgiving week (2019-11-28)
- `analogues/owid_covid_new_cases/United States@2021-12-01`: daily COVID-19 cases, United States: Omicron wave after the variant was named on 2021-11-26
- `analogues/owid_covid_new_cases/Germany@2021-12-01`: daily COVID-19 cases, Germany: Omicron wave after the variant was named on 2021-11-26
- `analogues/owid_covid_new_cases/Japan@2021-12-01`: daily COVID-19 cases, Japan: Omicron wave after the variant was named on 2021-11-26
- `analogues/monash_pedestrian_counts/T27@2019-11-04`: hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles
- `analogues/monash_pedestrian_counts/T49@2019-11-04`: hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles
- `analogues/monash_pedestrian_counts/T58@2019-11-04`: hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles
- `analogues/monash_pedestrian_counts/T18@2019-11-04`: hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles
- `analogues/monash_pedestrian_counts/T31@2019-11-04`: hourly pedestrian counts at sensors of an Australian city in a normal week: sensors driven by the same daily and weekly cycles
- `analogues/monash_bitcoin/transactions`: daily log change of bitcoin network series transactions: fat-tailed
- `analogues/monash_bitcoin/confirmation_time`: daily log change of bitcoin network series confirmation_time: fat-tailed
- `analogues/monash_pedestrian_counts/T43@2020-03-09`: hourly pedestrian counts at sensors of an Australian city: COVID-19 restrictions in March 2020 inside the horizon
- `analogues/monash_pedestrian_counts/T6@2020-03-09`: hourly pedestrian counts at sensors of an Australian city: COVID-19 restrictions in March 2020 inside the horizon
- `analogues/monash_pedestrian_counts/T31@2020-03-09`: hourly pedestrian counts at sensors of an Australian city: COVID-19 restrictions in March 2020 inside the horizon
- `analogues/monash_pedestrian_counts/T18@2020-03-09`: hourly pedestrian counts at sensors of an Australian city: COVID-19 restrictions in March 2020 inside the horizon

## In-prior set

500 series from `in_prior(seed)` for seeds 0 to 499, context 2048, horizon 128, built with
`make_sample(in_prior(seed), 2048, 128, seed=seed)` (500 oracle paths). The oracle quantiles are the Monte
Carlo continuation of the true process from its true state and are meaningful here. Seasonal periods: 7, 12, 24, 48, 96, 168.
