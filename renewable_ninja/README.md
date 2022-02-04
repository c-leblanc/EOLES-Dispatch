This directory contains time-series data for the hourly capacity factors of solar and wind technologies. This data results from the simulation model developped by Stefan Pfenninger and Iain Staffell freely available at https://www.renewables.ninja/.

Raw data was downloaded from https://www.renewables.ninja/ on the 2021-06-15. The present data is based on country-wide hourly generation data from MERRA-2 model, including for all six countries the following files (if available for the country):
- PV (MERRA-2)
- Wind (Current fleet, onshore/offshore separated, MERRA-2)
- Wind (Near-term future fleet, onshore/offshore separated, MERRA-2)
- Wind (Long-term future fleet, onshore/offshore separated, MERRA-2)
The formatting procedure applied to this data to make it usable by the EOLES-Dispatch model is detailed in the R script <renewable_ninja_format.R>

The previouly mentionned formatting procedure result in:
- 1 file for solar PV gathering the average CF for each hour since 1980-01-001 and for each of the seven countries modeled in EOLES-Dispatch (<pv.csv>)
- 2 files for onshore wind gathering the average CF for each hour since 1980-01-001 and for each of the seven countries modeled in EOLES-Dispatch: one version describing the current fleet (<onshore_CU.csv>) and one version describing the near-term fleet (<onshore_NT.csv>)
- 3 files for offshore wind gathering the average CF for each hour since 1980-01-001 and for each of the seven countries modeled in EOLES-Dispatch: one version describing the current fleet (<offshore_CU.csv>), one version describing the near-term fleet (<offshore_NT.csv>) and one version describing the long-term fleet (<offshore_LT.csv>)
NB: The appropriate file to use should be specified in the <format_inputs.R> script in the root directory.
