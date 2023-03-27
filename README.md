# EOLES-Dispatch Model

## Description
EOLES-Dispatch is an cost minimization dispatch model aimed at simulating realistic wholesale electricity prices at an hourly time-step for the French power system. It takes as input a detailed description of the installed generation capacity and time series of determinants of the power dispatch such as hourly demand, renewables availability, hydro resources, nuclear maintenance planning or fossil fuel price variations.

## Steps to run the model
1. Specify a scenario by copying and modifying the <Scenario_BASELINE.xlsx> file
2. In the last section of <EOLES-Dispatch.py>, specify all the simulations needed with the 'run_model()' function and the following specifications:
      - 'scenario' -- Name of the .xlsx in which specifications of the scenario to simulate are gathered (created in step 1)
      - 'year'     -- Which year (among 2016-2019) should be considered for weather related and other time-varying inputs (hourly power demand, wind and solar production, ...)
      - 'outputs'  -- Name of the directory in which outputs will be saved
3. Run the model by running the whole <EOLES-Dispatch.py> script
4. Get output data from the 'outputs' directory

## Contents
- <EOLES-Dispatch.py> - Python script running the EOLES-Dispatch optimization model
- <format_inputs.R> - R script selecting and formatting data to make it usable by the EOLES-Dispatch.py script
- <Scenario_FORMAT.xlsx> - Excel file containing all static scenario inputs and technical parameters (installed capacity per country, fuel costs, interconnection capacity etc.)
- dir .\time-varying_inputs\ - Complete set of raw historic data (years 2015-2019) to be selected from by the formatting script; Source: https://transparency.entsoe.eu/
- dir .\renewable_ninja\ - Capacity factors for wind and solar in each country; Source: https://renewables.ninja
- dir .\inputs\ - Empty directory where data formatted by the <format_inputs.R> script will be stored
- dir .\outputs\ - Emplty directory where outputs from the model will be stored after running <EOLES-Dispatch.py>
- <cbc.exe> - Default solver used by the EOLES-Dispatch.py script. All information about the Cbc solver is here : 
https://projects.coin-or.org/Cbc

## Acknowledgement
- EOLES-Dispatch was developped based on the EOLES model developped by Behrang Shirizadeh, Quentin Perrier & Philippe Quirion available on Behrang Shirizadeh's GitHub page: https://github.com/BehrangShirizadeh
- This model's script is based on the python version of the EOLES model which was written by Nilam De Oliveira-Gill and available at https://gitlab.in2p3.fr/nilam.de_oliveira-gill/eoles
- Renewable generation data is taken from the model developped by Iain Staffel and Stefan Pfenninger and made freely available at https://www.renewables.ninja/
