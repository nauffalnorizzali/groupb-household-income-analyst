# Assignment 1: DOSM Household Income and Poverty Audit (2016–2024)

## Group Information
- **Group:** B
- **Members:**
  1. Muhammad Danish Bin Ghazali (313464) — Data Steward
  2. Nauffal Bin Norizzali (309324) — Structures Engineer
  3. Muhammad Saif Aqil bin Norazman (310509) — Matrix Analyst A
  4. Irdina Balqis Binti Sheikh Khalid (308161) — Matrix Analyst B
  5. Elias Mohammed Aqel Abujhesheh (303072) — Integration & Test Lead

## Setup and reproducibility
1. Use Python 3.11 or later.
2. From this folder, install the dependency: `python -m pip install -r requirements.txt`.
3. Run the unit and integration tests: `python -m unittest -v test_solution.py`.
4. Open `assignment1.ipynb` in Jupyter and choose **Restart Kernel and Run All Cells**.

The notebook and tests use the saved JSON files under `data/raw/`. They do not need to download fresh data when these files are present. To refresh data intentionally, run `python fetch_data.py` with an internet connection. The saved data files are included to support repeatable results.

## Project files
- `solution.py`: analysis functions.
- `assignment1.ipynb`: analysis and Task 5 comparison/reproducibility section.
- `test_solution.py`: unit and real-data integration tests.
- `data/raw/hh_income_state.json`, `data/raw/hh_poverty_state.json`: saved source data.
- `CONTRIBUTIONS.md`: contribution record.
