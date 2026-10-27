import json
import numpy as np

# Fixed Constants
SURVEY_YEARS = [2016, 2019, 2020, 2022, 2024]
POVERTY_YEARS = [2019, 2020, 2022, 2024]

REGIONS = {
    "Northern": ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Central": ["Selangor", "W.P. Kuala Lumpur", "W.P. Putrajaya", "Negeri Sembilan"],
    "Southern": ["Melaka", "Johor"],
    "East Coast": ["Kelantan", "Terengganu", "Pahang"],
    "East Malaysia": ["Sabah", "Sarawak", "W.P. Labuan"],
}

def get_group_states(matric_numbers: list[int], income_records: list[dict]) -> list[str]:
    """Calculate and assign unique focus states for group members based on matric numbers.
    
    Owner: Role 1. Reviewed by: Role 2.
    """
    states = sorted(list({r["state"] for r in income_records}))
    group_states = []
    for matric in sorted(matric_numbers):
        remainder = matric % len(states)
        candidate_idx = remainder
        while states[candidate_idx] in group_states:
            candidate_idx = (candidate_idx + 1) % len(states)
        group_states.append(states[candidate_idx])
    return group_states

def validate_records(records: list[dict], required_keys: tuple[str, ...]) -> None:
    """Raise an informative exception if records are not a usable list of keyed records.
    
    Owner: Role 1. Reviewed by: Role 2.
    """
    if not isinstance(records, list):
        raise TypeError("Records must be provided as a list.")
    if len(records) == 0:
        raise ValueError("Records list is empty.")
    
    seen = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Item at index {idx} is not a dictionary.")
        for key in required_keys:
            if key not in record:
                raise KeyError(f"Record at index {idx} is missing required key: '{key}'")
        
        if "state" in record and "date" in record:
            key = (record["state"], str(record["date"])[:4])
            if key in seen:
                raise ValueError(f"Duplicate record found for state-date combination: {key}")
            seen.add(key)

def index_by_key(records: list[dict]) -> dict:
    """Return a dictionary keyed by (state, year) and raise an exception on a duplicate key.
    
    Owner: Role 2. Reviewed by: Role 1.
    """
    indexed = {}
    for r in records:
        year = int(str(r["date"])[:4])
        key = (r["state"], year)
        if key in indexed:
            raise ValueError(f"Duplicate key detected: {key}")
        indexed[key] = r
    return indexed

def join_records(income_records: list[dict], poverty_records: list[dict]) -> tuple[list[dict], set, set]:
    """Match records on (state, year) and return joined records, income-only keys, and poverty-only keys.
    
    Owner: Role 2. Reviewed by: Role 1.
    """
    inc_idx = index_by_key(income_records)
    pov_idx = index_by_key(poverty_records)
    
    inc_keys = set(inc_idx.keys())
    pov_keys = set(pov_idx.keys())
    
    common = inc_keys.intersection(pov_keys)
    income_only = inc_keys - pov_keys
    poverty_only = pov_keys - inc_keys
    
    joined = [{**inc_idx[k], **pov_idx[k]} for k in common]
    return joined, income_only, poverty_only

def to_float(value) -> float:
    """Return np.nan for None and a float otherwise.
    
    Owner: Role 2. Reviewed by: Role 1.
    """
    if value is None:
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def build_matrices(income_records: list[dict], poverty_records: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build STATES, MEDIAN, MEAN, and POVERTY float matrices from records.
    
    Owner: Role 2. Reviewed by: Role 1.
    """
    states = sorted(list({r["state"] for r in income_records}))
    inc_idx = index_by_key(income_records)
    pov_idx = index_by_key(poverty_records)
    
    median_matrix = np.full((len(states), len(SURVEY_YEARS)), np.nan)
    mean_matrix = np.full((len(states), len(SURVEY_YEARS)), np.nan)
    poverty_matrix = np.full((len(states), len(POVERTY_YEARS)), np.nan)
    
    for i, state in enumerate(states):
        for j, year in enumerate(SURVEY_YEARS):
            rec = inc_idx.get((state, year))
            if rec:
                median_matrix[i, j] = to_float(rec.get("income_median"))
                mean_matrix[i, j] = to_float(rec.get("income_mean"))
        for j, year in enumerate(POVERTY_YEARS):
            rec = pov_idx.get((state, year))
            if rec:
                poverty_matrix[i, j] = to_float(rec.get("poverty_absolute"))
                
    return states, median_matrix, mean_matrix, poverty_matrix

def analyze_focus_state_vectors(focus_state: str, states: list[str], survey_years: list[int], median_matrix: np.ndarray):
    """Calculate consecutive changes, percentage changes, per-year changes, and falling periods for a focus state.
    
    Owner: Role 1. Reviewed by: Role 5.
    """
    state_idx = states.index(focus_state)
    medians = median_matrix[state_idx]
    year_gaps = np.diff(survey_years)
    
    abs_changes = np.diff(medians)
    pct_changes = (abs_changes / medians[:-1]) * 100
    per_year_changes = abs_changes / year_gaps
    falling_mask = abs_changes < 0
    return abs_changes, pct_changes, per_year_changes, falling_mask

def calculate_poverty_change_2016_2019(poverty_matrix: np.ndarray, poverty_years: list[int], states: list[str]) -> tuple[np.ndarray, int]:
    """Calculate absolute poverty change between 2019 and later years (or 2016 baseline check) across all states.
    
    Owner: Role 1. Reviewed by: Role 5.
    """
    # Note: Poverty years start at 2019. Difference between 2022 and 2019 or 2020 and 2019.
    pov_change = np.diff(poverty_matrix, axis=1) # columns: 2019->2020, 2020->2022, 2022->2024
    rising_count = int(np.sum(pov_change[:, 0] > 0)) # 2019 to 2020 change
    return pov_change, rising_count

def create_region_mapping(regions: dict[str, list[str]], states: list[str]) -> dict[str, str]:
    """Map each state to its region and ensure exact single membership (Item 4.6).
    
    Owner: Role 4. Reviewed by: Role 3.
    """
    state_to_region = {}
    for region, region_states in regions.items():
        for state in region_states:
            if state in state_to_region:
                raise ValueError(f"State {state} is assigned to multiple regions.")
            state_to_region[state] = region
            
    for state in states:
        if state not in state_to_region:
            raise ValueError(f"State {state} is missing from regional definitions.")
            
    return state_to_region

def calculate_relative_income_index(median_matrix: np.ndarray) -> np.ndarray:
    """Calculate relative income index using broadcasting (Item 4.4).
    
    Owner: Role 3. Reviewed by: Role 4.
    """
    year_medians = np.median(median_matrix, axis=0, keepdims=True)
    return median_matrix / year_medians

def analyze_matrices(median_matrix: np.ndarray, mean_matrix: np.ndarray, poverty_matrix: np.ndarray, states: list[str], regions: dict[str, list[str]]):
    """Perform matrix calculations for Task 4 (Items 4.1 to 4.7).
    
    Owner: Role 3 & Role 4. Reviewed by: Role 4 & Role 3.
    """
    # 4.1 Mean and median of state medians
    medians_mean = np.mean(median_matrix, axis=0)
    medians_median = np.median(median_matrix, axis=0)
    
    # 4.2 Percentage growth from 2016 to 2024
    growth_2016_2024 = ((median_matrix[:, -1] - median_matrix[:, 0]) / median_matrix[:, 0]) * 100
    
    # 4.3 Highest and lowest median income by survey year
    highest_indices = np.argmax(median_matrix, axis=0)
    lowest_indices = np.argmin(median_matrix, axis=0)
    
    # 4.4 Relative income index
    rel_index = calculate_relative_income_index(median_matrix)
    
    # 4.5 Index below 0.8 in 2024 and lower in 2024 than 2016
    below_08_2024 = [states[i] for i in range(len(states)) if rel_index[i, -1] < 0.8]
    lower_in_2024 = [states[i] for i in range(len(states)) if rel_index[i, -1] < rel_index[i, 0]]
    
    # 4.6 Regional unweighted mean of median income
    state_to_region = create_region_mapping(regions, states)
    region_means = {}
    for region, reg_states in regions.items():
        mask = np.array([state_to_region[s] == region for s in states])
        region_means[region] = np.mean(median_matrix[mask, :], axis=0)
        
    # 4.7 Boolean matrix for periods 2019-2020, 2020-2022, 2022-2024 where income rose but absolute poverty did not fall
    # Survey years index: 2016(0), 2019(1), 2020(2), 2022(3), 2024(4)
    # Poverty years index: 2019(0), 2020(1), 2022(2), 2024(3)
    income_diff = median_matrix[:, [2, 3, 4]] - median_matrix[:, [1, 2, 3]]
    poverty_diff = np.diff(poverty_matrix, axis=1)
    flagged_matrix = (income_diff > 0) & (poverty_diff >= 0)
    
    return {
        "medians_mean": medians_mean,
        "medians_median": medians_median,
        "growth_2016_2024": growth_2016_2024,
        "highest_states": [states[i] for i in highest_indices],
        "lowest_states": [states[i] for i in lowest_indices],
        "relative_index": rel_index,
        "below_08_2024": below_08_2024,
        "lower_in_2024": lower_in_2024,
        "region_means": region_means,
        "flagged_matrix": flagged_matrix,
    }

def build_state_profile(
    state: str,
    states: list[str],
    years: list[int],
    median: np.ndarray,
    mean: np.ndarray,
    poverty: np.ndarray,
    poverty_years: list[int],
    regions: dict[str, list[str]],
) -> dict:
    """Return a JSON-serializable profile of one state without using global variables.
    
    Owner: Role 4. Reviewed by: Role 3.
    """
    if state not in states:
        raise ValueError(f"Unknown state: {state}")
        
    state_to_region = create_region_mapping(regions, states)
    region = state_to_region[state]
    
    idx = states.index(state)
    state_medians = median[idx].tolist()
    state_means = mean[idx].tolist()
    
    growth = float(((state_medians[-1] - state_medians[0]) / state_medians[0]) * 100)
    
    year_medians = np.median(median, axis=0)
    latest_indices = median[:, -1] / year_medians[-1]
    rel_index = float(latest_indices[idx])
    
    ranks = np.argsort(np.argsort(latest_indices)[::-1]) + 1
    rank = int(ranks[idx])
    
    # Region mean median in latest year and whether state is above it
    mask = np.array([state_to_region[s] == region for s in states])
    region_latest_mean = float(np.mean(median[mask, -1]))
    above_region_mean = bool(median[idx, -1] > region_latest_mean)
    
    # Task 4.7 flags for this state
    income_diff = median[:, [2, 3, 4]] - median[:, [1, 2, 3]]
    poverty_diff = np.diff(poverty, axis=1)
    flagged = (income_diff > 0) & (poverty_diff >= 0)
    periods = ["2019-2020", "2020-2022", "2022-2024"]
    flagged_periods = [periods[p_idx] for p_idx, val in enumerate(flagged[idx]) if val]
    
    profile = {
        "state": state,
        "region": region,
        "survey_years": years,
        "median_incomes": state_medians,
        "mean_incomes": state_means,
        "growth_percentage": growth,
        "latest_relative_index": rel_index,
        "latest_rank": rank,
        "region_latest_mean": region_latest_mean,
        "above_region_mean": above_region_mean,
        "absolute_poverty": poverty[idx].tolist(),
        "poverty_years": poverty_years,
        "flagged_periods": flagged_periods,
    }
    
    return json.loads(json.dumps(profile))