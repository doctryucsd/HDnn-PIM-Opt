import json
import os
import glob

# ==========================================
# CONFIGURATION & THRESHOLDS
# ==========================================
# Set your constraint thresholds here.
# Only points that meet ALL these conditions are eligible for the Pareto front.
CONSTRAINTS = {
    "min_accuracy": 0.79,  # Maximize: e.g., 0.85
    "max_energy": 0.5,    # Minimize: e.g., 0.05
    "max_timing": 0.5,    # Minimize: e.g., 0.10
    "max_area": 0.2       # Minimize: e.g., 0.20
}

def is_dominated(p1, p2):
    """
    Returns True if p1 is dominated by p2.
    Logic: p2 dominates p1 if it is better-than-or-equal in all metrics
    AND strictly better in at least one metric.
    """
    # Accuracy: Higher is better
    # Energy, Timing, Area: Lower is better
    better_or_equal = (
        p2['accuracy'] >= p1['accuracy'] and
        p2['energy'] <= p1['energy'] and
        p2['timing'] <= p1['timing'] and
        p2['area'] <= p1['area']
    )
    
    strictly_better = (
        p2['accuracy'] > p1['accuracy'] or
        p2['energy'] < p1['energy'] or
        p2['timing'] < p1['timing'] or
        p2['area'] < p1['area']
    )
    
    return better_or_equal and strictly_better

def find_pareto_front(points):
    """
    Identifies non-dominated points from the global pool of eligible points.
    """
    pareto_front = []
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i == j:
                continue
            if is_dominated(p1, p2):
                dominated = True
                break
        if not dominated:
            pareto_front.append(p1)
    return pareto_front

def process_directory(base_path):
    all_eligible_points = []
    
    # Use recursive glob pattern to find all .json files in all subdirectories
    search_pattern = os.path.join(base_path, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"Error: No .json files found in {base_path} or its subdirectories.")
        return

    print(f"Scanning {len(json_files)} files...")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metrics and parameters
            accs = data.get("accuracy", [])
            enes = data.get("energy", [])
            times = data.get("timing", [])
            areas = data.get("area", [])
            params = data.get("param", [])

            # Check for consistent data length within the file
            num_points = len(accs)
            if not all(len(lst) == num_points for lst in [enes, times, areas, params]):
                print(f"Warning: Mismatched list lengths in {file_path}. Skipping file.")
                continue

            for i in range(num_points):
                # 6. Apply constraints before Pareto evaluation
                if (accs[i] >= CONSTRAINTS["min_accuracy"] and
                    enes[i] <= CONSTRAINTS["max_energy"] and
                    times[i] <= CONSTRAINTS["max_timing"] and
                    areas[i] <= CONSTRAINTS["max_area"]):
                    
                    all_eligible_points.append({
                        "accuracy": accs[i],
                        "energy": enes[i],
                        "timing": times[i],
                        "area": areas[i],
                        "param": params[i],
                        "file": file_path # Keep full path for identification
                    })

        except Exception as e:
            # 4. Handle exceptions and print warning
            print(f"Warning: Failed to process {file_path}. Error: {e}")

    if not all_eligible_points:
        print("No points met the specified constraint thresholds.")
        return

    # 3. Compute Pareto front from the combined pool
    pareto_points = find_pareto_front(all_eligible_points)

    # 4. Print results to stdout
    print(f"\n{'='*100}")
    print(f"PARETO FRONT RESULTS: {len(pareto_points)} non-dominated points found across all directories")
    print(f"{'='*100}\n")

    for idx, pt in enumerate(pareto_points):
        print(f"Point #{idx + 1}")
        print(f"  [Metrics] Acc: {pt['accuracy']:.4f} | Energy: {pt['energy']:.4e} | "
              f"Timing: {pt['timing']:.4f} | Area: {pt['area']:.4f}")
        print(f"  [Source]  {pt['file']}")
        print(f"  [Params]  {pt['param']}")
        print(f"{'-'*100}")

if __name__ == "__main__":
    import sys
    # Accept directory path as argument; default to current directory if not provided
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    process_directory(target_path)