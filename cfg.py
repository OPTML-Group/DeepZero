import os

# Modify path config according to your need!

base_path = f"{os.getenv('HOME')}/workspace"
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "projects", "DeepZero", "results")