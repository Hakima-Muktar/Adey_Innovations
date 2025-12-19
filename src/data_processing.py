
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV, Excel, or JSON files based on file extension.
    Supported formats: .csv, .xlsx, .xls, .json
    """
    # Extract the file extension by splitting at the dot and taking the last part
    # Example: "data.csv" → "csv"
    file_ext = file_path.lower().split('.')[-1]

    try:
        # If file extension is CSV → use pandas read_csv()
        if file_ext == "csv":
            return pd.read_csv(file_path)
         # If file extension is Excel (xlsx or xls) → use pandas read_excel()
        elif file_ext in ("xlsx", "xls"):
            return pd.read_excel(file_path)
         # If file extension is JSON → use pandas read_json()
        elif file_ext == "json":
            return pd.read_json(file_path)
        else:
            # If file format is not supported, raise an explicit error
            raise ValueError(f"Unsupported file type: .{file_ext}")
    except Exception as e:
        # If any error occurs (e.g., file missing, corrupt), wrap it in a ValueError
        # This gives a clear and consistent error message for the user
        raise ValueError(f"Failed to load file '{file_path}': {e}")


