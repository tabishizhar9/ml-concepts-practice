import importlib
import os
import nbformat
import pathlib

API_CONVERTED_SCRIPTS_MODULE = "api_converted_scripts"

def extract_definitions_and_imports_filtered(notebook_path, output_dir="."):
    """
    Reads a notebook, extracts all general import statements and only code cells that contain
    function or class definitions, ignoring top-level execution logic AND specific Colab imports.
    """
    
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except FileNotFoundError:
        print(f"Error: Notebook file not found at {notebook_path}")
        return
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return

    # Initialize lists to store imports and definitions separately
    all_imports = set()
    extracted_definitions = []
    
    DEFINITION_KEYWORDS = ('def ', 'class ')
    IMPORT_KEYWORDS = ('import ', 'from ')
    # Keyword to exclude imports related to Google Colab
    EXCLUDE_IMPORT_KEYWORD = 'google.colab'

    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_source = cell.source.strip()
            code_lines = code_source.split('\n')
            
            # 1. Check for Definitions
            is_definition = any(
                line.strip().startswith(DEFINITION_KEYWORDS) 
                for line in code_lines
            )
            
            if is_definition:
                # Add definition code directly
                extracted_definitions.append(f"# Extracted Cell: {notebook.cells.index(cell)}")
                extracted_definitions.append(code_source)
            
            # Check for Imports in ALL code cells
            for line in code_lines:
                line_stripped = line.strip()
                
                # Check for top-level import statements
                if line_stripped.startswith(IMPORT_KEYWORDS):
                    if EXCLUDE_IMPORT_KEYWORD not in line_stripped:
                        # Set to automatically handle duplicates
                        all_imports.add(line_stripped)
                    # The line is skipped as it contains 'google.colab'
                    
    # Write the extracted code to a new .py file
    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    output_filename = os.path.join(output_dir, base_name + '.py')
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine imports and definitions
    import_content = "\n".join(sorted(list(all_imports)))
    definition_content = "\n\n".join(extracted_definitions)
    
    final_script_content = (
        f"# Script containing only function/class definitions and filtered imports from {os.path.basename(notebook_path)}\n\n"
        f"{import_content}\n\n"
        f"{definition_content}"
    )
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(final_script_content)
            
    # print(f"Successfully exported module (Imports + Definitions, Colab filtered) from {notebook_path} to {output_filename}")
    return base_name



def output_dir_hint(output_folder, base_name):
    """Path of the extracted module, for error messages."""
    return os.path.join(str(output_folder), f"{base_name}.py")

def extract_module(notebook_name):
    # output_folder_name = "api_converted_scripts_test"
    TESTS_DIR = pathlib.Path(__file__).resolve().parent
    CODE_DIR = TESTS_DIR.parent

    output_folder = TESTS_DIR / API_CONVERTED_SCRIPTS_MODULE

    notebook_file = CODE_DIR / f"{notebook_name}.ipynb"
    
    base_name = extract_definitions_and_imports_filtered(notebook_file, output_folder)
    try:
        return importlib.import_module(f"{API_CONVERTED_SCRIPTS_MODULE}.{base_name}")
    except ModuleNotFoundError as exc:
        # Almost always a package missing from the environment. This used to be silent
        # and return None, after which every test in the file failed with
        # "'NoneType' object has no attribute ..." and the score silently went to zero.
        # Say what is actually wrong instead.
        raise ModuleNotFoundError(
            f"Could not import the code extracted from {notebook_file.name}, because "
            f"{exc.name!r} is not installed.\n"
            f"Create the assignment environment with `conda env create -f environment.yaml` "
            f"and select it as the notebook kernel, then run the tests again."
        ) from exc
    except Exception as exc:
        # A syntax error in the notebook, or a cell that fails at import time.
        raise ImportError(
            f"Could not import the code extracted from {notebook_file.name}: "
            f"{type(exc).__name__}: {exc}\n"
            f"The extracted file is {output_dir_hint(output_folder, base_name)}; open it to "
            f"see what the notebook produced."
        ) from exc


if __name__ == "__main__":
    notebook_file = "./derivatives_and_graphs.ipynb"
    output_folder = "./api_converted_scripts"
    extract_definitions_and_imports_filtered(notebook_file, output_folder)