import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_script(notebook_path, script_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert the notebook to a Python script
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook)
    
    # Save the script
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)

notebook_path = './predictive_algoritms.ipynb'
script_path = 'converted_script.py'
convert_notebook_to_script(notebook_path, script_path)
