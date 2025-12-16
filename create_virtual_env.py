import os
import subprocess
import sys
import zipfile
from pathlib import Path

def create_locator_file(folder_path):
    open(os.path.join(folder_path, '.locator'), 'w').close()

def delete_empty_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
            except OSError as e:
                print(f"Failed to delete {dirpath}: {e}")

def create_video_folders():

    # Create folder and place locator file there
    carts_dir = os.path.abspath('video_examples')
    os.makedirs(carts_dir, exist_ok=True)
    create_locator_file(carts_dir)

    # Process zip files 
    zip_files = [f for f in os.listdir('.') if f.endswith('zip')]

    for zip_file in zip_files:

        if 'test' not in zip_file:
            zip_name = os.path.splitext(zip_file)[0]
            extract_path = os.path.join(carts_dir, zip_name)
            os.makedirs(extract_path, exist_ok=True)

        else:
            extract_path = os.getcwd()

        # Extract all files to current folder
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Get only the file name (drop path)
                filename = os.path.basename(member.filename)
                if not filename:
                    continue  # Skip directories
                # Construct full path in current directory
                target_path = os.path.join(carts_dir, filename)

                # Extract to memory and write manually to avoid folders
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

def setup_venv(venv_dir='venv', requirements_file='requirements.txt'):
    """
    Creates a virtual environment and installs packages from requirements.txt.
    
    Parameters:
    - venv_dir (str): Name of the virtual environment directory.
    - requirements_file (str): Path to the requirements.txt file.
    """
    venv_path = Path(venv_dir)
    req_path = Path(requirements_file)

    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file '{requirements_file}' not found.")

    # Create virtual environment
    subprocess.check_call([sys.executable, '-m', 'venv', str(venv_path)])
    print(f"Virtual environment created at: {venv_path}")

    # Determine pip path in the venv
    if os.name == 'nt':  # Windows
        pip_path = venv_path / 'Scripts' / 'pip.exe'
    else:  # Unix/Linux/macOS
        pip_path = venv_path / 'bin' / 'pip'

    # Install from requirements.txt
    subprocess.check_call([str(pip_path), 'install', '-r', str(req_path)])
    print(f"Installed packages from '{requirements_file}'")

if __name__ == "__main__":
    venv_name = 'wheel_event_virtual_env'
    requirements = 'requirements.txt'

    try:
        create_video_folders()
        delete_empty_folders(".")
        setup_venv(venv_name, requirements)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)