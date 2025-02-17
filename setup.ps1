# PowerShell Script to Set Up Python Virtual Environment and Install Dependencies from requirements.txt

# Add the current directory to the PYTHONPATH environment variable
$env:PYTHONPATH="$($PWD.Path);$env:PYTHONPATH"

# Get the current working directory
$projectDir = Get-Location

# Define the venv folder path
$venvPath = Join-Path $projectDir "venv"

# Check if the virtual environment already exists
if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found. Creating a new one at $venvPath..."
    
    # Create a virtual environment
    python -m venv $venvPath
    
    # Activate the virtual environment
    Write-Host "Activating virtual environment..."
    & "$venvPath\Scripts\Activate"

    # Check if requirements.txt exists in the current folder
    $requirementsPath = Join-Path $projectDir "requirements.txt"
    if (Test-Path $requirementsPath) {
        Write-Host "Found requirements.txt at $requirementsPath"
        
        # Install dependencies from requirements.txt
        Write-Host "Installing dependencies..."
        pip install -r $requirementsPath

        Write-Host "Virtual environment setup complete. Dependencies installed successfully."
    } else {
        Write-Host "requirements.txt not found in $projectDir. Virtual environment created but no packages installed."
    }
} else {
    Write-Host "Virtual environment already exists at $venvPath. Skipping creation and dependency installation."
    
    # Activate the existing virtual environment
    Write-Host "Activating existing virtual environment..."
    & "$venvPath\Scripts\Activate"
}
# Note: You can manually deactivate later using the `deactivate` command inside the activated venv.
