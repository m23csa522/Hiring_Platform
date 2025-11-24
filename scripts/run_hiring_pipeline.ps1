# run_hiring_pipeline.ps1

# (Optional) echo what we're doing
Write-Host "Running hiring platform pipeline in 'hiring_platform' conda env..." -ForegroundColor Cyan

# Run the pipeline in the correct env
conda run -n hiring_platform python scripts/resume_mapping/run_pipeline.py
