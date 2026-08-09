import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path):
    """Runs a python script and handles errors."""
    logging.info(f"Starting {script_path}...")
    try:
        # Run the script and stream the output to the console
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True, 
            capture_output=False
        )
        logging.info(f"Successfully finished {script_path}\n{'-'*50}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running {script_path}")
        logging.error(f"Exit code: {e.returncode}")
        sys.exit(1)

def main():
    logging.info("Starting OctWave 3.0 End-to-End Pipeline")
    print("=" * 50)
    
    # Define the sequential pipeline steps
    pipeline_scripts = [
        "src/data_processing/preprocess.py",
        "src/modeling/train.py",
        "src/modeling/ensemble.py",
        "src/modeling/advanced_ensemble.py",
        "src/inference/predict.py"
    ]
    
    # Execute each script in order
    for script in pipeline_scripts:
        run_script(script)
        
    logging.info("Pipeline completed successfully! All predictions generated.")

if __name__ == "__main__":
    main()
