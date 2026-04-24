import subprocess
import sys

def run_script(script_name, step_description):
    print(f"\n--- {step_description} ---")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"Execution successful: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {script_name}. Exiting.")
        sys.exit(1)

def main():
    print("="*50)
    print("TWEET EMOTION CLASSIFICATION PIPELINE")
    print("="*50)
    
    run_script("eda_preprocessing.py", "Step 1: EDA & Preprocessing")
    run_script("pipeline.py", "Step 2: Feature Extraction & ML Training")
    
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main()
