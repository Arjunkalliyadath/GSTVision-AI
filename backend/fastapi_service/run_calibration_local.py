import asyncio
import os
import sys

# Setting paths for the backend script
sys.path.append(os.path.abspath("c:/DSA/Intern2/GST2_Converter/backend/fastapi_service"))
os.chdir("c:/DSA/Intern2/GST2_Converter/backend/fastapi_service")

# Dummy the imports that require settings
os.environ["MEDIA_ROOT"] = "../../training2"

from training.training_pipeline import run_training

async def main():
    print("Starting training for 30 epochs...")
    data_path = "../../training2/gstr2a_bw"
    
    def on_progress(info):
        print(f"[{info['percent']}%] {info['message']}")
        
    res = await run_training(
        training_data_dir=data_path,
        epochs=30,
        use_gpu=True,
        progress_callback=on_progress
    )
    print("\nTraining completed.")
    print(f"Status: {res.get('status')}")
    if res.get('status') == 'complete':
        print(f"Best Accuracy: {res['summary']['best_accuracy']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
