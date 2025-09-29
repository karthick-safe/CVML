"""
Monitor YOLO training progress for CardioChek Plus detection
"""

import os
import time
import pandas as pd
from pathlib import Path

def monitor_training():
    """Monitor the YOLO training progress"""
    
    results_file = Path("backend/cardio_chek_models/cardio_chek_detector3/results.csv")
    
    if not results_file.exists():
        print("❌ Training results file not found")
        return
    
    print("🔍 Monitoring YOLO Training Progress")
    print("=" * 50)
    
    while True:
        try:
            # Read the latest results
            df = pd.read_csv(results_file)
            latest_epoch = df.iloc[-1]
            
            epoch = int(latest_epoch['epoch'])
            train_loss = latest_epoch['train/cls_loss']
            val_loss = latest_epoch['val/cls_loss']
            
            print(f"📊 Epoch {epoch}/100")
            print(f"   Training Loss: {train_loss:.4f}")
            print(f"   Validation Loss: {val_loss:.4f}")
            print(f"   Progress: {epoch}%")
            
            if epoch >= 100:
                print("\n🎉 Training Completed Successfully!")
                print("✅ YOLO model trained for CardioChek Plus detection")
                break
            
            print("⏳ Training in progress...")
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"❌ Error monitoring training: {e}")
            time.sleep(10)
    
    # Check if model files exist
    weights_dir = Path("backend/cardio_chek_models/cardio_chek_detector3/weights")
    if weights_dir.exists():
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        
        if best_model.exists():
            print(f"✅ Best model saved: {best_model}")
        if last_model.exists():
            print(f"✅ Last model saved: {last_model}")
    
    print("\n🚀 Training Complete! Your CardioChek Plus detection model is ready!")

if __name__ == "__main__":
    monitor_training()
