"""
Script to download SMS Spam Collection dataset
"""
import urllib.request
import zipfile
import os
from pathlib import Path

def download_sms_spam_dataset():
    """
    Downloads the UCI SMS Spam Collection dataset
    """
    # Create data/raw directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = data_dir / "smsspamcollection.zip"
    
    print("📥 Downloading SMS Spam Collection dataset...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✅ Downloaded to: {zip_path}")
        
        # Extract the zip file
        print("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"✅ Extracted to: {data_dir}")
        
        # Remove zip file
        os.remove(zip_path)
        print("🗑️  Removed zip file")
        
        # List downloaded files
        print("\n📂 Files in data/raw/:")
        for file in data_dir.iterdir():
            print(f"  - {file.name} ({file.stat().st_size / 1024:.2f} KB)")
        
        print("\n✅ Dataset ready for preprocessing!")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")

if __name__ == "__main__":
    download_sms_spam_dataset()
