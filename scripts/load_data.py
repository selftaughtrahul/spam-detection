"""
Data loader for SMS Spam Collection dataset
"""
import pandas as pd
from pathlib import Path

def load_sms_spam_data():
    """
    Load and parse the SMS Spam Collection dataset
    
    Returns:
        pd.DataFrame: DataFrame with 'label' and 'message' columns
    """
    data_path = Path(__file__).parent.parent / "data" / "raw" / "SMSSpamCollection"
    
    # Read tab-separated file
    df = pd.read_csv(
        data_path,
        sep='\t',
        names=['label', 'message'],
        encoding='utf-8'
    )
    
    # Convert labels to binary (0=ham, 1=spam)
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"📊 Dataset loaded successfully!")
    print(f"   Total messages: {len(df)}")
    print(f"   Spam messages: {df['label_binary'].sum()} ({df['label_binary'].sum()/len(df)*100:.1f}%)")
    print(f"   Ham messages: {(df['label_binary']==0).sum()} ({(df['label_binary']==0).sum()/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = load_sms_spam_data()
    print("\n📝 Sample messages:")
    print(df.head(10))
    
    # Save as CSV for easier processing
    output_path = Path(__file__).parent.parent / "data" / "raw" / "spam_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed data to: {output_path}")
