from sec_edgar_downloader import Downloader
import os

# SEC requires your name and email to identify API requests
dl = Downloader("FinSight", "marketing2.thenxsshop@gmail.com", "./data/raw")

# Start small — 2 filings each to test
print("Downloading Apple 10-Ks...")
dl.get("10-K", "AAPL", limit=2)

print("Downloading Goldman Sachs 10-Ks...")
dl.get("10-K", "GS", limit=2)

print("Downloading Microsoft 10-Ks...")
dl.get("10-K", "MSFT", limit=2)

print("\nDone! Checking downloaded files...")
for root, dirs, files in os.walk("./data/raw"):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filepath} ({size_mb:.1f} MB)")