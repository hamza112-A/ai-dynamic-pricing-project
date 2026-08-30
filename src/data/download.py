import os
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def download_cicids2017(output_dir):
    
    output_dir = Path(output_dir) / "cicids2017"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CICIDS2017 Dataset Download Instructions")
    print("=" * 70)
    print("\nPlease manually download the CICIDS2017 dataset from:")
    print("https://www.unb.ca/cic/datasets/ids-2017.html")
    print("\nDownload all CSV files and place them in:")
    print(f"  {output_dir.absolute()}")
    print("\nExpected files:")
    expected_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv", 
        "Wednesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours.pcap_ISCX.csv",
        "Friday-WorkingHours.pcap_ISCX.csv"
    ]
    for f in expected_files:
        print(f"  - {f}")
    print("=" * 70)
    
    return output_dir


def download_unsw_nb15(output_dir):
    
    output_dir = Path(output_dir) / "unsw-nb15"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("UNSW-NB15 Dataset Download Instructions")
    print("=" * 70)
    print("\nPlease manually download the UNSW-NB15 dataset from:")
    print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("\nDownload the following files and place them in:")
    print(f"  {output_dir.absolute()}")
    print("\nExpected files:")
    expected_files = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv",
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv"
    ]
    for f in expected_files:
        print(f"  - {f}")
    print("=" * 70)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download NIDS datasets (CICIDS2017, UNSW-NB15)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cicids2017", "unsw-nb15", "both"],
        help="Which dataset to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/",
        help="Output directory for datasets"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ["cicids2017", "both"]:
        download_cicids2017(output_dir)
    
    if args.dataset in ["unsw-nb15", "both"]:
        download_unsw_nb15(output_dir)
    
    print("\n✓ Download instructions displayed.")
    print("Please follow the instructions above to obtain the datasets.")


if __name__ == "__main__":
    main()
