from pathlib import Path

def main():
    csv_files = sorted(Path(".").glob("*.csv"))
    print(f"当前目录下共有 {len(csv_files)} 个 CSV 文件")


if __name__ == "__main__":
    main()