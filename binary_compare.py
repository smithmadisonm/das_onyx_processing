import sys
import os

def compare_binary_files(file1, file2, chunk_size=8192):
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            chunk1 = f1.read(chunk_size)
            chunk2 = f2.read(chunk_size)
            
            if chunk1 != chunk2:
                return False
            
            if not chunk1:
                return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python binary_compare.py <folder1> <folder2>")
        sys.exit(1)

    folder1, folder2 = sys.argv[1], sys.argv[2]
    
    # Get all files in the folders
    files1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]
    # Sort
    files1.sort()
    files2.sort()
    # Ensure file count is the same
    if len(files1) != len(files2):
        print(f"File count mismatch: {len(files1)} files in {folder1} and {len(files2)} files in {folder2}")
    # Compare the files in the folders
    different = 0
    for file1, file2 in zip(files1, files2):
        # Check that the names are the same
        if os.path.basename(file1) != os.path.basename(file2):
            print(f"File names mismatch: {file1} and {file2}")
            sys.exit(1)
        if compare_binary_files(file1, file2):
            print(f"{file1} and {file2} are identical (excluding metadata).")
        else:
            print(f"{file1} and {file2} are different.")
            different += 1
    print(f"Found {different} different files.")

