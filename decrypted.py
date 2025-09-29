import os
from Crypto.Cipher import AES
from supabase import create_client

SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"

DECRYPTED_RAW = 'decrypted_raw_videos/'
DECRYPTED_MOTION = 'decrypted_motion_videos/'
DECRYPTED_ANOMALY = 'decrypted_anomaly_videos/'
DECRYPTED_PERSON = 'decrypted_person_images/'

os.makedirs(DECRYPTED_RAW, exist_ok=True)
os.makedirs(DECRYPTED_MOTION, exist_ok=True)
os.makedirs(DECRYPTED_ANOMALY, exist_ok=True)
os.makedirs(DECRYPTED_PERSON, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not os.path.exists("aes_key.bin"):
    print("ERROR: aes_key.bin not found!")
    print("Make sure the AES key file is in the same directory.")
    exit(1)

with open("aes_key.bin", "rb") as f:
    shared_key = f.read()

def aes_decrypt(encrypted_data, key):
    if len(encrypted_data) < 32:
        print(f"Error: Encrypted data too short ({len(encrypted_data)} bytes)")
        return None
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext
    except ValueError as e:
        print(f"Decryption failed: {e}")
        return None

def download_file_from_supabase(bucket_name, file_name):
    try:
        storage = supabase.storage.from_(bucket_name)
        file_data = storage.download(file_name)
        return file_data
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return None

def list_files_in_bucket(bucket_name):
    try:
        storage = supabase.storage.from_(bucket_name)
        files = storage.list()
        return [f for f in files if f['name'] != '.emptyFolderPlaceholder']
    except Exception as e:
        print(f"Error listing files in {bucket_name}: {e}")
        return []

def decrypt_and_save(bucket_name, file_name, output_dir):
    print(f"  Downloading: {file_name}...", end=' ')
    encrypted_data = download_file_from_supabase(bucket_name, file_name)
    if encrypted_data is None:
        print("Download failed")
        return False
    print(f"({len(encrypted_data)} bytes)", end=' ')
    print("Decrypting...", end=' ')
    decrypted_data = aes_decrypt(encrypted_data, shared_key)
    if decrypted_data is None:
        print("Decryption failed")
        return False
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    print(f"Saved ({len(decrypted_data)} bytes)")
    return True

def decrypt_bucket(bucket_name, output_dir):
    print(f"\n{'='*60}")
    print(f"Processing bucket: {bucket_name}")
    print(f"{'='*60}")
    files = list_files_in_bucket(bucket_name)
    if not files:
        print(f"No files found in {bucket_name}")
        return
    print(f"Found {len(files)} file(s)\n")
    success_count = 0
    for file_info in files:
        file_name = file_info['name']
        if decrypt_and_save(bucket_name, file_name, output_dir):
            success_count += 1
    print(f"\n{'='*60}")
    print(f"Successfully decrypted {success_count}/{len(files)} files from {bucket_name}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}")

def decrypt_specific_file(bucket_name, file_name, output_dir):
    print(f"\n{'='*60}")
    print(f"Decrypting: {file_name} from {bucket_name}")
    print(f"{'='*60}")
    if decrypt_and_save(bucket_name, file_name, output_dir):
        print(f"\nDecryption successful! Saved to: {output_dir}")
    else:
        print(f"\nDecryption failed!")

def main_menu():
    while True:
        print("\n" + "="*60)
        print("VIDEO DECRYPTION TOOL")
        print("="*60)
        print("1. Decrypt all RAW videos")
        print("2. Decrypt all MOTION videos")
        print("3. Decrypt all ANOMALY videos")
        print("4. Decrypt all PERSON images")
        print("5. Decrypt ALL files from all buckets")
        print("6. Decrypt specific file")
        print("7. List files in a bucket")
        print("8. Exit")
        print("="*60)
        choice = input("\nEnter your choice (1-8): ").strip()
        if choice == '1':
            decrypt_bucket('raw_videos', DECRYPTED_RAW)
        elif choice == '2':
            decrypt_bucket('motion_videos', DECRYPTED_MOTION)
        elif choice == '3':
            decrypt_bucket('anomaly_videos', DECRYPTED_ANOMALY)
        elif choice == '4':
            decrypt_bucket('person_images', DECRYPTED_PERSON)
        elif choice == '5':
            decrypt_bucket('raw_videos', DECRYPTED_RAW)
            decrypt_bucket('motion_videos', DECRYPTED_MOTION)
            decrypt_bucket('anomaly_videos', DECRYPTED_ANOMALY)
            decrypt_bucket('person_images', DECRYPTED_PERSON)
        elif choice == '6':
            print("\nAvailable buckets:")
            print("1. raw_videos")
            print("2. motion_videos")
            print("3. anomaly_videos")
            print("4. person_images")
            bucket_choice = input("Select bucket (1-4): ").strip()
            bucket_map = {
                '1': ('raw_videos', DECRYPTED_RAW),
                '2': ('motion_videos', DECRYPTED_MOTION),
                '3': ('anomaly_videos', DECRYPTED_ANOMALY),
                '4': ('person_images', DECRYPTED_PERSON)
            }
            if bucket_choice in bucket_map:
                bucket_name, output_dir = bucket_map[bucket_choice]
                file_name = input("Enter file name: ").strip()
                decrypt_specific_file(bucket_name, file_name, output_dir)
            else:
                print("Invalid bucket choice!")
        elif choice == '7':
            print("\nAvailable buckets:")
            print("1. raw_videos")
            print("2. motion_videos")
            print("3. anomaly_videos")
            print("4. person_images")
            bucket_choice = input("Select bucket (1-4): ").strip()
            bucket_map = {
                '1': 'raw_videos',
                '2': 'motion_videos',
                '3': 'anomaly_videos',
                '4': 'person_images'
            }
            if bucket_choice in bucket_map:
                bucket_name = bucket_map[bucket_choice]
                files = list_files_in_bucket(bucket_name)
                print(f"\nFiles in {bucket_name}:")
                print("-" * 60)
                if not files:
                    print("  (No files found)")
                else:
                    for i, file_info in enumerate(files, 1):
                        size_kb = file_info.get('metadata', {}).get('size', 0) / 1024
                        print(f"{i}. {file_info['name']} ({size_kb:.2f} KB)")
                print("-" * 60)
                print(f"Total: {len(files)} file(s)")
            else:
                print("Invalid bucket choice!")
        elif choice == '8':
            print("\nExiting... Goodbye!")
            break
        else:
            print("\nInvalid choice! Please select 1-8.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SURVEILLANCE VIDEO DECRYPTION TOOL")
    print("="*60)
    print("\nThis tool will decrypt encrypted videos from Supabase storage.")
    print(f"Decrypted files will be saved to:")
    print(f"  - Raw videos: {DECRYPTED_RAW}")
    print(f"  - Motion videos: {DECRYPTED_MOTION}")
    print(f"  - Anomaly videos: {DECRYPTED_ANOMALY}")
    print(f"  - Person images: {DECRYPTED_PERSON}")
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
