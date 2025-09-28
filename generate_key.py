from Crypto.Random import get_random_bytes

def simulate_pqc_key_exchange():
    return get_random_bytes(32)

shared_key = simulate_pqc_key_exchange()
with open("aes_key.bin", "wb") as f:
    f.write(shared_key)

print("Shared AES key generated and saved to aes_key.bin")
