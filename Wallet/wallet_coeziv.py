import os
import hashlib
import hmac
import json

# ==============================
#  UTILE
# ==============================

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def b58encode(b: bytes) -> str:
    """Encodare Base58 simplă (pentru adrese), fără dependințe externe."""
    # Transformă bytes în integer mare
    num = int.from_bytes(b, byteorder="big")
    encode = ""
    while num > 0:
        num, rem = divmod(num, 58)
        encode = BASE58_ALPHABET[rem] + encode

    # Prefixează cu '1' pentru fiecare zero inițial în bytes
    leading_zeros = 0
    for byte in b:
        if byte == 0:
            leading_zeros += 1
        else:
            break
    return "1" * leading_zeros + encode


# ==============================
#  NCW – NUCLEUL COEZIV AL WALLETULUI
# ==============================

class WalletCohesiv:
    """
    Model educativ de wallet Bitcoin cu Nucleu Coeziv (NCW).
    NU este implementare completă BIP39/BIP32 și NU se folosește pentru bani reali.
    """

    def __init__(self, entropy: bytes | None = None):
        # -----------------------------
        # ESC – ENTROPIE STRUCTURALĂ COEZIVĂ
        # -----------------------------
        # Dacă nu primim entropie din afară, o generăm local.
        if entropy is None:
            entropy = os.urandom(16)  # 128 bits – minimal educativ

        self.entropy = entropy  # rădăcina internă, nu se expune în mod normal

        # Seed derivat din entropie (simplificat, nu BIP39 real)
        self.seed = self._entropy_to_seed(self.entropy)

        # -----------------------------
        # SDD – STRAT DE DERIVARE DETERMINISTĂ
        # -----------------------------
        # Din seed generăm o "cheie master" privată – model simplificat
        self.master_private_key = self._seed_to_master_private_key(self.seed)

        # Pentru exemplu: derivăm o singură cheie de cont / adresă (index 0)
        self.account_index = 0
        self.private_key = self._derive_child_key(self.master_private_key, self.account_index)

        # "Cheie publică" – aici folosim un substitut hash, nu SECP256k1 real!
        self.public_key = self._fake_public_key(self.private_key)

        # -----------------------------
        # SEC – STRAT DE EXPUNERE CONTROLATĂ
        # -----------------------------
        # Adresă Bitcoin de TESTNET (nu mainnet)
        self.address = self._public_key_to_testnet_address(self.public_key)

    # ====================================
    # ESC – FUNCȚII PENTRU ENTROPIE & SEED
    # ====================================

    @staticmethod
    def _entropy_to_seed(entropy: bytes) -> bytes:
        """
        ESC/SDD: din entropie obținem un seed stabil.
        În realitate, BIP39 ar folosi mnemonic + salt + PBKDF2.
        Aici simplificăm la SHA256(entropie) pentru educație.
        """
        return hashlib.sha256(entropy).digest()

    @staticmethod
    def _seed_to_master_private_key(seed: bytes) -> bytes:
        """
        SDD: transformă seed-ul într-o cheie master privată.
        În implementări reale se folosește HMAC-SHA512 cu un "key" specific (ex: "Bitcoin seed").
        Aici rămânem simpli.
        """
        return hashlib.sha256(seed).digest()

    # ====================================
    # SDD – DERIVARE DE CHEI
    # ====================================

    @staticmethod
    def _derive_child_key(master_priv: bytes, index: int) -> bytes:
        """
        SDD: derivare deterministă a unei chei copil din master + index.
        Versiune educativă: HMAC-SHA256(master_priv, index_bytes).
        """
        index_bytes = index.to_bytes(4, "big")
        return hmac.new(master_priv, index_bytes, hashlib.sha256).digest()

    @staticmethod
    def _fake_public_key(private_key: bytes) -> bytes:
        """
        SDD: placeholder de cheie publică.
        În realitate: cheie privată pe curva secp256k1 -> cheie publică (ECDSA).
        Aici: hash simplu pentru a putea merge mai departe conceptual.
        """
        return hashlib.sha256(private_key).digest()

    # ====================================
    # SEC – EXPUNERE CONTROLATĂ (ADRESĂ)
    # ====================================

    @staticmethod
    def _public_key_to_testnet_address(pubkey: bytes) -> str:
        """
        SEC: adresa care este expusă în lume.
        - Hash160 (RIPEMD160(SHA256(pubkey)))
        - Prefix testnet (0x6F)
        - Checksum = primele 4 bytes din dublu SHA256
        - Encodare Base58
        """
        sha = hashlib.sha256(pubkey).digest()
        ripemd = hashlib.new("ripemd160")
        ripemd.update(sha)
        hashed = ripemd.digest()

        prefix = b"\x6F"  # testnet
        prefixed = prefix + hashed
        checksum = hashlib.sha256(hashlib.sha256(prefixed).digest()).digest()[:4]
        address_bytes = prefixed + checksum
        return b58encode(address_bytes)

    # ====================================
    # SEC – INTERFAȚĂ PUBLICĂ
    # ====================================

    def export_public_view(self) -> dict:
        """
        SEC: ce putem expune unui UI / API extern.
        Nu arătăm chei private, doar informații "safe".
        """
        return {
            "type": "wallet_coeziv_demo",
            "network": "testnet",
            "account_index": self.account_index,
            "address": self.address,
        }

    def export_internal_debug(self) -> dict:
        """
        DOAR PENTRU LABORATOR: expunem tot nucleul.
        Nu faci așa ceva în producție!
        """
        return {
            "entropy_hex": self.entropy.hex(),
            "seed_hex": self.seed.hex(),
            "master_private_key_hex": self.master_private_key.hex(),
            "derived_private_key_hex": self.private_key.hex(),
            "public_key_hex": self.public_key.hex(),
            "address": self.address,
        }

    def save_to_file(self, path: str, include_private: bool = False) -> None:
        """
        SEC: exemplu de persistență.
        - dacă include_private=False: salvăm doar view public + metadate.
        - dacă include_private=True: laborator, nu productie!
        """
        data = {
            "public": self.export_public_view(),
        }
        if include_private:
            data["internal_debug"] = self.export_internal_debug()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ==============================
# EXEMPLE DE UTILIZARE
# ==============================

if __name__ == "__main__":
    # Creăm un wallet coeziv nou
    wallet = WalletCohesiv()

    print("=== VEDERE PUBLICĂ (SEC) ===")
    print(json.dumps(wallet.export_public_view(), indent=2))

    print("\n=== VEDERE INTERNĂ (LABORATOR, NU PRODUCȚIE) ===")
    print(json.dumps(wallet.export_internal_debug(), indent=2))

    # Salvăm walletul într-un fișier (exemplu)
    wallet.save_to_file("wallet_coeziv_demo.json", include_private=True)
    print("\nWallet salvat în 'wallet_coeziv_demo.json' (doar pentru laborator).")
