from wallet_coeziv import WalletCohesiv
from tranzactii_coeziv import TxInput, create_cohesive_tx_from_wallet

# 1. Creăm wallet-ul
wallet = WalletCohesiv()
print("Adresa walletului:", wallet.address)

# 2. Inventăm un UTXO (fonduri disponibile) - laborator
utxo = TxInput(
    source_tx_id="abc123",
    source_index=0,
    address=wallet.address,
    amount_sats=100_000
)

# 3. Adresă destinație (poate fi orice string)
destinatie = "tb1qDESTINATIE0000000000000000000"

# 4. Creăm o tranzacție coezivă
tx = create_cohesive_tx_from_wallet(
    wallet=wallet,
    available_utxo=utxo,
    destination_address=destinatie,
    amount_sats=30_000,
    fee_sats=1_000
)

# 5. Afișăm tranzacția
from pprint import pprint
pprint(tx.to_public_dict())
