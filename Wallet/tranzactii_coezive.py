import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import List


@dataclass
class TxInput:
    """
    Reprezintă o sursă de fonduri (UTXO) în modelul nostru educativ.
    În Bitcoin real ar avea: txid, vout, scriptPubKey, amount, etc.
    Aici simplificăm.
    """
    source_tx_id: str      # ID-ul tranzacției din care vine
    source_index: int      # indexul output-ului folosit
    address: str           # adresa "deținătoare"
    amount_sats: int       # valoarea în satoshi


@dataclass
class TxOutput:
    """
    Reprezintă o ieșire (destinație) de fonduri.
    """
    address: str
    amount_sats: int


class TxCohesiv:
    """
    Tranzacție coezivă educativă, NU tranzacție Bitcoin reală.
    - are intrări + ieșiri
    - are un ID (hash) determinist
    - are o "semnătură coezivă" bazată pe cheia privată a walletului
    """

    def __init__(self, inputs: List[TxInput], outputs: List[TxOutput], created_at: float | None = None):
        self.inputs = inputs
        self.outputs = outputs
        self.created_at = created_at if created_at is not None else time.time()

        # ID-ul tranzacției (hash al conținutului + timestamp)
        self.tx_id = self._compute_tx_id()

        # Semnătura coezivă – NU ESTE SEMNĂTURĂ BITCOIN REALĂ
        self.signature_hex: str | None = None

    # -------------------------------
    # STRUCTURĂ COEZIVĂ – ID TRANZACȚIE
    # -------------------------------
    def _compute_tx_id(self) -> str:
        """
        ID-ul tranzacției = SHA256( serialize(inputs, outputs, created_at) )
        """
        h = hashlib.sha256()
        # serializăm intrările
        for txin in self.inputs:
            h.update(txin.source_tx_id.encode())
            h.update(txin.source_index.to_bytes(4, "big"))
            h.update(txin.address.encode())
            h.update(txin.amount_sats.to_bytes(8, "big"))
        # serializăm ieșirile
        for txout in self.outputs:
            h.update(txout.address.encode())
            h.update(txout.amount_sats.to_bytes(8, "big"))

        # adăugăm timestamp-ul
        h.update(str(self.created_at).encode())
        return h.hexdigest()

    # -------------------------------
    # FLUX COEZIV – SEMNARE
    # -------------------------------
    def sign_with_private_key(self, private_key: bytes) -> str:
        """
        "Semnătură" coezivă, educativă:
        HMAC-SHA256(private_key, tx_id)
        În Bitcoin real se folosește ECDSA pe curba secp256k1, cu digest al tranzacției.
        """
        sig = hmac.new(private_key, self.tx_id.encode(), hashlib.sha256).hexdigest()
        self.signature_hex = sig
        return sig

    # -------------------------------
    # SEC – EXPUNERE CONTROLATĂ
    # -------------------------------
    def to_public_dict(self) -> dict:
        """
        Ce poate vedea lumea despre tranzacție.
        Nu include cheia privată, evident.
        """
        return {
            "tx_id": self.tx_id,
            "created_at": self.created_at,
            "inputs": [
                {
                    "source_tx_id": i.source_tx_id,
                    "source_index": i.source_index,
                    "address": i.address,
                    "amount_sats": i.amount_sats,
                }
                for i in self.inputs
            ],
            "outputs": [
                {
                    "address": o.address,
                    "amount_sats": o.amount_sats,
                }
                for o in self.outputs
            ],
            "signature_hex": self.signature_hex,
        }

    def total_input_sats(self) -> int:
        return sum(i.amount_sats for i in self.inputs)

    def total_output_sats(self) -> int:
        return sum(o.amount_sats for o in self.outputs)

    def fee_sats(self) -> int:
        """
        Fee = total_input - total_output
        Dacă e negativ, tranzacția e invalidă logic.
        """
        return self.total_input_sats() - self.total_output_sats()

    def is_balanced(self) -> bool:
        """
        Verifică dacă tranzacția are fee >= 0.
        """
        return self.fee_sats() >= 0
