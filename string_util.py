from typing import List, Tuple, Callable
import unicodedata

import torch

EOT_MARKER = '\x03'


def normalize_string(inp: str, normalize_unicode: bool = True, strip_accents: bool = True, strip_control: bool = True,
                     normalize_space: bool = True, lower_case: bool = True,
                     custom_transforms: List[Tuple[Callable, Callable]] = None,
                     eos_marker: bool = True) -> str:
    if custom_transforms is None:
        custom_transforms = []
    normalized = unicodedata.normalize('NFKD', inp) if normalize_unicode else inp
    res = []
    for c in normalized:
        if strip_accents and unicodedata.combining(c):
            continue
        if strip_control and unicodedata.category(c)[0] == 'C':
            continue
        if normalize_space and unicodedata.category(c)[0] == 'Z':
            c = ' '
        if lower_case:
            c = c.lower()
        for p, t in custom_transforms:
            if p(c):
                c = t(c)
        res.append(c)
    if eos_marker:
        res.append(EOT_MARKER)
    return ''.join(res)


def encode_string(inp: str, alphabet: str) -> List[int]:
    encoded = []
    for c in inp:
        try:
            i = alphabet.index(c)
            encoded.append(i)
        except ValueError:
            if c != EOT_MARKER:
                encoded.append(len(alphabet))
            else:
                encoded.append(len(alphabet) + 1)
    return encoded


def encode_and_pack_list(inp: List[str], alphabet: str, max_chain_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_string(normalize_string(w, eos_marker=False), alphabet) for w in inp]
    lens = torch.tensor([len(w) for w in encoded])
    encoded = torch.tensor([w + [0] * max(0, max_chain_length - int(l))
                            for (w, l) in zip(encoded, lens)])
    return lens, encoded


def decode_string(inp: List[int], alphabet: str, unknown_marker='�') -> str:
    decoded = []
    for i in inp:
        if i == len(alphabet):
            decoded.append(unknown_marker)
        elif i == len(alphabet) + 1:
            decoded.append(EOT_MARKER)
        else:
            decoded.append(alphabet[i])
    return ''.join(decoded)
