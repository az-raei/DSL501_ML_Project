import langid
import re

def sarcasm_score(text):
    return int(bool(re.search(r"/s|sarcasm|yeah right|sure you are", text.lower())))

def code_mixing_ratio(text):
    words = text.split()
    en = sum(langid.classify(w)[0] == "en" for w in words)
    hi = sum(langid.classify(w)[0] in ["hi", "mr", "ur"] for w in words)
    if en+hi == 0: return 0
    return min(en, hi) / max(en, hi)

def cultural_mismatch(response):
    formal_markers = ["मैं मान्यता देता हूँ", "मैं आपकी भावनाओं को समझता हूँ"]
    score = sum(fm in response for fm in formal_markers)
    return score
