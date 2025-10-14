import re

def code_mixing_index(text):
    eng_words = len(re.findall(r"[a-zA-Z]+", text))
    dev_words = len(re.findall(r"[\u0900-\u097F]+", text))
    total = eng_words + dev_words
    if total == 0:
        return 0
    return round((eng_words / total) * 100, 2)

sample = "Mujhe bohot stress hai yaar, exams ke baad bhi dimag relax nahi ho raha"
print("Code-mixing index:", code_mixing_index(sample), "%")
