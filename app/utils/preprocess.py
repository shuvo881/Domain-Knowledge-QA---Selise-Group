

import re
from typing import List


def clean_text(text: str) -> str:
    """ Remove special characters and extra whitespace """

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size=500) -> List[str]:
    """ Split text into chunks of specified size """
    
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
