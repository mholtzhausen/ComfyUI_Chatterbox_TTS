from typing import Dict

class AudiobookSection:
    def __init__(self, text: str, narrator_details: Dict, characters_details: Dict, index: int, filename: str):
        self.text: str = text
        self.narrator: Dict = narrator_details
        self.characters: Dict = characters_details
        self.index: int = index
        self.filename: str = filename