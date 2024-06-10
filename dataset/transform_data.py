import io
import re
from random import random

TITLE_LENGTH = 20
FILEPATH = "./wikipedia.txt"
FILEPATH2 = "./wikipedia_small.txt"
SMALL_SIZE = 0.25



def transform_file(filepath: str, mode = "normal"):
    with io.open(filepath, "w", encoding="utf-8") as wikipedia:
        wikipedia.write("")

    with io.open(filepath, "a", encoding="utf-8") as wikipedia:
        if mode == "normal":
            for line in raw_text:
                if len(line) > TITLE_LENGTH and not bool(re.search(r'[^\x00-\x7F]', line)):
                    wikipedia.write(line) # This is really fast somehow.
        
        elif mode == "small":
            for line in raw_text:
                if len(line) > TITLE_LENGTH and not bool(re.search(r'[^\x00-\x7F]', line)) and random() < SMALL_SIZE:
                    wikipedia.write(line) # This is really fast somehow.

# Make sure to save a copy of the data if you wish to have the article titles and empty lines.

if __name__ == "__main__":
    with io.open(FILEPATH, "r", encoding="utf-8") as wikipedia:
        raw_text = wikipedia.readlines()
    
    transform_file(FILEPATH2, mode="small")