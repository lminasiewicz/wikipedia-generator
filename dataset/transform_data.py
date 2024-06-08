import io
import re

TITLE_LENGTH = 20
FILEPATH = "./wikipedia.txt"

with io.open(FILEPATH, "r", encoding="utf-8") as wikipedia:
    raw_text = wikipedia.readlines()

with io.open(FILEPATH, "w", encoding="utf-8") as wikipedia:
    wikipedia.write("")

with io.open(FILEPATH, "a", encoding="utf-8") as wikipedia:
    for line in raw_text:
        if len(line) > TITLE_LENGTH and not bool(re.search(r'[^\x00-\x7F]', line)):
            wikipedia.write(line) # This is really fast somehow.

# Make sure to save a copy of the data if you wish to have the article titles and empty lines.