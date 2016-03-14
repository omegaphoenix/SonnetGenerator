import random


input_file = 'complete_bigram_rhyme_poem5.txt'
output_file = 'output.txt'
i = 0
random.seed(13)
for line in open(input_file, 'r'):
    new_line = line.capitalize().strip()
    new_line = new_line.replace(" i ", " I ")
    i += 1
    if i % 4 == 0 or i == 14:
        if random.random() < 0.8:
            new_line += ".\n"
        else:
            new_line += "?\n"
    else:
        if random.random() < 0.8:
            new_line += ","
    if i > 12:
        new_line = "  " + new_line
    print new_line
