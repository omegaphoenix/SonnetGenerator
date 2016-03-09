# This code processes the complete_shakespeare.txt to remove the chaf.

# Consider each work as its own token. So that means start
# thinking when we get a line with only a number on it.
# Have to do it separately for the sonnets.
import string
__LINE_NO = 0

def is_year(s):
    """ Checks if a string is a year. The only allowable
    years are those that Shakespeare worked, so four digit
    numbers starting with 15 or 16. """
    try:
        int(s)
    except ValueError, e:
        return False
    else:
        if 1500 <= int(s) and int(s) < 1700:
            return True
        return False

def process_play(f,g):
    """ Assumes that f is a file object and that 
    readline is currently pointing to a year value."""

    line = f.readline()
    global __LINE_NO
    __LINE_NO += 1
    while line != "" or "THE END" in line:
        if "ACT " in line[:3] or "SCENE" in line[:5]:
            process_scene(f,g)
        line = f.readline()
        __LINE_NO += 1
        if __LINE_NO >= 124370:
            return
    g.write("\n")

def process_scene(f,g):
    """ Assumes that we are starting at a line beginning with
    ACT or SCENE because that is how every scene begins """
    global __LINE_NO
    line = f.readline()
    __LINE_NO += 1
    print line
    while len(line.split()) == 0 or line.split()[0].upper() != line.split()[0]:
        line = f.readline()
        __LINE_NO += 1
    while line != "":
        if "ACT " in line[:3] or "THE END" in line or "SCENE" in line[:5]:
            return
        line_list = line.split()
        for i in line_list:
            i = i.translate(string.maketrans("",""), string.punctuation)
            if i == "I":
                g.write(i + "\n")
            elif i.upper() == i:
                pass
            elif i in ["Exit","Exeunt","Exit.","Exuent."]:
                break
            else:
                g.write(i + "\n")
        line = f.readline()
        __LINE_NO += 1
        if __LINE_NO >= 124370:
            return

    

# Delete all lines starting with << 

f = open("complete_shakespeare.txt", "r")
g = open("complete_shakespeare_words.txt", "w")

line = f.readline()
__LINE_NO += 1
while line != "":
    if is_year(line):
        # Now we know that we are starting a new work
        # because we have a new year written down.
        line = f.readline()
        line = f.readline() # Now we have the title
        __LINE_NO += 2
        if "THE SONNETS" in line:
            # We need a special case for the sonnets because the plays
            # have spoken lines
            line = f.readline()
            line = f.readline() # Now we are up to the author line
            __LINE_NO += 2
            while line != "":
                line = f.readline()
                __LINE_NO += 1
                line_list = line.split()
                if len(line_list) == 1: # We have a number, so new sonnet
                    print line
                    if line_list[0] == "1":
                        pass
                    # Not sure if we want to exclude these two
                    # elif line_list[0] == "99" or line_list[0] == "126":
                    #     continue
                    else:
                        g.write("\n")
                else:
                    print line
                    for word in line_list:
                        g.write(word)
                        g.write("\n")
                if "THE END" in line:
                    print line
                    g.write("\n")
                    break
        else:
            # Then it must be a play.
            # print line
            # while "THE END" not in line:
            #     if "ACT " in line or "SCENE " in line:
            #         g.write("\n")
            #         line = f.readline()
            #         line = f.readline()
            #         line = f.readline()
            #         line = f.readline()
            #     line = f.readline()
            #     line_list = line.split()
            #     if len(line_list) > 1 and line_list[0].upper() == line_list[0]: # Removes the name
            #         for i in line_list:
            #             if i.upper() == i and i != "I": # Multiword names
            #                 continue
            #             if i not in ["Exit","Exeunt","Exit.","Exuent."]:
            #                 if "[" in i: # removes [Aside] and [Kneeling] etc.
            #                     continue
            #                 # TODO remove the punctuation
            #                 i = i.translate(string.maketrans("",""), string.punctuation)
            #                 g.write(i)
            #                 g.write("\n")
            #             else:
            #                 print line_list
            #                 break
            process_play(f,g)
                        
    else:
        line = f.readline()
        __LINE_NO += 1
    if __LINE_NO >= 124370:
            exit()


