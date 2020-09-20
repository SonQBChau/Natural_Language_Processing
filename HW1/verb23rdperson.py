import argparse
import re

def read_verbs(file_path):
    with open(file_path) as f:
        for verb in f.readlines():
            verb = verb.rstrip()
            if len(verb) < 1:
                continue
            yield verb.rstrip()


def get_3rdperson(verb):
    # YOUR CODE GOES HERE
    # Use the re module to do string matching. A good solution is as short as 5 lines

    pattern_1 = "(.+((ss)|(ch)|(sh)|[xo])$)"
    pattern_2 = "(.*[b-df-hj-np-tv-z]y$)"
    third_person_verb = '_3rdperson'
    
    if re.match(pattern_1, verb): 
        third_person_verb = 'es'
    elif re.match(pattern_2, verb): 
        third_person_verb = 'ies'
        verb = (re.sub('y$', '',verb))
    else: 
        third_person_verb = 's'
    

    verb_3rdperson = verb + third_person_verb
    return verb_3rdperson


def main(file_path):
    for verb in read_verbs(file_path):
        verb3rdperson = get_3rdperson(verb)
        print(f"{verb:10} {verb3rdperson}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE_PATH",
                        help="Path to file with verbs in their base form, one verb per line")
    args = parser.parse_args()

    main(args.FILE_PATH)