#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # isort:skip
from src import (convert, estimate_feature_statistics, estimate_twf_and_jnt,  # isort:skip # pylint: disable=C0413
                 extract_features, train_GMM)

USES = ("train","eval")
LIST_SUFFIXES = {
    use: "_" + use + ".list" for use in USES}

SOURCE_FOLDER = "/home/anurag/Downloads/l2arctic_release"
CONF_DIR = SOURCE_FOLDER + "/" + "conf"
DATA_DIR = SOURCE_FOLDER + "/" + "data"
LIST_DIR = SOURCE_FOLDER + "/" + "list"
WAV_DIR = SOURCE_FOLDER

def list_lengths_are_all_same(first_path, *remain_paths):

    def count_words_in_file(path):
        with open(str(path)) as handler:
            words = len(handler.read().split())
            return words

    n_words_in_first_file = count_words_in_file(first_path)
    return all((count_words_in_file(path) == n_words_in_first_file
                for path in remain_paths))   

if __name__ == "__main__":
    
    LABELS = {};
    LABELS["source"] = "BDL_0.99"
    LABELS["target"] = "ABA_16k"

    SOURCE_TARGET_PAIR = LABELS["source"] + "-" + LABELS["target"]
    PAIR_DIR = DATA_DIR + "/" + "pair" + "/" + SOURCE_TARGET_PAIR
    LIST_FILES = {
        speaker_part: {
            use: LIST_DIR + "/" + (speaker_label + LIST_SUFFIXES[use])
            for use in USES}
        for speaker_part, speaker_label in LABELS.items()}

    SPEAKER_CONF_FILES = {
        part: CONF_DIR + "/" + "speaker" + "/" + (label + ".yml")
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = CONF_DIR + "/pair/" + (SOURCE_TARGET_PAIR + ".yml")    
    
    for use in USES:
        list_lengths_are_all_same(
            *[list_files_per_part[use]
            for list_files_per_part in LIST_FILES.values()])
    print(LIST_FILES)
    print(SPEAKER_CONF_FILES)    
    #os.makedirs(str(PAIR_DIR), exist_ok=True)    

    for execute_steps in range(4,6): 

        if(execute_steps == 1):
            print("Extract acoustic features")
            for speaker_part, speaker_label in LABELS.items():            
                extract_features.main("--overwrite",
                    speaker_label, str(SPEAKER_CONF_FILES[speaker_part]),
                    str(LIST_FILES[speaker_part]['train']),
                    str(WAV_DIR), str(PAIR_DIR))   

        if(execute_steps==2):
            print("### 2. Estimate acoustic feature statistics ###")
            for speaker_part, speaker_label in LABELS.items():
                estimate_feature_statistics.main(
                    speaker_label, str(LIST_FILES[speaker_part]["train"]),
                    str(PAIR_DIR))        


        if(execute_steps==3):
            print("3. Estimate time warping function and jnt")
            estimate_twf_and_jnt.main(
                str(SPEAKER_CONF_FILES["source"]),
                str(SPEAKER_CONF_FILES["target"]),
                str(PAIR_CONF_FILE),
                str(LIST_FILES["source"]["train"]),
                str(LIST_FILES["target"]["train"]),
                str(PAIR_DIR))
        
        if(execute_steps==4):
            print("4.Train GMM and converted GV")
            train_GMM.main(
                str(LIST_FILES["source"]["train"]),
                str(PAIR_CONF_FILE),
                str(PAIR_DIR))

        if(execute_steps==5):
            print("### 5. Conversion based on the trained models ###")
            EVAL_LIST_FILE = LIST_FILES["source"]["eval"]
            # convertsion based on the trained GMM
            convert.main(
                LABELS["source"], LABELS["target"],
                str(SPEAKER_CONF_FILES["source"]),
                str(PAIR_CONF_FILE),
                str(EVAL_LIST_FILE),
                str(WAV_DIR),
                str(PAIR_DIR))
            convert.main(
                "-gmmmode", "diff",
                LABELS["source"], LABELS["target"],
                str(SPEAKER_CONF_FILES["source"]),
                str(PAIR_CONF_FILE),
                str(EVAL_LIST_FILE),
                str(WAV_DIR),
                str(PAIR_DIR))



