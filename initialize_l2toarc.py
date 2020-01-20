#!/usr/bin/env/ python3
import os
import shutil
import sys
from pathlib import Path
from docopt import docopt
from src import initialize_speaker

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

SOURCE_FOLDER = "/home/anurag/Downloads/l2arctic_release"
USES = ("train", "eval")
LIST_SUFFIXES = {use: "_" + use + ".list" for use in USES}

SOURCE_FOLDER = "/home/anurag/Downloads/l2arctic_release"
CONF_DIR = SOURCE_FOLDER + "/" + "conf"
DATA_DIR = SOURCE_FOLDER + "/" + "data"
LIST_DIR = SOURCE_FOLDER + "/" + "list"
WAV_DIR = SOURCE_FOLDER

def create_configure(dest,base,exist_ok=True):
	if not(os.path.exists(os.path.split(dest)[0])):
		os.makedirs(os.path.split(dest)[0])
	if not(os.path.exists(os.path.split(base)[0])):
		os.makedirs(os.path.split(base)[0])

	if os.path.exists(str(dest)):  # Wrapping in str is for Python 3.5
		message = "The configuration file {} already exists.".format(dest)
		if exist_ok:
			print(message)
		else:
			raise FileExistsError(message)
	else:
		print("Generate {}".format(dest), file=sys.stderr)
		shutil.copy(str(base), str(dest))

def create_list(dest,wav_dir,exist_ok=True):
	if not(os.path.isdir(os.path.split(dest)[0])):
		os.makedirs(os.path.split(dest)[0])

	if(os.path.exists(str(dest))):
		message="The list file {} already exists.".format(dest)	
		if exist_ok:
			print(message)
		else:
			raise FileExistsError(message)
	else:
		print("Generate {}".format(dest))
		speaker_label = os.path.basename(os.path.split(str(wav_dir))[0])
		print(speaker_label)	
	print(dest)	
	lines = []
	for wav_file_name in os.listdir(wav_dir)[0:100]:
		if ".wav" in wav_file_name:
			lines.append(speaker_label + "/" +"wav" + "/" + os.path.splitext(wav_file_name)[0])	
		
	for wav_file_name in os.listdir(wav_dir)[100:150]:
		if ".wav" in wav_file_name:
			lines.append(speaker_label + "/" +"wav" + "/" + os.path.splitext(wav_file_name)[0])	
	
	if "train" in dest:
		lines = sorted(lines)[0:100]
	if "eval" in dest:
		lines = sorted(lines)[100:150]	
	with open(str(dest),"w") as file_handler:
		for line in sorted(lines):
			print(line, file=file_handler)

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
			speaker_part: CONF_DIR + "/" + "speaker" + "/" + speaker_label + ".yml"
		for speaker_part, speaker_label in LABELS.items()} 
	
	PAIR_CONF_FILE = CONF_DIR + "/" + "pair" + "/" + SOURCE_TARGET_PAIR + ".yml"
	SAMPLING_RATE = 16000
	print(LIST_FILES)
	execute_steps = 3


	if(execute_steps == 1):
		print("###create initial file list")
		for use in USES:
			for part, speaker in LABELS.items():
				create_list(LIST_FILES[part][use],WAV_DIR + "/" + speaker + "/" + "wav")

	if(execute_steps == 2):
		print("##create configure files")
		for part, speaker in LABELS.items():
			create_configure(
			SPEAKER_CONF_FILES[part],
			CONF_DIR + "/" + "default" + "/" + "speaker_default_{}.yml".format(SAMPLING_RATE))
		create_configure(PAIR_CONF_FILE,
			os.path.join(str(CONF_DIR),"default","pair_default.yml"))	

	if(execute_steps == 3):
		print("create figures to define parameters")
		for part, speaker in LABELS.items():
			print(speaker)
			initialize_speaker.main(
			speaker, str(LIST_FILES[part]["train"]),
			str(WAV_DIR), str(CONF_DIR + "/" + "figure")		
			)	
				
