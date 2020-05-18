import subprocess
import sys
import os


if __name__ == '__main__':
    command = sys.argv.pop(1)
    if command != "download":
        print("Only available command is download")
        exit()

    download_target = sys.argv.pop(1)
    if download_target == "models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    elif download_target == "deeplearning-models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_trf_bertbaseuncased_lg'])
    else:
        print(f"{download_target} is not one of models,deeplearning-models")
