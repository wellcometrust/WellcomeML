import subprocess
import sys


def download(download_target):
    if download_target == "models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    elif download_target == "deeplearning-models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_trf_bertbaseuncased_lg'])
    elif download_target == "non_pypi_packages":
        subprocess.run([
            'pip', 'install', 'git+https://github.com/epfml/sent2vec.git'])
    else:
        print(f"{download_target} is not one of models,deeplearning-models")


if __name__ == '__main__':
    command = sys.argv.pop(1)
    if command != "download":
        print("Only available command is download")
        exit()

    download_target = sys.argv.pop(1)
    download(download_target)
