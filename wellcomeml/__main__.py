import subprocess
import sys


def download(download_target):
    if download_target == "models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    elif download_target == "deeplearning-models":
        subprocess.run([
            'python', '-m', 'spacy', 'download', 'en_core_web_trf'])
    elif download_target == "non_pypi_packages":
        # This is a workaround to pin sent2vec
        sent_2_vec_commit = 'f00a1b67f4330e5be99e7cc31ac28df94deed9ac'

        subprocess.run([
            'pip', 'install', f'git+https://github.com/epfml/sent2vec.git@{sent_2_vec_commit}'])
    else:
        print(f"{download_target} is not one of models,deeplearning-models")


if __name__ == '__main__':
    command = sys.argv.pop(1)
    if command != "download":
        print("Only available command is download")
        exit()

    download_target = sys.argv.pop(1)
    download(download_target)
