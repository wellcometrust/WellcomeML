import json
from io import BytesIO
import gzip
import os


class PolicyDocumentsDownloader:
    """
    Interact with S3 to get policy document texts

    Args:
        bucket_name: the S3 bucket name to get data from
        dir_path: the directory path within this s3 bucket to get policy data from
    """
    def __init__(self, s3, bucket_name, dir_path):

        self.s3 = s3
        self.bucket_name = bucket_name
        self.dir_path = dir_path
        self.pdf_keys = self.get_all_s3_keys()

    def get_all_s3_keys(self):
        """
        https://alexwlchan.net/2017/07/listing-s3-keys/
        Get a list of all keys to look for pdfs in the S3 bucket.
        """
        keys = []

        kwargs = {'Bucket': self.bucket_name}
        while True:
            resp = self.s3.list_objects_v2(**kwargs)
            for obj in resp['Contents']:
                keys.append(obj['Key'])

            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

        pdf_keys = [k for k in keys if self.dir_path in k]

        return pdf_keys

    def get_hashes(self, word_list=None):
        """
        Get a list of policy document hashes from the S3 location

        Args:
            word_list(list): a list of words to look for in documents, if None then get hashes for
                all
        Returns:
            list: a list of dicts with the file hash and the policy doc source where it is from
        """

        print("Getting hashes for policy documents")
        hashes = []
        for key in self.pdf_keys:
            print("Loading "+key)
            key_name = os.path.split(key)[-1]
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read()
            with gzip.GzipFile(fileobj=BytesIO(content), mode='rb') as fh:
                for line in fh:
                    document = json.loads(line)
                    if document['text']:
                        if word_list:
                            if not any(word.lower() in document['text'].lower()
                                       for word in word_list):
                                continue
                        hashes.append({
                            "source": key_name,
                            "file_hash": document['file_hash']
                        })
                print(str(len(hashes))+" documents")

        return hashes

    def download(self, hash_list=None):
        """
        Download the policy document data from S3

        Args:
            hash_list: a list of hashes to specifically download, if None then download all
        """

        print("Downloading policy documents")
        documents = []
        hashes_found = set()  # A checker so we dont download duplicates
        for key in self.pdf_keys:
            print("Loading "+key)
            key_name = os.path.split(key)[-1]
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read()
            with gzip.GzipFile(fileobj=BytesIO(content), mode='rb') as fh:
                for line in fh:
                    document = json.loads(line)
                    if ((document['text']) and (document['file_hash'] not in hashes_found)):
                        if hash_list:
                            if document['file_hash'] not in set(hash_list):
                                continue
                        document["source"] = key_name
                        documents.append(document)
                        hashes_found.add(document['file_hash'])

                print(str(len(documents))+" documents")

        return documents

    def save_json(self, documents, file_name):
        print("Saving data ...")
        with open(file_name, 'w', encoding='utf-8') as output_file:
            for document in documents:
                json.dump(document, output_file)
                output_file.write("\n")
        print("Number of documents saved: " + str(len(documents)))
