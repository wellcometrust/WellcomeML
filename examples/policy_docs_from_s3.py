import boto3

from wellcomeml.io import PolicyDocumentsDownloader

word_list = ["malaria"]

s3 = boto3.client("s3")
policy_s3 = PolicyDocumentsDownloader(
    s3=s3,
    bucket_name="datalabs-dev",
    dir_path="reach-airflow/output/policy/parsed-pdfs",
)
hash_dicts = policy_s3.get_hashes(word_list=word_list)

hash_list = [hash_dict["file_hash"] for hash_dict in hash_dicts]

print(hash_list[0:10])

documents = policy_s3.download(hash_list=hash_list[0:10])

# Get the first 100 characters of the text from
# these documents
print([d["text"][0:100] for d in documents])
