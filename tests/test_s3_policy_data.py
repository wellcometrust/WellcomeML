#!/usr/bin/env python3
# coding: utf-8
import boto3
from botocore.stub import Stubber

from wellcomeml.io import s3_policy_data


def stubber_responses(stubber, mock_hash_file=None):

    list_buckets_response = {
        "Contents": [
            {
                "Key": "good/path/file1.json"
            },
            {
                "Key": "bad/path/file2.json"
            }
        ]
        }
    expected_params = {'Bucket': 'datalabs-dev'}
    stubber.add_response('list_objects_v2', list_buckets_response, expected_params)

    if mock_hash_file:
        get_object_response = {
            "Body": mock_hash_file
            }
        expected_params = {'Bucket': 'datalabs-dev', 'Key': 'good/path/file1.json'}
        stubber.add_response('get_object', get_object_response, expected_params)

    return stubber


def policy_downloader(s3):
    return s3_policy_data.PolicyDocumentsDownloader(
            s3=s3,
            bucket_name="datalabs-dev",
            dir_path="good/path"
            )


def test_get_keys():

    s3 = boto3.client('s3')
    stubber = Stubber(s3)
    stubber = stubber_responses(stubber)

    with stubber:
        policy_s3 = policy_downloader(s3)
        pdf_keys = policy_s3.pdf_keys

    assert pdf_keys == ['good/path/file1.json']


def test_get_hashes_with_word():

    s3 = boto3.client('s3')
    stubber = Stubber(s3)

    with open('tests/test_data/mock_s3_contents.json.gz', 'rb') as mock_hash_file:
        stubber = stubber_responses(stubber, mock_hash_file)

        with stubber:
            policy_s3 = policy_downloader(s3)
            hash_dicts = policy_s3.get_hashes(word_list=['the'])
    hash_list = [hash_dict['file_hash'] for hash_dict in hash_dicts]

    assert hash_list == ['x002']


def test_get_hashes():

    s3 = boto3.client('s3')
    stubber = Stubber(s3)

    with open('tests/test_data/mock_s3_contents.json.gz', 'rb') as mock_hash_file:
        stubber = stubber_responses(stubber, mock_hash_file)

        with stubber:
            policy_s3 = policy_downloader(s3)
            hash_dicts = policy_s3.get_hashes()
    hash_list = [hash_dict['file_hash'] for hash_dict in hash_dicts]
    hash_list.sort()

    assert hash_list == ['x001', 'x002']


def test_download_all_hash():

    s3 = boto3.client('s3')
    stubber = Stubber(s3)

    with open('tests/test_data/mock_s3_contents.json.gz', 'rb') as mock_hash_file:
        stubber = stubber_responses(stubber, mock_hash_file)

        with stubber:
            policy_s3 = policy_downloader(s3)
            documents = policy_s3.download(hash_list=None)

    document_hashes = [document['file_hash'] for document in documents]
    document_hashes.sort()

    assert document_hashes == ['x001', 'x002']


def test_download_one_hash():

    s3 = boto3.client('s3')
    stubber = Stubber(s3)

    with open('tests/test_data/mock_s3_contents.json.gz', 'rb') as mock_hash_file:
        stubber = stubber_responses(stubber, mock_hash_file)

        with stubber:
            policy_s3 = policy_downloader(s3)
            documents = policy_s3.download(hash_list=['x002'])

    document_hashes = [document['file_hash'] for document in documents]
    assert document_hashes == ['x002']
