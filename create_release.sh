#!/bin/sh
VERSION=$(python setup.py --version)
VIRTUALENV=build/virtualenv

echo "\x1B[31m******* DANGER ********\x1B[0m"
echo Creating a new release for v$VERSION
echo This is going to upload files to upload files to AWS, create a github release and change the pypi registry
read -p 'Are you sure you want to proceed (y/n)? ' PROCEED

if [[ ! $PROCEED =~ ^[Yy]$ ]]
then
	exit 1
fi

$VIRTUALENV/bin/python3 setup.py sdist bdist_wheel
aws s3 sync dist/ s3://datalabs-packages/wellcomeml
aws s3 cp --recursive --acl public-read dist/ s3://datalabs-public/wellcomeml

echo $VERSION

curl --request POST \
  --url https://api.github.com/repos/wellcometrust/wellcomeml/releases \
  --header 'authorization: token '$GITHUB_TOKEN'' \
  --header 'content-type: application/json' \
  --data '{
  "tag_name": "v'$VERSION'",
  "target_commitish": "master",
  "name": "v'$VERSION'",
	"prerelease": true
}'


RELEASE_ID=$(curl -XGET --silent "https://api.github.com/repos/wellcometrust/WellcomeML/releases/tags/v$VERSION" | jq .id)

cd dist/

curl --request POST --silent --header "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $(file -b --mime-type wellcomeml-$VERSION.tar.gz)" --data-binary @wellcomeml-$VERSION.tar.gz --url "https://uploads.github.com/repos/wellcometrust/WellcomeML/releases/$RELEASE_ID/assets?name=wellcomeml-$VERSION.tar.gz"
curl --request POST --silent --header "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $(file -b --mime-type wellcomeml-$VERSION.tar.gz)" --data-binary @wellcomeml-$VERSION.tar.gz --url "https://uploads.github.com/repos/wellcometrust/WellcomeML/releases/$RELEASE_ID/assets?name=wellcomeml-$VERSION-py3-none-any.whl"

echo "Release created"
echo "Please change the description at https://github.com/wellcometrust/WellcomeML/releases/tag/v$VERSION"
