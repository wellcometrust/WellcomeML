VERSION=$(python setup.py --version)

RELEASE=$(curl --request post --silent  \
  --url https://api.github.com/repos/wellcometrust/wellcomeml/releases \
  --header 'content-type: application/json' \
  --header "authorization: token $GITHUB_TOKEN" \
  --data '{
  "tag_name": "v'$version'",
  "target_commitish": "master",
  "name": "v'$version'",
  "body": "pre-release '$version'",
  "draft": true,
  "prerelease": true
}')

RELEASE_ID=$(curl -XGET --silent "https://api.github.com/repos/wellcometrust/WellcomeML/releases/tags/v$VERSION" | jq .id)

cd dist/

curl --request POST --silent --header "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $(file -b --mime-type wellcomeml-$VERSION.tar.gz)" --data-binary @wellcomeml-$VERSION.tar.gz --url "https://uploads.github.com/repos/wellcometrust/WellcomeML/releases/$RELEASE_ID/assets?name=wellcomeml-$VERSION.tar.gz"
curl --request POST --silent --header "Authorization: token $GITHUB_TOKEN" -H "Content-Type: $(file -b --mime-type wellcomeml-$VERSION.tar.gz)" --data-binary @wellcomeml-$VERSION.tar.gz --url "https://uploads.github.com/repos/wellcometrust/WellcomeML/releases/$RELEASE_ID/assets?name=wellcomeml-$VERSION-py3-none-any.whl"

echo "Draft release created"
echo "Please change the release description and upload artifacts at https://github.com/wellcometrust/WellcomeML/releases/tag/v$VERSION"

