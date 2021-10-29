"""
Vectorizer that exposes sklearn interface to sent2vec
paper and codebase. https://github.com/epfml/sent2vec
"""
from wellcomeml.utils import check_cache_and_download, throw_extra_import_message

required_modules = 'sklearn'
required_extras = 'core'

try:
    from sklearn.base import TransformerMixin, BaseEstimator
except ImportError as e:
    throw_extra_import_message(error=e, required_modules=required_modules,
                               required_extras=required_extras)


class Sent2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, pretrained=None):
        self.pretrained = pretrained

    def fit(self, *_):

        try:
            import sent2vec
        except ImportError:
            from wellcomeml.__main__ import download

            download("non_pypi_packages")
            import sent2vec

        if self.pretrained:
            model_path = check_cache_and_download(self.pretrained)
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(model_path)
        else:
            # Custom training not yet implemented
            raise NotImplementedError(
                "Fit only implemented for loading pretrained models"
            )
        return self

    def transform(self, X):
        return self.model.embed_sentences(X)
