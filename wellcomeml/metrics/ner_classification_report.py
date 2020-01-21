
from nervaluate import Evaluator


def ner_classification_report(y_true, y_pred, groups, tags):
    """
    Evaluate the model's performance for each grouping of data
    for the NER labels given in 'tags'

    Input:
        y_pred: a list of predicted entities
        y_true: a list of gold entities
        groups: (str) the group each of the pred or gold entities belong to

    Output:
        report: evaluation metrics for each group
                in a nice format for printing
    """

    unique_groups = sorted(set(groups))
    outputs = []

    for group in unique_groups:
        pred_doc_entities = [y_pred[i] for i, g in enumerate(groups) if g == group]
        true_doc_entities = [y_true[i] for i, g in enumerate(groups) if g == group]

        evaluator = Evaluator(
            true_doc_entities,
            pred_doc_entities,
            tags=tags
            )
        results, _ = evaluator.evaluate()

        output_dict = {
            'precision (partial)': results['partial']['precision'],
            'recall (partial)': results['partial']['recall'],
            'f1-score': results['partial']['f1'],
            'support': len(pred_doc_entities)
            }
        output = [group]
        output.extend(list(output_dict.values()))
        outputs.append(output)

    headers = output_dict.keys()

    width = max(len(cn) for cn in unique_groups)
    head_fmt = '{:>{width}s} ' + ' {:>17}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>17.{digits}f}' * 3 + ' {:>17}\n'

    for row in outputs:
        report += row_fmt.format(*row, width=width, digits=3)

    report += '\n'

    return report
