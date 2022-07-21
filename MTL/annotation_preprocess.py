import pandas as pd
import os


def extract_single_task(annotation_file:str, dataset:str):
    df = pd.read_csv(annotation_file.format(dataset))
    dir_path='ABAW5-MTL/'
    df_va = df.loc[:, ['image_path', 'valence', 'arousal']]
    df_va = df_va[df.apply(lambda x: x['valence']!=-5 and x['arousal']!=-5, axis=1)]
    df_va.to_csv(os.path.join(dir_path, '{}_VA_annotation.csv'.format(dataset)), index=False)
    df_expr = df.loc[:, ['image_path', 'expression']]
    df_expr = df_expr[df.expression.map(lambda x: x != -1)]
    df_expr.to_csv(os.path.join(dir_path, '{}_EXPR_annotation.csv'.format(dataset)), index=False)
    df_au = df.loc[:, ['image_path', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23',
                   'AU24', 'AU25', 'AU26']]
    df_au = df_au[df.apply(lambda x: -1 not in x, axis=1, raw=True)]
    df_au.to_csv(os.path.join(dir_path, '{}_AU_annotation.csv'.format(dataset)), index=False)


if __name__ == '__main__':
    extract_single_task('ABAW5-MTL/{}_set_annotations.txt', dataset='training')
    extract_single_task('ABAW5-MTL/{}_set_annotations.txt', dataset='validation')
