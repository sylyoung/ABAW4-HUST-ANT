import numpy as np
import pandas as pd


def get_predefined_vars():
    predefined_dict = {
        'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26'],
        'EXPR': ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other'],
        'VA': ['valence', 'arousal'],
        'Hidden_AUs': ['AU5', 'AU9', 'AU14', 'AU16', 'AU17']}
    train_annotation_file = 'ABAW5-MTL/training_{}_annotation.csv'
    AU_annot_df = pd.read_csv(train_annotation_file.format('AU'), usecols=predefined_dict['AU'])
    EXPR_annot_df = pd.read_csv(train_annotation_file.format('EXPR'))
    pos_sample_weight = AU_annot_df.apply(lambda x: np.sum(x == 0) / np.sum(x == 1))
    predefined_dict['AU_weight'] = pos_sample_weight
    sample_weight = [np.sum(EXPR_annot_df['expression'] == 7) / np.sum(EXPR_annot_df['expression'] == i) for i in
                     range(8)]
    predefined_dict['EXPR_weight'] = sample_weight
    return predefined_dict
