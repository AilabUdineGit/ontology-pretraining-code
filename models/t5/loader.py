"""IRMS Dataset"""

import json
import os
import datasets
import sys
import pandas as pd


logger = datasets.logging.get_logger(__name__)


_CITATION = """None"""
_DESCRIPTION = """None"""
_URL = "None"
_URLS = "None"


class ADEConfig(datasets.BuilderConfig):
    """BuilderConfig."""
    def __init__(self, **kwargs):
        super(ADEConfig, self).__init__(**kwargs)


class ADEDataset(datasets.GeneratorBasedBuilder):
    """IRMS Dataset"""

    BUILDER_CONFIGS = [
        ADEConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "full_text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="None",
            citation=_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        num = ""#input("Choose IRMS dataset to load (1-10): ")
        if self.config.data_dir.split("/")[-1].isnumeric():
            self.config.data_dir = self.config.data_dir[:self.config.data_dir.rfind("/")]
            
        train_path = self.config.data_dir + '/train.pickle'
        valid_path = self.config.data_dir + '/valid.pickle'
        test_path = self.config.data_dir + '/test.pickle'

        self._shuffle_data()
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": train_path, "datatype": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"datapath": valid_path, "datatype": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"datapath": test_path, "datatype": "test"},
            ),
        ]

    def _generate_examples(self, datapath, datatype):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", datapath)
        key = 0
        
        df = pd.read_pickle(datapath).fillna("")
        
        for idx, row in df.iterrows():
    
                guid = "%s-%s-%s" % (str(key), datatype, row.index)
                text_a = row.ae #line[1]
                label = row.term #line[2]
                
                yield key, {
                    "id": guid,
                    "full_text": row.text,
                    "text": row.ae,
                    "label": row.term,
                    }
                key = key + 1

        
    def _shuffle_data(self):
        
        train_path = self.config.data_dir + '/train.pickle'
        valid_path = self.config.data_dir + '/valid.pickle'
        test_path = self.config.data_dir + '/test.pickle'
        
        
        datapath = self.config.data_dir + '/ae_to_term_all_levels.pickle'
        df = pd.read_pickle(datapath).fillna("")

        random_ids = pd.Series(df.samp_id.unique()).sample(frac=1).tolist()
        train_perc = 0.8

        train_ids = random_ids[:int(len(random_ids)*0.8)]
        test_ids = random_ids[int(len(random_ids)*0.8):]

        df.index = df.samp_id

        df_train = df.loc[train_ids].copy().reset_index(drop=True)
        df_test = df.loc[test_ids].copy().reset_index(drop=True)

        print("train data:", len(df_train))
        print("test data:", len(df_test))

        df_train.to_pickle(train_path)
        df_test.to_pickle(valid_path)
        df_test.to_pickle(test_path)
