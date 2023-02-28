import datasets
import pandas as pd


logger = datasets.logging.get_logger(__name__)


_CITATION = """None"""
_DESCRIPTION = """None"""

class GeneralConfig(datasets.BuilderConfig):
    """BuilderConfig."""
    def __init__(self, **kwargs):
        super(GeneralConfig, self).__init__(**kwargs)


class GeneralConfig(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        GeneralConfig(
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

            supervised_keys=None,
            homepage="None",
            citation=_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        DATASET = self.config.data_dir.split("|")[0]
        RUN = self.config.data_dir.split("|")[1]
            
        train_path = f"../dataset/{DATASET}/run_{RUN}/train.csv"
        valid_path = f"../dataset/{DATASET}/run_{RUN}/test.csv"
        test_path = f"../dataset/{DATASET}/run_{RUN}/test.csv"

        # self._shuffle_data()
        
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
        
        df = pd.read_csv(datapath)
        
        for idx, row in df.iterrows():
    
                guid = f"{idx}|{row.samp_id}"
                
                yield key, {
                    "id": guid,
                    "full_text": row.text,
                    "text": row.ae,
                    "label": row.term,
                    }
                key = key + 1
