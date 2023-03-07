from constants import model_related_const

# ---------------------------------------------------


def setup_parser(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=model_related_const.keys(),
        help="Name of the model to finetune/evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        # choices=dataset_related_const.keys(),
        help="Name of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Dataset split to use (if more are available).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Defaults to 20.",
    )
    return parser
