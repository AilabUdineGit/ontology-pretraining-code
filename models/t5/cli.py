def setup_parser(parser):
    parser.add_argument(
        "--location",
        type=str,
        default="default",
        help="Machine on which the code is run. Determines save path variables and deepspeed config.",
    )
    parser.add_argument("--dataset_test", type=str, default=None, help="")
    parser.add_argument("--model", type=str, help="Name of the model to finetune/evaluate.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to evaluate on."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. Defaults to 3.",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="Multi task training",
    )
    parser.add_argument(
        "--train_save_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--train_load_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--test_load_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--results_save_path",
        type=str,
        default=None,
        help="",
    )
    return parser
