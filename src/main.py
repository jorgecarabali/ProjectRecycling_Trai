###src/main.py

import argparse
from pipeline import ClassificationPipeline


def main():
    """Main entry point for the command line interference; parses arguments and runs the selected pipeline process"""

    parser=argparse.ArgumentParser(
        description="Run different processes: Preprocess, train, inference"
    )

    parser.add_argument(
        "process",
        type=str.lower,
        choices=["preprocess", "train","inference"],
        help="The process to run by selecting one of the following: preprocess, train,inference",
    )
    args=parser.parse_args()

    pipeline = ClassificationPipeline()
    if args.process == "preprocess":
        pipeline.preprocess_images()
    elif args.process == "train":
        pipeline.train_model()
    elif args.process == "inference":
        print(pipeline.inference())  
 


if __name__ == "__main__":
    main()    