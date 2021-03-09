import json
import logging
import os
from datetime import datetime
from os import mkdir
from os.path import isdir, join

from SourceCodeTools.code.data.sourcetrail.Dataset import read_or_create_dataset
from SourceCodeTools.models.graph import RGCNSampling, RGAN, RGGAN
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from params import rgcnsampling_params, rggan_params


def main(models, args):
    for model, param_grid in models.items():
        for params in param_grid:

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = read_or_create_dataset(args=args, model_base=model_base)

            if args.training_mode == "multitask":

                if args.intermediate_supervision:
                    # params['use_self_loop'] = True  # ????
                    from SourceCodeTools.models.graph.train.sampling_multitask_intermediate_supervision import evaluation_procedure
                else:
                    from SourceCodeTools.models.graph.train.sampling_multitask2 import evaluation_procedure

                evaluation_procedure(dataset, model, params, args, model_base)

            else:
                raise ValueError("Issue! ", args.training_mode)


if __name__ == "__main__":

    import argparse
    from main_sampling import add_train_args

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_train_args(parser)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    model_out = args.model_output_dir

    saved_args = json.loads(open(os.path.join(model_out, "metadata.json")).read())

    models_ = {
        eval(saved_args.pop("name").split("-")[0]): [saved_args.pop("parameters")]
    }

    args.__dict__.update(saved_args)
    args.restore_state = True
    args.model_output_dir = model_out

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)
