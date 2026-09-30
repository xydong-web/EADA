import json
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.utils import comm
from eada.config import add_eada_config
import eada.data.builtin
import eada.model
from eada.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from eada.evaluation.coco_evaluation import COCOEvaluator


class _EADATwoStreamLoader:
    """Concatenate one support batch and one base-data batch per iteration."""

    def __init__(self, support_loader, base_loader):
        self.support_loader = support_loader
        self.base_loader = base_loader

    def __iter__(self):
        support = iter(self.support_loader)
        base = iter(self.base_loader)
        while True:
            try:
                support_batch = next(support)
            except StopIteration:
                support = iter(self.support_loader)
                support_batch = next(support)
            try:
                base_batch = next(base)
            except StopIteration:
                base = iter(self.base_loader)
                base_batch = next(base)
            yield list(support_batch) + list(base_batch)

    def __len__(self):
        return len(self.support_loader)


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        support_loader = build_detection_train_loader(cfg)
        if not cfg.DATASETS.EADA_TWO_STREAM:
            return support_loader
        if not cfg.DATASETS.BASE_TRAIN:
            raise ValueError("EADA two-stream adaptation requires DATASETS.BASE_TRAIN")
        base_cfg = cfg.clone()
        base_cfg.defrost()
        base_cfg.DATASETS.TRAIN = tuple(cfg.DATASETS.BASE_TRAIN)
        base_cfg.DATASETS.EADA_TWO_STREAM = False
        base_cfg.freeze()
        base_loader = build_detection_train_loader(base_cfg)
        return _EADATwoStreamLoader(support_loader, base_loader)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
        kind = MetadataCatalog.get(dataset_name).evaluator_type
        if kind == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if kind == "coco":
            return COCOEvaluator(dataset_name, True, output_folder)
        raise ValueError(kind)


def main(args):
    cfg = get_cfg()
    add_eada_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if args.eval_only:
        if not cfg.MODEL.WEIGHTS:
            raise ValueError("Evaluation requires MODEL.WEIGHTS")
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        results = Trainer.test(cfg, model)
        if comm.is_main_process():
            with open(os.path.join(cfg.OUTPUT_DIR, "evaluation_results.json"), "w") as handle:
                json.dump(results, handle, indent=2)
        return results
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    arguments = default_argument_parser().parse_args()
    launch(main, arguments.num_gpus, num_machines=arguments.num_machines, machine_rank=arguments.machine_rank, dist_url=arguments.dist_url, args=(arguments,))
