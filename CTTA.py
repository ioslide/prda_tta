import argparse
import warnings
import methods
from conf import cfg
from core.model import build_model
from loguru import logger as log
from setproctitle import setproctitle
from core.utils import seed_everything, save_df, set_logger
from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_cifar100c, load_imagenet3dcc, load_imagenetc

warnings.filterwarnings("ignore")


def eval(cfg):
    model = build_model(cfg).cuda()
    tta_model = getattr(methods, cfg.ADAPTER.NAME).setup(model, cfg)
    tta_model.cuda()
    dataset_loaders = {
        "cifar100": load_cifar100c,
        "imagenet_3dcc": load_imagenet3dcc,
        "imagenet": load_imagenetc,
    }
    load_image = dataset_loaders.get(cfg.CORRUPTION.DATASET)

    for severity in cfg.CORRUPTION.SEVERITY:
        new_results = {
            "method": cfg.ADAPTER.NAME,
            "dataset": cfg.CORRUPTION.DATASET,
            "model": cfg.MODEL.ARCH,
            "batch_size": cfg.TEST.BATCH_SIZE,
            "seed": cfg.SEED,
            "severity": severity,
            "note": cfg.NOTE,
            "order": cfg.CORRUPTION.ORDER_NUM,
            "Avg": 0,
        }
        log.info(f"==>> Start Eval {new_results}")
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            x_test, y_test = load_image(
                cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type]
            )
            acc = accuracy(
                model=tta_model,
                x=x_test.cuda(),
                y=y_test.cuda(),
                batch_size=cfg.TEST.BATCH_SIZE,
                is_enable_progress_bar=cfg.ENABLE_PROGRESS_BAR,
            )
            err = 1.0 - acc
            new_results[f"{corruption_type}"] = acc * 100
            new_results["Avg"] += acc * 100
            log.info(f"[{corruption_type}{severity}]: Acc {acc:.2%} || Error {err:.2%}")

        new_results["Avg"] = new_results["Avg"] / len(cfg.CORRUPTION.TYPE)
        log.info(
            f"[Avg {severity}]: Acc {new_results['Avg']:.2f} || Error {100-new_results['Avg']:.2f}"
        )
        save_df(
            new_results,
            f"./results/{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv",
        )


def main():
    parser = argparse.ArgumentParser(
        "Pytorch Implementation for Continual Test Time Adaptation!"
    )
    parser.add_argument(
        "-acfg",
        "--adapter-config-file",
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str,
    )
    parser.add_argument(
        "-dcfg",
        "--dataset-config-file",
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str,
    )
    parser.add_argument(
        "-ocfg",
        "--order-config-file",
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify the configuration by command line",
        nargs=argparse.REMAINDER,
        default=None,
    )

    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR, cfg.ADAPTER.NAME)
    setproctitle(f"CTTA:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}")

    log.info(
        f"Loaded configuration file: \n"
        f"\tadapter: {args.adapter_config_file}\n"
        f"\tdataset: {args.dataset_config_file}\n"
        f"\torder: {args.order_config_file}"
    )

    try:
        eval(cfg)
    except Exception as e:
        log.info(
            f"Error in TTA {e} \n {cfg.SEED} TTA: {cfg.ADAPTER.NAME} DATASET: {cfg.CORRUPTION.DATASET} BS: {cfg.TEST.BATCH_SIZE} MODEL: {cfg.MODEL.ARCH}  ORDER: {cfg.CORRUPTION.ORDER_NUM} SEVERITY: {cfg.CORRUPTION.SEVERITY} \n {cfg.NOTE}"
        )
        raise


if __name__ == "__main__":
    main()
