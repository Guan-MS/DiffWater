import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config.yml',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    args = parser.parse_args()

    opt = Logger.dict_to_nonedict(Logger.parse(args))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    wandb_logger = WandbLogger(opt) if opt['enable_wandb'] else None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        restore_img = Metrics.tensor2img(visuals['output'][-1])  # uint8
        target_img = Metrics.tensor2img(visuals['target'])  # uint8
        input_img = Metrics.tensor2img(visuals['input'])  # uint8

        Metrics.save_img(input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
        Metrics.save_img(restore_img, '{}/{}_{}_output.png'.format(result_path, current_step, idx))
        output = Metrics.tensor2img(visuals['output'])
        Metrics.save_img(output, '{}/{}_{}_denoising_process.png'.format(result_path, current_step, idx))
        Metrics.save_img(target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(restore_img, Metrics.tensor2img(visuals['input'][-1]), target_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)


if __name__ == "__main__":
    main()
