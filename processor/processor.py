import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda:0"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("FSRL.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "adv_loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "xld_loss": AverageMeter(),
        "hc_loss": AverageMeter(),
        #"sdmcross_loss": AverageMeter(),
        "itm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mlm2_loss": AverageMeter(),
        "mlmpre_loss": AverageMeter(),
        "mlmerror_loss": AverageMeter(),
        #"help_loss":AverageMeter(),
        #"mco_loss":AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()#,
        #"help_acc":AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    store1 = []################
    store2 = []################
    idstore = []###############
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()      
        store1_tem = []################
        store2_tem = []################
        idstore_tem = []###############
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            #ret = model(batch,epoch,store1,store2,idstore)
            ret,ifeats,tfeats = model(batch,epoch,store1,store2,idstore)#########################################
            store1_tem.append(ifeats)################
            store2_tem.append(tfeats)################
            idstore_tem.append(batch['pids'])###############

            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['adv_loss'].update(ret.get('adv_loss', 0), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['xld_loss'].update(ret.get('xld_loss', 0), batch_size)
            meters['hc_loss'].update(ret.get('hc_loss', 0), batch_size)
            meters['itm_loss'].update(ret.get('itm_loss', 0), batch_size)
            #meters['sdmcross_loss'].update(ret.get('sdmcross_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['mlm2_loss'].update(ret.get('mlm2_loss', 0), batch_size)
            meters['mlmpre_loss'].update(ret.get('mlmpre_loss', 0), batch_size)
            meters['mlmerror_loss'].update(ret.get('mlmerror_loss', 0), batch_size)
            #meters['help_loss'].update(ret.get('help_loss', 0), batch_size)
            #meters['mco_loss'].update(ret.get('mco_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            #meters['help_acc'].update(ret.get('help_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        store1 = store1_tem[:]################
        store2 = store2_tem[:]################
        idstore = idstore_tem[:]###############
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("FSRL.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
