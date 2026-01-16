
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import pytorch_lightning as pl
import os
from utils import accuracy, MetricLogger
import math



class TrainerModelPretraining(pl.LightningModule):
    
    def __init__(self, model, log,opt,checkpoint):
        super().__init__()
        self.model = model
        self.log = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = False
        self.start_time  = None
        os.makedirs(opt["save_path"],exist_ok=True)
        self.loss_ce = nn.CrossEntropyLoss()
        
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        return len(self.trainer._data_connector._train_dataloader_source.dataloader())
    
    @property
    def num_batch_size(self) -> int:
        return self.trainer._data_connector._train_dataloader_source.dataloader().batch_size
    
    
    def get_loss_acc(self, pred, gt, smoothing=False):
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss_cls = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss_cls = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss_cls, acc * 100

    def adjust_learning_rate(self,idx):
        """Decays the learning rate with half-cycle cosine after warmup"""
        warmup_epochs = self.opt["warmup_epoch"]
        lr = self.opt["learning_rate"]
        minlr = self.opt.get("min_lr", 0.0)
        epochs = self.opt.get("cycle_epoch", self.opt["epoch"])
        epoch = self.current_epoch % epochs + idx / self.num_training_steps
        
        if epoch < warmup_epochs:
            lr = minlr + (lr - minlr) * epoch / warmup_epochs
        else:
            lr = minlr + (lr - minlr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        for param_group in self.optimizers().param_groups:
            param_group['lr'] = lr
    

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output 
    
    
    def training_step(self,data,idx):
        
        event = data["event"]
        label = data["label"]
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        self.adjust_learning_rate(idx)
        
        output = self.model(event)
        
        loss_event, _ = self.get_loss_acc(output, label)
        
        self.manual_backward(loss_event)
        
        self.optimizers().step()
        self.optimizers().zero_grad()
        
        self.produce_log(loss_event, output, label, idx)


    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if "max_grad_norm" in self.opt:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt["max_grad_norm"]) 


    def training_epoch_end(self, training_step_outputs):
        if self.trainer.is_global_zero and (self.current_epoch + 1) % self.opt["save_every"] == 0:
            self.save()

    def produce_log(self,loss_event, pred, label,idx):
        
        loss_event = self.all_gather(loss_event).mean().item()
        pred_labels = self.all_gather(pred).view(-1,1000)
        labels = self.all_gather(label).view(-1)
        acc_1, acc_5 = accuracy(pred_labels.cpu(), labels.cpu(), topk=(1, 5))
        
        if self.trainer.is_global_zero and idx % 100 == 0:
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
            lr = self.optimizers().param_groups[0]['lr']
            self.log(
                {"current_epoch": self.current_epoch,
                 "max_epochs": self.trainer.max_epochs,  
                 "idx": idx,
                 "len_loader":len_loader,
                 "time_left": time_left,
                 "loss_event": loss_event, 
                  "acc_1": acc_1,
                  "acc_5": acc_5,
                  "lr": lr,
                    }
                )
            
            self.log.save_train(
                self.current_epoch,
                idx,
                {"loss_event": loss_event, 
                  "acc_1": acc_1,
                  "acc_5": acc_5,
                  "lr": lr,
                    }
                )
        del loss_event, pred_labels, labels, acc_1, acc_5

        
    def on_validation_epoch_start(self):
        self.metric_logger = MetricLogger()
        self.log.raw("Initialized metric_logger")
            
    def validation_step(self, data, idx):
        event = data["event"]
        label = data["label"]

        out = self.model(event)

        loss_event, _ = self.get_loss_acc(out, label)
        
        acc_1, acc_5 = accuracy(out.cpu(), label.cpu(), topk=(1, 5))
        batch_size = event.size(0)
        self.metric_logger.update(loss=loss_event.item())
        self.metric_logger.meters['acc_1'].update(acc_1, n=batch_size)
        self.metric_logger.meters['acc_5'].update(acc_5, n=batch_size)

        
    def validation_epoch_end(self,outputs):
        self.metric_logger.synchronize_between_processes()
        loss_event = self.metric_logger.loss.global_avg
        acc_1, acc_5 = self.metric_logger.acc_1.global_avg, self.metric_logger.acc_5.global_avg
        
        if self.trainer.is_global_zero and self.trainer.num_gpus!=0: 
            self.log.raw(
                    "[loss_event :%f, acc_1:%f, acc_5:%f]" %
                    (
                        loss_event,
                        acc_1,
                        acc_5,
                     )         
                         ) 
            self.log.save_eval(
                self.current_epoch,
                {"loss_event": loss_event,
                  "acc_1": acc_1,
                  "acc_5" : acc_5
                    }
                )

    def save(self):
        output_path = os.path.join(self.opt["save_path"], f"{self.current_epoch + 1}.pt")
        torch.save(
            {
             "checkpoint": self.model.state_dict(),
             "optimizer": self.optimizers().state_dict(),
             }
            , output_path)
        self.log.raw("Model saved")

    def configure_optimizers(self):
        self.opt["learning_rate"] = self.trainer.num_gpus * self.trainer.num_nodes * self.num_batch_size / 256 * self.opt["base_lr"]
        self.log.raw(f"learning_rate: {self.opt['learning_rate']}")
        b1 = self.opt.get("b1",0.9)
        b2=self.opt.get("b2",0.999)
        self.log.raw(f" betas {b1} {b2}")
        
        optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr = self.opt["learning_rate"],
                betas = (b1, b2),
                weight_decay = self.opt["weight_decay"],
                )

        if self.checkpoint != None:
            pass
        
        return optimizer                     
                                
        