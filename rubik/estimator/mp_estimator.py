import torch 
from rubik.commom import nccl
from rubik.estimator.mp_wapper import MPWapper, Future
from rubik.summary import RubikSummary
import horovod.torch as hvd
from horovod.torch.mpi_ops import allreduce_async_, all_gather

class MPEstimator(MPWapper):
    def __init__(self, hparams, model, devcies_ids=None, multiprocessing_method='fork'):
        super().__init__(device_ids, multiprocessing_method)

        model = model.share_memory()
        nccl_uid = nccl.get_unique_id()
        self.init(hparams, model, nccl_uid)
        
    #init 
    def init(self, hparams, model, nccl_uid):
        self.logger = RubikSummary(hparams)
        Future.gen_list([self.call_async(rank, '_async_init', hparams=hparams, 
                            model=model, nccl_uid=nccl_uid)
                        for rank in range(self.num_replicas)])
    def _async_init(self, rank, device_id, hparams, model, nccl_uid):
        torch.manual_seed(hparams.seed)
        torch.cuda.set_device(device_id)
        nccl.initialize(self.num_replicas, nccl_uid, device_id)
        self.model = model.cuda()
        self.train_states = None #this should contain the params and batchnorm states
        self.optimizer = MixedAdam(self.model.parameters())
        self.lr_scheduler = WarmupLRSheduler(self.optimizer)
        self.flat_grads = None
        self.flat_param_buffer = None

    #Load Checkpoint
    def load_checkpoint(self, filename):
        results = Future.gen_list([self.call_async(rank, '_async_load_checkpoint',
                                        filename=filename)
                                  for rank in range(self.num_replicas)])
        epoch, batch_offset, metric_info = results[0]
    def _async_load_checkpoint(self, rank, device_id, filename):
        return checkpoint.load_checkpoint(filename, self.model, self.optimizer,
                                        self.lr_scheduler, cuda_device=device_id)

    #Save Checkpoint
    def save_checkpoint(self, hparams, epoch, batch_offset, metric_info=None, filename=None):
        self.call_async(0, '_async_save_checkpoint', hparams=hparams, epoch=epoch,
                        batch_offset=batch_offset, metric_info=None,
                        filename=None).gen()
    def _async_save_checkpoint(self, hparams, epoch, batch_offset, metric_info, filename):
        checkpoint.save_checkpoint(hparams, epoch, batch_offset, self.model, 
                                    self.optimizer, self.lr_scheduler, metric_info,
                                    filename)

    def train(self, hparams, trainset, devset):
        batch_list = []
        while self.epoch_idx<hparams.max_epoch:
            for bid, batch_data in enumerate(trainset):
                batch_list.append(batch_data)
                if len(batch_list)==self.num_replicas!=0:
                    continue
                else:
                    self.batch_shift += 1
                    self.scatter_batch_data(batch_list)
                    batch_list = []
                forward_out_list = [
                    self.call_asyn(rank, '_async_train_step', train_step=batch_shift,
                    for rank in range(self.num_replicas))
                ]
                metrics, local_reduce_tensors = Future.gen_tuple_list(forward_out_list)
                self.dist_reduce(metrics, local_reduce_tensors)
                Fture.gen_list([self.call_async(rank, '_async_optim_step', train_step=batch_shift)
                                for rank in range(self.num_replicas)
                                ])
                if batch_shift%hparams.checkpoint_freq==0:
                    self.validate(hparasm, devset)
            #The rest batch_list data is left for next epoch
            self.epoch_idx += 1
    def _async_train_step(self, rank, device_id, train_step):
        self.model.train()
        scalar_metric = {}
        forward_out = self.model(self.batch_data)
        loss = forward_out['loss']
        loss.backward()
        for k,v in forward_out.items():
            nccl.all_reduce(v)
            scalar_metric[k] = (v.dev_(self.num_replicas)).item()
        local_allreduce_tensor = self.local_node_allreudce(train_step)
        return local_allreduce_tensor, scalar_metric
    def _async_optim_step(self, rank, device_ids, train_step):
        if self.hparasm.sync_sgd:
            if train_step%self.hparams.allreduce_freq==0:
                nccl.broadcast(self.flat_grads, 0)
                self.optimizer.step()
                self.lr_schedulr.step()
                self.optimizer.zero_grad()
        else:
            if train_step%self.hparams.allreduce_freq==0:
                nccl.boradcast(self.flat_param_buffer, 0)
                self.optimizer.bmuf_step()
    

    def local_node_allreduce(self, train_step):
        #sync
        if self.hparams.sync_sgd:
            if self.flat_grads is None:
                self.flat_grads = self.flatten_grads(self.train_states)
            if train_step%self.hparasm.allreduce_freq==0:
                nccl.all_reduce(self.flat_grads)
                self.flat_grads.dev_(self.num_replicas)
                grad_norm = self._clip_grads(self.flat_grads, 
                                             self.hparams.clip_norm)
            return self.flat_grads
        #async BMUF
        else:
            if self.flat_param_buffer is None:
                self.flat_param_buffer = self.flatten_params(self.train_states)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if train_step%self.hparasm.allreduce_freq==0:
                nccl.all_reduce(self.flat_param_buffer)
                self.flat_param_buffer.div_(self.num_replicas)
            return flat_param_buffer

    def dist_node_allreduce(self, metric, local_allreuce_tensor, batch_shift):
        reduce_chunk = 409600 #one reduce chunk has 409600*4bytes(16MB)
        allreduce_chunk_num = ((local_allreduce_tensor[0]).size(0)
                                +reduce_chunk-1)//reduce_chunk
        hvd_handles = []
        if hvd.size()>1 and batch_shift%self.hparams.allreduce_freq==0:
            self.allgather_metric(metric)
            for i in range(allreduce_chunk_num):
                hvd_handles.append(allreduce_async_(
                    local_allreduce_tensor[0][i*allreduce_chunk_num:
                                             (i+1)*allreduce_chunk_num], 
                                    average=True))
            for handle in hvd_handles:
                synchronize(handle)




        
