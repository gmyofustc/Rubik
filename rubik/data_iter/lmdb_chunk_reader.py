from rubik.data_iter import ChunkReader


class LMDB_ChunkReader(ChunkReader):
    def __init__(self, lmdb_file_list, lmdb_key_list, global_chunk_size=2000, local_chunk_size=50,
                worker_nums=4, shuffle=True):
        self.lmdb_file_list = lmdb_file_list
        self.lmdb_key_list = lmdb_key_list
        self.global_chunk_size = global_chunk_size
        self.local_chunk_size = local_chunk_size
        self.worker_nums = worker_nums
        self.shuffle = shuffle
        self.chunk_queue_cache_num = global_chunk_size//local_chunk_size
        self._reset()
        self._init_lmdb_utils()

    def _reset(self):
        self.start_key_queue = mp.Queue(max_size=len(self.start_key_list))
        for start_key in self.start_key_list:
            self.start_key_queue.put(start_key)
        self.chunk_queue = mp.Queue(maxsize=self.chunk_queue_cache_num)
        self.workers = [mp.Queue(target=self._worker_parser,
                                 args=(self.start_key_queue,
                                       self.chunk_queue,
                                       self.start_key2lmdb_idx,
                                       self.lmdb_file_list,
                                       self.local_chunk_size,
                                       i)
                                )
                        for i in range(self.worker_nums)
                        ]
        for w in self.workers:
            d.damon = True
            w.start()

    def _init_lmdb_utils(self):
        self.start_key_list = []
        self.start_key2lmdb_idx = {}
        for lmdb_idx, lmdb_key_file in enumerate(self.lmdb_key_list):
            with open(lmdb_key_file) as key_handler:
                curr_lmdb_key_lines = key_handler.readlines()
                chunk_num = (len(curr_lmdb_key_lines)+self.local_chunk_size-1)//self.local_chunk_size
                for chunk_idx in range(chunk_num):
                    start_key = curr_lmdb_key_lines[chunk_idx*self.local_chunk_size].strip()
                    self.start_key_list.append(start_key)
                    self.start_key2lmdb_idx[start_key] = lmdb_idx
        if self.shuffle:
            random.shuffle(self.start_key_list)

    def _worker_parser(self, start_key_queue, chunk_queue, start_key2lmdb_idx,
                        lmdb_file_list, local_chunk_size, worker_id):
        while 1:
            start_key = start_key_queue.get()
            if start_key is not None:
                out_list = []
                lmdb_file = lmdb_file_list[start_key2lmdb_idx[start_key]]
                lmdb_env = lmdb.open(lmdb_file, readonly=True, lock=False)
                lmdb_txn = lmdb_env.begin()
                lmdb_cursor = lmdb_txn.cursor()
                lmdb_cursor.set_key(start_key.encode())
                for i in range(local_chunk_size):
                    out_dict = self._parse_func(lmdb_cursor)
                    if out_dict is not None:
                        out_list.append(out_dict)
                    not_finished = lmdb_cursor.next()
                    if not not_finished:
                        break
                lmdb_env.close()
                del lmdb_env
                del lmdb_txn
                del lmdb_cursor
                chunk_queue.put(out_list)
            else:
                logging.info('DataReader worker{} has finished'.format(worker_id))
    
    def _parse_func(self, lmdb_curosr):
        datum = Datum()
        datum.ParseFromstring(lmdb_curosr.value())
        mel_spec = np.fromstring(datum.mel_spec, dtype=np.float32).reshape(-1, 80)
        state_target = np.fromstring(datum.state_lab, dtype=np.int32)
        out_dict = {}
        out_dict['mel_spec'] = mel_spec
        out_dict['state_lab'] = state_target
        #return None if this item does not Need
        return out_dict


    def _get_train_chunk(self):
        chunk_out_list = []
        for i in range(chnk_queue_cache_num):
            out_buffer = self._get_buffer_from_queue()
            if out_buffer is None:
                pass
            else:
                chunk_out_list.append(out_buffer)
        chunk_out = list(sum(chunk_out_list, []))
        if len(chunk_out)==0:
            self.epoch_idx += 1
            for w in self.workers:
                w.terminate()
            raise StopIteration
        return chunk_out
        
    def _get_val_chunk(self):
        pass
    

