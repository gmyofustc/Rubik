from rubik.data_iter import ChunkReader


class LMDB_ChunkReader(ChunkReader):
    def __init__(self, lmdb_file_list, lmdb_key_list, global_chunk_size=2000, local_chunk_size=50,
                worker_nums=4, shuffle=True):
        self.lmdb_file_list = lmdb_file_list
        self.lmdb_key_list = lmdb_key_list
        pass

    def _reset(self):
        pass

    def _init_lmdb_utils(self):
        pass

    def _worker_parser(self):
        pass

    def _parser_func(self):
        pass

    def _get_train_chunk(self):
        pass

    def _get_val_chunk(self):
        pass
    

