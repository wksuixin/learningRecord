import torch

'''
DataLoader和Dataset(IterableDataset)处理数据
映射型数据集：Dataset   必须实现__getitem__() 一般还会实现__len__()
迭代型数据集：IterableDataset  必须实现__iter__()
'''

'''
多线程读取时，加载迭代型数据时必须进行专门的设置，否则会读取重复样本
如果不进行设置，每个dataloader都会获得完整的数据集拷贝
因此需要在Dataloader中设置worker_init_fn来定义每一个进程的数据集拷贝
'''
import math
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info

class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=1, end=10)

print(list(DataLoader(ds, num_workers=0)))
print(list(DataLoader(ds, num_workers=1)))


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(overall_end, dataset.start + per_worker)

print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
print(list(DataLoader(ds, num_workers=8, worker_init_fn=worker_init_fn)))

'''
映射型数据集
'''
from torch.utils.data import Dataset
class CustomImageDateset(Dataset):
    def __init__(self):
        self.data = list(range(1, 10))
        self.label = [x+1000 for x in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

ds = CustomImageDateset()    
print(list(DataLoader(ds)))


'''
DataLoaders提供以下参数
batch_size:
shuffle: 是否打乱数据集
sampler: 采样器，索引上的迭代器
collate_fn: 批处理函数，对采样出的batch进行处理
'''
