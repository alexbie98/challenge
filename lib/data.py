import torch

def transform(batch, f):
    inputs = f([x for x in batch['image']], return_tensors='pt')
    inputs['labels'] = batch['labels']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
