import os
import torch
"""
Utility functions to load and save torch model checkpoints 
"""
def load_checkpoint(net, optimizer=None, step='max', save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    checkpoints = [x for x in os.listdir(save_dir) if not x.startswith('events')]
    if step == 'max':
        step = 0
        if checkpoints:
            step, last_checkpoint = max([(int(x.split('.')[0]), x) for x in checkpoints])
    else:
        last_checkpoint = str(step) + '.pth'
    if step:
        save_path = os.path.join(save_dir, last_checkpoint)
        state = torch.load(save_path, map_location='cpu')
        net.load_state_dict(state['net'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Loaded checkpoint %s' % save_path)
    return step

def save_checkpoint(net, optimizer, step, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(step) + '.pth')

    torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict()), save_path)
    print('Saved checkpoint %s' % save_path)
