import torch
import models
import params
#

# Model Hyperparameters 
MODELS = {
    'MLP': {
        'name': 'MLP',
        'ref':models.MLP,
        'disc':'',
        'tag': '',
        'hyperparams':{
            'hidden dim':512,
            'tv only': False,
            'task': params.DUAL,
            'curriculum loss': False,
            'curriculum seq': False,
            'curriculum virtual':False,
        },
        'optimizer': torch.optim.Adam,
        'lc loss function': torch.nn.CrossEntropyLoss,
        'ttlc loss function': torch.nn.MSELoss,
        'data type': 'state',
        'state type': '',
    },
    'VLSTM': {
        'name': 'VLSTM',
        'ref':models.VanillaLSTM,
        'disc':'',
        'tag': '',
        'hyperparams':{
            'layer number': 1,
            'tv only': False,
            'hidden dim':512,
            'task': params.REGRESSION,
            'curriculum loss': False,
            'curriculum seq': False,
            'curriculum virtual':False,
        },
        'optimizer': torch.optim.Adam,
        'lc loss function': torch.nn.CrossEntropyLoss,
        'ttlc loss function': torch.nn.MSELoss,
        'data type': 'state',
        'state type': '',
    },
    'VGRU': {
        'name': 'VGRU',
        'ref':models.VanillaGRU,
        'disc':'',
        'tag': '',
        'hyperparams':{
            'layer number': 1,
            'tv only': False,
            'hidden dim':512,
            'task': params.REGRESSION,
            'curriculum loss': False,
            'curriculum seq': False,
            'curriculum virtual':False,
        },
        'optimizer': torch.optim.Adam,
        'lc loss function': torch.nn.CrossEntropyLoss,
        'ttlc loss function': torch.nn.MSELoss,
        'data type': 'state',
        'state type': '',
    },
    'VCNN':{
        'name': 'VCNN',
        'ref':models.VanillaCNN,
        'disc':'',
        'tag': '',
        'hyperparams':{
            'kernel size': 3,
            'channel number':16,
            'merge channels': True,
            'task': params.DUAL,
            'curriculum loss': False,
            'curriculum seq': False,
            'curriculum virtual':False,

        },
        'optimizer': torch.optim.Adam,
        'lc loss function': torch.nn.CrossEntropyLoss,
        'ttlc loss function': torch.nn.MSELoss,
        'data type': 'image',
        'state type': '',
    },
    'REGIONATTCNN3':{
        'name': 'REGIONATTCNN3',
        'ref':models.ATTCNN3,
        'disc':'attention weights for quad-regions',
        'tag': '',
        'hyperparams':{
            'kernel size': 3,
            'channel number':16,
            'merge channels': True,
            'task': params.DUAL,
            'curriculum loss': False,
            'curriculum seq': False,
            'curriculum virtual':False,
        },
        'optimizer': torch.optim.Adam,
        'lc loss function': torch.nn.CrossEntropyLoss,
        'ttlc loss function': torch.nn.MSELoss,
        'data type': 'image',
        'state type': '',
    },
}