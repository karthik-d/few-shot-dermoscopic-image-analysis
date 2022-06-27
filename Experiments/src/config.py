from utils.DotDict import DotDict

import os

config = DotDict()

config.update(
    DotDict(
        root_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            *(os.path.pardir,)*1
        ) 
    )
)

config.update(
    DotDict(
        logs_root_path = os.path.join(
            config.root_path, 
            'logs'
        ) 
    )
)