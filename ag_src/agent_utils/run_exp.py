# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : run_exp.py 
   Description :  
   Author :       HX
   date :    2024/3/22 9:38 
-------------------------------------------------
"""
from grail_src.GRAIL import *
from webqsp_src.WebQSP import *
from meta_src.META import *
from graphq_src.GRAPHQ import *
from wikisql_src.WIKISQL import *
# from wtq_src.WTQ import *

if __name__ == "__main__":
    if config['dataset'] == 'grailqa':
        grailqa_main()
    elif config['dataset'] == 'graphq':
        graphq_main()
    elif config['dataset'] == 'webqsp':
        webqsp_main()
    elif config['dataset'] == 'metaqa':
        metaqa_main()
    elif config['dataset'] == 'wikisql':
        wikisql_main()
    else:
        raise ValueError("Invalid dataset, valid dataset include: grailqa, graphq, webqsp and metaqa, while your dataset is ",
                         config['dataset'])
