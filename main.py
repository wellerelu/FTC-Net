import argparse
import os
import random
from torch.backends import cudnn

from dataset.loaddata import read_data
from dataset.dataset import get_loader
from utils.LayerSegSolver import Solver
from dataset.loaddata_prince_version import read_data_prince
from scipy import io as io2

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','FTC_Net']:
        print('ERROR!! model_type should be selected in U_Net/FTC_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    train_loader = get_loader(config.save_path,'train',config.batch_size,config.num_workers)
    test_loader = get_loader(config.save_path, 'test', 1, config.num_workers)
    print('train dataset len %d' % (len(train_loader) * config.batch_size))
    print('test dataset len %d' % len(test_loader))

    results_path = config.result_path
    layer_results_path = os.path.join(results_path, 'layers')
    dst_results_path = os.path.join(results_path, 'dst_layer')
    att_result_path = os.path.join(results_path, 'attention_map')
    confi_result_path = os.path.join(results_path, 'confidence_map')

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(layer_results_path):
        os.mkdir(layer_results_path)
        os.mkdir(dst_results_path)
    if not os.path.exists(att_result_path):
        os.mkdir(att_result_path)
    if not os.path.exists(confi_result_path):
        os.mkdir(confi_result_path)

    if config.flag == 'FTC_Net':
        solver = Solver(config, train_loader, test_loader,)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        # os.mkdir(contour_results_path)
        # os.mkdir(attmap_path)
        # solver.test(layer_results_path,dst_results_path,att_result_path,config.is_save_attsmat,'./meter')
        solver.test(layer_results_path)
    elif config.mode == 'confi':
        solver.cal_confi(confi_result_path)
    else:
        solver.get_atts(layer_results_path,dst_results_path,att_result_path,config.is_save_attsmat)

if __name__ == '__main__':     # confidence net变成平均池化
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type = str, default='CSR')  # [prince/duke/CSR]
    parser.add_argument('--flag', type=str, default='FTC_Net')

    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--save-path',type=str, default='')
    parser.add_argument('--cuda-idx', type=int, default=1)

    parser.add_argument('--model-type', type=str, default='FTC_Net')
    parser.add_argument('--mode', type=str, default='train') # [train/test/confi/att]
    parser.add_argument('--img-ch', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=11)
    parser.add_argument('--result-path',type=str,default='./result')
    parser.add_argument('--pretrained-model', type=str, default='')
    parser.add_argument('--pretrained-model-confi', type=str, default='')

    # 训练参数
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--start-weight',type=int,default=1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--confi-lr',type=float,default=1e-4)
    # parser.add_argument('--beta1', type=float, default=0.9)
    # parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--root', type=str, default='./OCT')

    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--lambda1',type=float,default=0.2)
    parser.add_argument('--lambda2',type=float,default=0.8)
    parser.add_argument('--lambda3', type=float, default=0.2)

    parser.add_argument('--is-save-attsmat',type=int,default=0)

    config = parser.parse_args()

    main(config)
