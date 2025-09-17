from pipeline_ecg import kd_student_learning_ho,student_learning_without_tea
from datacollection import dataset_organize,ECGdataset_prepare_finetuning_sepe
import os
import argparse
import numpy as np
import warnings
def ECG_config(seed,root,zo_config='Hybrid'):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, default='within')
    parser.add_argument('--model_config', type=str, default='medium')
    parser.add_argument('--semi_config', type=str, default='default')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--ranklist', type=str, default='lora_ave')
    parser.add_argument('--root', type=str, default=root)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--finetune_epoch', type=int, default=200)  # 200
    parser.add_argument('--num_class', type=int, default=25)
    parser.add_argument('--finetune_label_ratio', type=float, default=0.10)#0.10
    parser.add_argument('--pretrain_dataset', type=str, default='CODE_test')
    parser.add_argument('--finetune_dataset', type=str, default='WFDB_ChapmanShaoxing')
    # dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    parser.add_argument('--r', type=int, default=8)  # LoRA rank：[4, 8, 16]
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--interval', type=int, default=50)
    ## zero-order optimization parameter
    parser.add_argument('--zo_config', type=str, default=zo_config)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--bp_batch', type=int, default=2)
    parser.add_argument('--zo_eps', type=float, default=1e-3)
    parser.add_argument('--trainer', type=str, default='zo')  # zo_sign_opt
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_sparsity', type=float, default=None)
    parser.add_argument('--perturbation_mode', type=str, default='two_side')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--tune_bp', type=bool, default=False)
    ## hyper-parameter for hybrid_tuning
    parser.add_argument('--coef', type=float, default=0.85)
    parser.add_argument('--no_grad_correct', type=bool, default=False)
    ## hyper-parameter for know distillation
    parser.add_argument('--enable_distillation', type=bool, default=True)
    parser.add_argument('--student_load_pretrain', type=bool, default=True)
    parser.add_argument('--leads_for_teacher', type=int, default=12)
    parser.add_argument('--leads_for_student', type=int, default=12)
    ## additional hyper-parameter
    parser.add_argument('--positive_probably', type=float, default=0.08)
    parser.add_argument('--aux_weight', type=float, default=1)
    parser.add_argument('--tuning', type=str, default='H-tuning')
    args = parser.parse_args()
    return args

def exp_main_kd_ho(args):
    args.bp_batch = 8
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_ave'
    print('current method:', method)
    args.ranklist = method

    ## Determine Trainer Config
    if args.zo_config == 'purebp' or args.zo_config == 'purebp_noencoder':
        args.tuning = 'bp'
        save_file_name = 'LoRA_results.npy'
    elif args.zo_config == 'zo_sign':
        args.tuning = 'zo'
        save_file_name = 'Zo_Sign_results.npy'
    else:
        print('FineTune !!')
        args.tuning = 'FT'
        args.ranklist = 'FT'
        save_file_name = 'Full_FT_results.npy'
    # if args.leads_for_student < 12:
    #     save_file_name = ('TeaLead' + str(args.leads_for_teacher) + '_' + 'StuLead'
    #                       + str(args.leads_for_student) + '_' + save_file_name)
    print(save_file_name)

    ## Running our Experiments
    print('current seed', args.seed)
    result_dataset = []
    for i in range(4):
        args.finetune_dataset = dataset_list[i]
        args.num_class = num_class_list[i]
        print(args.finetune_dataset)
        # print('labeled_ratio:', args.finetune_label_ratio)
        if args.enable_distillation:
            result_dataset.append(kd_student_learning_ho(args=args))
        else:
            result_dataset.append(student_learning_without_tea(args=args))
    os.chdir(args.root + '/result')
    np.save(save_file_name, result_dataset)
    
# 网格搜索
def exp_main_kd_ho_gridsearch_all(args):
    args.aux_weight = 0.02
    # zo_eps
    print('!! Grid Search zo_eps !!')
    args.bp_batch = 8
    args.r = 16
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_ave'
    # print('current method:', method)
    args.ranklist = method
    save_file_name = 'Gridsearch_zoeps_Results.npy'
    if args.leads_for_student < 12:
        save_file_name = ('TeaLead' + str(args.leads_for_teacher) + '_' + 'StuLead'
                          + str(args.leads_for_student) + '_' + save_file_name)
    print(save_file_name)
    break_flag = False
    zo_eps_list = [1e-3,5e-4,2e-4,1e-4]
    print(zo_eps_list)
    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        result = np.load(save_file_name, allow_pickle=True).tolist()
        print('file exist')
    else:
        result = []
    for para in zo_eps_list:
        args.zo_eps = para
        # print('current bp batch', args.bp_batch)
        print('current zo eps', args.zo_eps)
        # print('current r', args.r)
        result_dataset = []
        for i in range(4):
            args.ranklist = method
            args.finetune_dataset = dataset_list[i]
            args.num_class = num_class_list[i]
            print(args.finetune_dataset)
            dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight, positive_weight_ave = ECGdataset_prepare_finetuning_sepe(args=args)
            args.positive_probably = positive_weight_ave
            # print('TyT->positive_probably:', args.positive_probably)
            # print('TyT->aux_weight:', args.aux_weight)
            result_dataset.append(kd_student_learning_ho(args=args))
        result.append(result_dataset)
        print('running progress', len(result))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result)
            
    #bp_batch
    print('!! Grid Search bp_batch !!')
    args.zo_eps = 1e-3
    args.r = 16
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_ave'
    # print('current method:', method)
    args.ranklist = method
    save_file_name = 'Gridsearch_bp_batch_Results.npy'
    if args.leads_for_student < 12:
        save_file_name = ('TeaLead' + str(args.leads_for_teacher) + '_' + 'StuLead'
                          + str(args.leads_for_student) + '_' + save_file_name)
    print(save_file_name)
    break_flag = False
    bp_batch_list = [2,4,8]
    print(bp_batch_list)
    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        result = np.load(save_file_name, allow_pickle=True).tolist()
        print('file exist')
    else:
        result = []
    for para in bp_batch_list:
        args.bp_batch = para
        print('current bp batch', args.bp_batch)
        # print('current zo eps', args.zo_eps)
        # print('current r', args.r)
        result_dataset = []
        for i in range(4):
            args.ranklist = method
            args.finetune_dataset = dataset_list[i]
            args.num_class = num_class_list[i]
            print(args.finetune_dataset)
            dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight, positive_weight_ave = ECGdataset_prepare_finetuning_sepe(args=args)
            args.positive_probably = positive_weight_ave
            # print('TyT->positive_probably:', args.positive_probably)
            # print('TyT->aux_weight:', args.aux_weight)
            result_dataset.append(kd_student_learning_ho(args=args))
        result.append(result_dataset)
        print('running progress', len(result))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result)

    # LoRA rank r
    print('!! Grid Search LoRA rank r !!')
    args.zo_eps = 1e-3
    args.bp_batch = 8
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_ave'
    # print('current method:', method)
    args.ranklist = method
    save_file_name = 'Gridsearch_r_Results.npy'
    if args.leads_for_student < 12:
        save_file_name = ('TeaLead' + str(args.leads_for_teacher) + '_' + 'StuLead'
                          + str(args.leads_for_student) + '_' + save_file_name)
    print(save_file_name)
    break_flag = False
    r_list = [4,8,16]
    print(r_list)
    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        result = np.load(save_file_name, allow_pickle=True).tolist()
        print('file exist')
    else:
        result = []
    for para in r_list:
        args.r = para
        print('current r', args.r)
        # print('current bp batch', args.bp_batch)
        # print('current zo eps', args.zo_eps)
        result_dataset = []
        for i in range(4):
            args.ranklist = method
            args.finetune_dataset = dataset_list[i]
            args.num_class = num_class_list[i]
            print(args.finetune_dataset)
            dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight, positive_weight_ave = ECGdataset_prepare_finetuning_sepe(args=args)
            args.positive_probably = positive_weight_ave
            # print('TyT->positive_probably:', args.positive_probably)
            # print('TyT->aux_weight:', args.aux_weight)
            result_dataset.append(kd_student_learning_ho(args=args))
        result.append(result_dataset)
        print('running progress', len(result))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result)

def exp_main_kd_ho_evolution(args):
    args.aux_weight = 1
    args.bp_batch = 8
    args.zo_eps = 1e-3
    args.r = 16
    coef = 0.9
    lr = 0.002
    
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_ave'
    print('current method:', method)
    args.ranklist = method
    save_file_name = 'Evolution_Results.npy'
    print(save_file_name)
    
    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        result = np.load(save_file_name, allow_pickle=True).tolist()
        print('file exist')
    else:
        result = []
    # args.coef = coef
    # args.learning_rate = lr
    
    
    print('current bp batch', args.bp_batch)
    # print('current learning rate', args.learning_rate)
    # print('current coef', args.coef)
    print('current zo eps', args.zo_eps)
    print('current r', args.r)
    print('current seed', args.seed)
    
    result_dataset = []
    for j in range(4):
        args.ranlist = method
        args.finetune_dataset = dataset_list[j]
        args.num_class = num_class_list[j]
        print('TyT->',args.finetune_dataset)
        dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight, positive_weight_ave = ECGdataset_prepare_finetuning_sepe(args=args)
        args.positive_probably = positive_weight_ave
        # print('positive_weight:',positive_weight)
        print('TyT->positive_probably:', args.positive_probably)
        print('TyT->aux_weight:', args.aux_weight)
        result_dataset.append(kd_student_learning_ho(args=args))
        result.append(result_dataset)
        os.chdir(args.root + '/result')
        np.save(save_file_name, result) 
    
def Task_ECG_KD(seed, root, zo_config='H_Tuning'):
    args = ECG_config(seed, root, zo_config)
    print('seed:', args.seed)
    print('device:', args.device)
    args.finetune_dataset = args.finetune_dataset
    args.model_config = 'medium'
    print('model_config:', args.model_config)
    # args.r = 16  # LoRA rank:[4, 8, 16]
    args.semi_config = 'nosemi'
    if zo_config == 'evolution':
        exp_main_kd_ho_evolution(args)
    elif zo_config=='search':
        exp_main_kd_ho_gridsearch_all(args)
    else:
        exp_main_kd_ho(args)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", category=FutureWarning)
    Task = 'ECG'
    root = os.getcwd()
    # root = os.getcwd() 
    # 从原始数据生成.hdf5预处理数据集
    # args = ECG_config(18, root, 'H_Tuning')
    # dataset_organize(args)
    # 实验入口
    #Task_ECG_KD(18, root, 'FT') #全量微调
    #Task_ECG_KD(18, root, 'purebp') #反向传播调优
    #Task_ECG_KD(18, root, 'purebp_noencoder') #反向传播调优
    # Task_ECG_KD(18, root, 'zo_sign') #零阶优化调优
    # Task_ECG_KD(18, root, 'search') #网格搜索超参数
    Task_ECG_KD(18, root, 'evolution') #改进混合调优