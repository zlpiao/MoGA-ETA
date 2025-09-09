from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_curve

from timeit import default_timer as timer
import time
from utils.utils import time_to_str


def eval0(valid_dataloader, model, norm_flag, return_prob=False):
  """
    Eval the FAS model on the target data.
    The function evaluates the model on the target data and return TPR@FPR, HTER
    and AUC.
    Used in train.py to eval the model after each epoch; used in test.py to eval
    the final model.
    """

  criterion = nn.CrossEntropyLoss()
  valid_losses = AverageMeter()
  valid_top1 = AverageMeter()
  prob_dict = {}
  label_dict = {}
  model.eval()
  output_dict_tmp = {}
  target_dict_tmp = {}
  mismatched_count = 0
  with torch.no_grad():
    for iter, (input, target, videoID, name,caption) in enumerate(valid_dataloader):
      input = Variable(input).cuda()
      target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

      starttime = time.time()
      cls_out, qformerloss = model(input, norm_flag) # for FLIP-V and FLIP-IT model
      # cls_out, _ = model.forward_eval(input, norm_flag) # for SSL-CLIP model

      prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
      label = target.cpu().data.numpy()
      videoID = videoID.cpu().data.numpy()
      #print("prob--", prob)
      #print("label--", label)
      
      # 根据张量元素是否大于0.5来预测结果
      predicted_labels = (prob > 0.5).astype(int)
      # 找出不匹配的索引号
      mismatched_indices = np.where(predicted_labels != label)[0]
      # 打印不匹配的索引号
      #print("Mismatched indices:", mismatched_indices)
      
      for i in mismatched_indices:             
          element = name[i]
          #print("name-path--", element)
      # 统计不匹配的张量索引号的总数
      mismatched_count = mismatched_count + len(mismatched_indices)

      #print("name-path--", name)
      #len1 = len(prob)
      #print("prob.shape--", prob.shape)
      #print("len(prob)--", len1)
      for i in range(len(prob)):
        if (videoID[i] in prob_dict.keys()):
          #print("here 0 -i-",i)
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
        else:
          #print("---here 1 -i-",i )  #推理测试发现进这里
          prob_dict[videoID[i]] = []
          label_dict[videoID[i]] = []
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]] = []
          target_dict_tmp[videoID[i]] = []
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
  #print("mismatched_count--", mismatched_count)        
  prob_list = []
  label_list = []
  for key in prob_dict.keys():
    avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
    avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
    prob_list = np.append(prob_list, avg_single_video_prob)
    label_list = np.append(label_list, avg_single_video_label)
    # compute loss and acc for every video
    avg_single_video_output = sum(output_dict_tmp[key]) / len(
        output_dict_tmp[key])
    avg_single_video_target = sum(target_dict_tmp[key]) / len(
        target_dict_tmp[key])
    loss = criterion(avg_single_video_output, avg_single_video_target.long())
    acc_valid = accuracy(
        avg_single_video_output, avg_single_video_target, topk=(1,))  
        
    #print("output,target,acc--",avg_single_video_output,avg_single_video_target,acc_valid) 
     
    valid_losses.update(loss.item())
    valid_top1.update(acc_valid[0])

  auc_score = roc_auc_score(label_list, prob_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  fpr, tpr, thr = roc_curve(label_list, prob_list)
  tpr_filtered = tpr[fpr <= 1 / 100]
  if len(tpr_filtered) == 0:
    rate = 0
  else:
    rate = tpr_filtered[-1]
  print("TPR@FPR = ", rate)

  if not return_prob:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate* 100
    ]
  else:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate* 100] , [ prob_list, label_list]


def eval(valid_dataloader, model, norm_flag, return_prob=False):
  """
    Eval the FAS model on the target data.
    The function evaluates the model on the target data and return TPR@FPR, HTER
    and AUC.
    Used in train.py to eval the model after each epoch; used in test.py to eval
    the final model.
    """

  criterion = nn.CrossEntropyLoss()
  valid_losses = AverageMeter()
  valid_top1 = AverageMeter()
  prob_dict = {}
  label_dict = {}
  model.eval()
  output_dict_tmp = {}
  target_dict_tmp = {}
  mismatched_count = 0
  # 新增：初始化存储图片路径、预测分数和标签的字典
  scores_pred = {}
  scores_gt = {}
  with torch.no_grad():
    for iter, (input, target, videoID, name,caption) in enumerate(valid_dataloader):
      input = Variable(input).cuda()
      target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

      starttime = time.time()
      cls_out, qformerloss = model(input, norm_flag) # for FLIP-V and FLIP-IT model
      # cls_out, _ = model.forward_eval(input, norm_flag) # for SSL-CLIP model

      prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
      label = target.cpu().data.numpy()
      videoID = videoID.cpu().data.numpy()
      
      # 新增：将图片路径、预测分数和标签存入对应字典
      for i in range(len(prob)):
          img_path = name[i]
          scores_pred[img_path] = np.array([prob[i]])
          scores_gt[img_path] = np.array([label[i]])
      
      # 以下为原有代码，保持不变
      predicted_labels = (prob > 0.5).astype(int)
      mismatched_indices = np.where(predicted_labels != label)[0]
      
      for i in mismatched_indices:             
          element = name[i]
      
      mismatched_count = mismatched_count + len(mismatched_indices)

      for i in range(len(prob)):
        if (videoID[i] in prob_dict.keys()):
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
        else:
          prob_dict[videoID[i]] = []
          label_dict[videoID[i]] = []
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]] = []
          target_dict_tmp[videoID[i]] = []
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
  
  # 新增：将结果保存为npz文件
  np.savez('test1.npz', scores_pred=scores_pred, scores_gt=scores_gt)
        
  prob_list = []
  label_list = []
  for key in prob_dict.keys():
    avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
    avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
    prob_list = np.append(prob_list, avg_single_video_prob)
    label_list = np.append(label_list, avg_single_video_label)
    
    avg_single_video_output = sum(output_dict_tmp[key]) / len(
        output_dict_tmp[key])
    avg_single_video_target = sum(target_dict_tmp[key]) / len(
        target_dict_tmp[key])
    loss = criterion(avg_single_video_output, avg_single_video_target.long())
    acc_valid = accuracy(
        avg_single_video_output, avg_single_video_target, topk=(1,))  
     
    valid_losses.update(loss.item())
    valid_top1.update(acc_valid[0])

  auc_score = roc_auc_score(label_list, prob_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  fpr, tpr, thr = roc_curve(label_list, prob_list)
  tpr_filtered = tpr[fpr <= 1 / 100]
  if len(tpr_filtered) == 0:
    rate = 0
  else:
    rate = tpr_filtered[-1]
  print("TPR@FPR = ", rate)

  if not return_prob:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate* 100
    ]
  else:
    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate* 100] , [ prob_list, label_list]
        
        
def eval_ViTAF(valid_dataloader, model, norm_flag):
  """
    Eval the FAS model on the target data.
    The function evaluates the model on the target data and return TPR@FPR, HTER
    and AUC.
    Used in train.py to eval the model after each epoch; used in test.py to eval
    the final model.
    """
  criterion = nn.CrossEntropyLoss()
  valid_losses = AverageMeter()
  valid_top1 = AverageMeter()
  prob_dict = {}
  label_dict = {}
  model.eval()
  output_dict_tmp = {}
  target_dict_tmp = {}
  with torch.no_grad():
    for iter, (input, target, videoID, name) in enumerate(valid_dataloader):
      input = Variable(input).cuda()
      target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
      cls_out, feature, total_loss = model(input, norm_flag)
      # cls_out, feature = model(input, norm_flag)
      prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
      label = target.cpu().data.numpy()
      videoID = videoID.cpu().data.numpy()

      for i in range(len(prob)):
        if (videoID[i] in prob_dict.keys()):
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
        else:
          prob_dict[videoID[i]] = []
          label_dict[videoID[i]] = []
          prob_dict[videoID[i]].append(prob[i])
          label_dict[videoID[i]].append(label[i])
          output_dict_tmp[videoID[i]] = []
          target_dict_tmp[videoID[i]] = []
          output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
          target_dict_tmp[videoID[i]].append(target[i].view(1))
  prob_list = []
  label_list = []
  for key in prob_dict.keys():
    avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
    avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
    prob_list = np.append(prob_list, avg_single_video_prob)
    label_list = np.append(label_list, avg_single_video_label)
    # compute loss and acc for every video
    avg_single_video_output = sum(output_dict_tmp[key]) / len(
        output_dict_tmp[key])
    avg_single_video_target = sum(target_dict_tmp[key]) / len(
        target_dict_tmp[key])
    loss = criterion(avg_single_video_output, avg_single_video_target.long())
    acc_valid = accuracy(
        avg_single_video_output, avg_single_video_target, topk=(1,))
    valid_losses.update(loss.item())
    valid_top1.update(acc_valid[0])

  auc_score = roc_auc_score(label_list, prob_list)
  cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
  ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
  cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

  fpr, tpr, thr = roc_curve(label_list, prob_list)
  tpr_filtered = tpr[fpr <= 1 / 100]
  if len(tpr_filtered) == 0:
    rate = 0
  else:
    rate = tpr_filtered[-1]
  print("TPR@FPR = ", rate)

  return [
      valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
      auc_score, threshold, ACC_threshold * 100, rate
  ]