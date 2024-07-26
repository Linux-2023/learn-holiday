"""
关于知识追踪模型的蓝图
"""
import heapq
import random
from collections import defaultdict

import torch
import numpy as np
import traceback
from flask import Blueprint, request
from icecream import ic
from scipy import sparse
from .. import db
from ..alg.params import DEVICE
from ..entity import Answer, User, Question, Skill, TOPIC

kt_bp = Blueprint('kt', __name__, url_prefix='/kt')
model = torch.load(f="C:\\KT\\Flask-BackEnd\\app\\alg\\model\\2024_05_15#12_33_06_9.pt", map_location=torch.device('cpu'))
qq_table = sparse.load_npz('C:\\KT\\Flask-BackEnd\\app\\alg\\data\\qq_table.npz').toarray()
qs_table = sparse.load_npz('C:\\KT\\Flask-BackEnd\\app\\alg\\data\\qs_table.npz').toarray()
ss_table = sparse.load_npz('C:\\KT\\Flask-BackEnd\\app\\alg\\data\\ss_table.npz').toarray()
edge_index = torch.tensor(np.load('C:\\KT\\Flask-BackEnd\\app\\alg\\data\\edge_index.npy'), dtype=torch.int64)


# 返回历史答题记录
@kt_bp.route('/history', methods=['GET'])
def history_questions():
    user_id = request.args.get('userId')
    page_index = int(request.args.get('pageIndex'))
    page_size = int(request.args.get('pageSize'))
    search = request.args.get('search')  # 习题id
    ic(search)
    if search == '':
        answers = Answer.query.filter_by(user_id=user_id).all()
    else:
        try:
            search = int(request.args.get('search'))  # 习题id
        except Exception:
            ic('输入格式有误')
            return {
                'msg': '输入格式有误'
            }
        answers = Answer.query.filter_by(user_id=user_id).filter(Answer.q_id == search).all()  # 历史问题列表
    answers_page = answers[(page_index - 1) * page_size: page_index * page_size]
    # db.Model对象一定要有'_sa_instance_state'属性
    return {
        'data': [
            {
                'id': answer.id,
                'username': User.query.get(answer.user_id).username,
                'q_id': answer.q_id,
                'skills': Question.query.get(answer.q_id).skills,  # get得到的是一个对象，不是一个列表
                'correct': answer.correct
            }
            for answer in answers_page
        ],
        'num': len(answers)
    }


@kt_bp.route('/history/add', methods=['POST'])
def add():
    history_dict = request.get_json()
    ic(history_dict)
    correct = history_dict['correct']
    q_id = history_dict['q_id']
    user_id = history_dict['user_id']
    new_answer = Answer(user_id=user_id, q_id=q_id, correct=correct)
    try:
        db.session.add(new_answer)
        db.session.commit()
    except Exception:
        return {
            'code': 1,
            'msg': '由于完整性约束，无法更新'
        }
    return {
        'code': 0
    }


# 返回习题
@kt_bp.route('/problem', methods=['GET'])
def problem():
    page_index = int(request.args.get('pageIndex'))
    page_size = int(request.args.get('pageSize'))
    answers = TOPIC.query.all()
    answers_page = answers[(page_index - 1) * page_size: page_index * page_size]
    # db.Model对象一定要有'_sa_instance_state'属性
    return {
        'data': [
            {
                'id': answer.id,
                'q_id': answer.q_id,
                'topic': answer.topic,
                'answer': answer.answer
            }
            for answer in answers_page
        ],
        'num': len(answers)
    }


# 返回预测结果
@kt_bp.route('/predict', methods=['GET'])
def predict():
    q_list_now = request.args.get('qList')
    user_id = int(request.args.get('userId'))
    history_answers = Answer.query.filter_by(user_id=user_id).all()  # 该同学所有相关的问答
    q_list_history = [a.q_id for a in history_answers]  # 所有回答过的问题id
    answers_list_history = [a.correct for a in history_answers]

    try:
        q_list_now = [int(q) for q in q_list_now.split(',')]
    except Exception:
        traceback.print_exc()
        return {
            'msg': '输入格式有误'
        }
    q_list = q_list_history + q_list_now
    response = answers_list_history + [random.randint(0, 1) for _ in range(len(q_list_now))]
    ones = torch.ones(size=[1, len(q_list_now), ], dtype=torch.int, device=DEVICE)
    c_list = model(
        question=torch.unsqueeze(torch.tensor(q_list, device=DEVICE), dim=0),  # 问题id列表
        response=torch.unsqueeze(torch.tensor(response, dtype=torch.int, device=DEVICE), dim=0),  # 推荐时回答全设置为1
        mask=ones,  # 题目都是有效的，maks也全为1
        edge_index=edge_index
    ).squeeze(dim=0).tolist()
    ic(c_list)
    c_list = [round(c, 4) for c in c_list[len(q_list_history):]]
    ic(c_list)
    s_set = set()
    for q_id in q_list_now:
        s_set.update(np.where(qs_table[q_id] > 0)[0].tolist())
    s_list = list(s_set)  # 相关技能的id数组
    ic(s_list)
    s_name = [s.name for s in Skill.query.filter(Skill.id.in_(s_list)).all()]  # 技能名称数组
    s_q_num, s_q_correct = [0 for _ in range(len(s_list))], [0 for _ in range(len(s_list))]
    # 相关技能答题数量，相关技能答题正确率总和
    for q_index, q_id in enumerate(q_list_now):
        s_list1 = np.where(qs_table[q_id] > 0)[0].tolist()
        for s_id in s_list1:
            s_index = s_list.index(s_id)  # 该技能在s_list中的索引
            s_q_num[s_index] += 1
            s_q_correct[s_index] += c_list[q_index]
    s_mastery = [correct / num for correct, num in zip(s_q_correct, s_q_num)]  # 知识点掌握程度
    ic(q_list, c_list)
    return {
        'data': {
            'qList': q_list_now,
            'cList': c_list,
            'skillIndicator': [  # 相关技能信息：{技能名称，最高掌握度（设置为1）}
                {
                    'name': name,
                    'max': 1
                } for name in s_name
            ],
            'skillMastery': s_mastery
        },
        'num': len(q_list)
    }


# 返回推荐习题和预测结果
@kt_bp.route('/recommend', methods=['GET'])
def recommend():
    num = int(request.args.get('num'))  # 推荐数量
    user_id = int(request.args.get('userId'))
    history_answers = Answer.query.filter_by(user_id=user_id).all()  # 该同学所有相关的问答
    q_list = [a.q_id for a in history_answers]  # 所有回答过的问题id
    ic(q_list)
    answers_list = [a.correct for a in history_answers]
    q_set_related = set()  # 所有相关问题的id集合
    q_set_related.update(q_list)
    for q_id in q_list:
        q_set_related.update(np.where((qq_table[q_id] > 0))[0].tolist())

    q_list_related = list(q_set_related)  # 所有相关问题的id数组
    if num > len(q_list_related):  # 相关题目比需要推荐的少，允许[重复]选
        q_list_related *= (num / len(q_list_related) + 1)
    answers_list = answers_list + [random.randint(0, 1) for _ in range(len(q_list_related) - len(q_list))]
    ones = torch.zeros(size=[1, len(q_list), ], dtype=torch.int, device=DEVICE)
    c_list = model(
        question=torch.unsqueeze(torch.tensor(q_list_related, device=DEVICE), dim=0),  # 问题id列表
        response=torch.unsqueeze(torch.tensor(answers_list, dtype=torch.int, device=DEVICE), dim=0),  # 推荐时回答全设置为1
        mask=ones,  # 题目都是有效的，maks也全为1
        edge_index=edge_index
    ).squeeze(dim=0).tolist()
    sum_dict = defaultdict(list)
    count_dict = defaultdict(int)
    for c, q in zip(c_list, q_list):
        sum_dict[q].append(c)
        count_dict[q] += 1
    # 计算每个q_list的平均值
    avg_dict = {q: sum(sum_dict[q]) / count_dict[q] for q in sum_dict}
    ic(avg_dict.values(), avg_dict.keys())
    # 将结果与q_list一起排序
    recommend = heapq.nsmallest(num, zip(avg_dict.values(), avg_dict.keys()))
    c_list, q_list_recommend = [round(rec[0], 4) for rec in recommend], [rec[1] for rec in recommend]
    # 计算推荐习题中技能的分布情况
    ic(c_list, q_list_recommend)
    s_set = set()  # 所有相关技能id集合
    for q_id in q_list_recommend:
        s_set.update(np.where(qs_table[q_id] > 0)[0].tolist())
    s_list = list(s_set)  # 相关技能的id数组
    ic(s_list)
    s_names = [s.name for s in Skill.query.filter(Skill.id.in_(s_list)).all()]  # 技能名称数组
    ic(s_names)
    s_q_num = [0 for _ in range(len(s_list))]  # 每个技能涉及的问题数量
    for q_id in q_list_recommend:
        s_list1 = np.where(qs_table[q_id] > 0)[0].tolist()
        for s_id in s_list1:
            s_q_num[s_list.index(s_id)] += 1
    ic(s_q_num)
    s_values = [num / sum(s_q_num) for num in s_q_num]  # 除以和，使其元素之和为1
    return {
        'data': {
            'qList': q_list_recommend,
            'cList': c_list,
            'skillData': [{
                'value': value,
                'name': name
            } for name, value in zip(s_names, s_values)]
        }
    }


@kt_bp.route('/skillGraph', methods=['GET'])
def skill_graph():
    s_list = request.args.get('sList')
    try:
        s_list = [int(s) for s in s_list.split(',')]
    except Exception:
        return {
            'msg': '输入格式有误'
        }
    s_data = [s for s in s_list]
    # 元素格式:{'id': int, 'name': str}
    s_links = []
    for idx0, s0 in enumerate(s_list):
        s_related = np.where(ss_table[s0] > 0)[0].tolist()  # 有关联的知识点
        for s1 in s_related:
            if s1 not in s_data:
                s_data.append(s1)
            if ({'source': s_data.index(s1), 'target': idx0} not in s_links) and \
                    ({'source': idx0, 'target': s_data.index(s1)} not in s_links):  # 正向或反向存在其一就不用添加了
                s_links.append({'source': idx0, 'target': s_data.index(s1)})
    s_data = [{'id': s} for s in s_data]  # 转化为字典数组
    ic(s_data)
    for s in s_data:
        skill = Skill.query.get(s['id'])
        s['name'] = str(s['id']) + '-' + (skill.name if skill.name is not None else 'Unknown Skill')
        s['symbolSize'] = (skill.num_q + 20) / 4
    ic(s_data, s_links)
    return {
        'data': {
            'data': s_data,
            'links': s_links
        }
    }
