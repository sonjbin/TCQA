import re
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor
import names
import random
import time
import spacy
import argparse
import numpy as np




dp_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def create_test_case(sent):
    """
    create test case for given sentence
    duplicate sentence and replace subject to random name and change year
    E.g.) He lived in Seoul in 1922.
    -> James lived in Seoul in 1922. Hellen lived in Seoul in 1938.
    param:
        sent: [str]base sentence
    return:
        triple of
        result_sent: [str]concatenated two edited sentences
        (name_a, name_b)
        (year_a, year_b)
    """
    #Replace to random name
    name_a = names.get_first_name()
    while True:
        name_b = names.get_first_name()
        if name_a != name_b:
            break
    dp_parsed = dp_predictor.predict(
        sentence=sent
    )
    #Ignore when there is no subject
    if not 'nsubj' in dp_parsed['predicted_dependencies']:
        return None
    subj_idx = dp_parsed['predicted_dependencies'].index('nsubj')

    if not dp_parsed['words'][subj_idx].isalpha():
        return None
    dp_parsed['words'][subj_idx] = name_a
    sent_a = ' '.join(dp_parsed['words'])
    dp_parsed['words'][subj_idx] = name_b
    sent_b = ' '.join(dp_parsed['words'])
    
    #Change year of sent_b
    p = re.compile(" in [\d]+|^in [\d]+")
    search_result = p.search(sent_b.lower())
    #Only use single year information
    if not search_result or len(search_result.group())>8:
        return None
    elif len(search_result.group())==8:
        year_a = int(sent_b[search_result.start()+4:search_result.end()])
    elif len(search_result.group())<8:
        year_a = int(sent_b[search_result.start()+3:search_result.end()])
    
    rand_y = random.randint(0,1)
    rand_range = 200
    if rand_y == 0:
        edit_y = random.randint(-1*rand_range,-5)
    else:
        edit_y = random.randint(5,rand_range)
    
    year_b = year_a + edit_y
    sent_b = sent_b.replace(str(year_a), str(year_b))
    result_sent = sent_a + " "+sent_b
    
    result = (result_sent, (name_a, name_b), (year_a, year_b))
    
    return result

def create_test_template(sent):
    """
    create test template for given sentence
    duplicate sentence and replace subject to random name and change year to template space
    E.g.) He lived in Seoul in 1922.
    -> James lived in Seoul [TMP1]. Hellen lived in Seoul [TMP2].
    param:
        sent: [str]base sentence
    return:
        triple of
        original_sent
        result_sent: [str]concatenated two edited sentences
        (name_a, name_b)

    """
    time_1 = ' [TMP1] '
    time_2 = ' [TMP2] '

    #Replace to random name
    name_a = names.get_first_name()
    while True:
        name_b = names.get_first_name()
        if name_a != name_b:
            break
    dp_parsed = dp_predictor.predict(
        sentence=sent
    )
    #Ignore when there is no subject
    if not 'nsubj' in dp_parsed['predicted_dependencies']:
        return None
    subj_idx = dp_parsed['predicted_dependencies'].index('nsubj')

    if not dp_parsed['words'][subj_idx].isalpha():
        return None
    dp_parsed['words'][subj_idx] = name_a
    sent_a = ' '.join(dp_parsed['words'])
    dp_parsed['words'][subj_idx] = name_b
    sent_b = ' '.join(dp_parsed['words'])
    
    #Change year of sent_b
    p = re.compile(" in [\d]{4} |^in [\d]{4} ")
    search_result = p.search(sent_b.lower())
    #Only use single year information and do not use number smaller than 1000
    if not search_result or len(search_result.group())>9 or len(search_result.group())<8:
        return None

    time_a = search_result.group()

    
    # sent_a = sent_a.lower().replace(str(time_a), time_1)
    # sent_b = sent_b.lower().replace(str(time_a), time_2)
    sent_a = (sent_a[0].lower()+sent_a[1:]).replace(str(time_a), time_1)
    sent_b = (sent_b[0].lower()+sent_b[1:]).replace(str(time_a), time_2)
    result_sent = sent_a + " "+sent_b
    
    result = (sent, result_sent, (name_a, name_b))
    
    return result




   

def generate_time_event_for_template(specifier,R=10, R_b=5,y_lower=1800, y_upper=2010):
    """
    generate random time for template with considering event duration
    params:
        R: [int] range of year
        R_b: [int] range gor between, from
        y_lower: [int] lower bound of the target year
        y_upper: [int] upper bound of the target year
        specifier: [str] one of time specifier in [in, between~and, after, before, since, until, from~to]
    return:
        generated_time: List[List[(str,str)]] generated temporal expression for 7 time specifier in, between~to, after, before, since, until, from~to
        target year and 2 expressions for each specifier(target year, answerable, unanswerable)
    """
    
    target_year = random.randint(y_lower,y_upper)
    
    #Specifier "in"
    if specifier == 'in':
        p_in = f"in {target_year + random.randint(0,R)}"
        unans_in = f"in {target_year - random.randint(1,R)}"
        
        output = [p_in, unans_in]
    
    #Specifier "between"
    if specifier == 'between':
        rand_btw = random.randint(0,1)
        if rand_btw == 0:#t1<y<t2
            p_btw = f"between {target_year-random.randint(0,R_b)} and {target_year+random.randint(0,R_b)}"

        else:#y<t1<t2
            t1 =  random.randint(0,2*R_b)
            p_btw = f"between {target_year + t1} and {target_year + random.randint(t1, 2*R_b)}"

        t1 =  random.randint(1,2*R_b)
        unans_btw = f"between {target_year - t1} and {target_year - random.randint(1, t1)}"
            
        output = [p_btw, unans_btw]
    
    #Specifier "from"
    if specifier == 'from':
        rand_from = random.randint(0,1)
        if rand_from == 0:#t1<y<t2
            p_from = f"from {target_year-random.randint(0,R_b)} to {target_year+random.randint(0,R_b)}"

        else:#y<t1<t2
            t1 =  random.randint(0,2*R_b)
            p_from = f"from {target_year + t1} to {target_year + random.randint(t1, 2*R_b)}"

        t1 =  random.randint(1,2*R_b)
        unans_from = f"from {target_year - t1} to {target_year - random.randint(1, t1)}"
            
        output = [p_from, unans_from]
    
    #Specifier "after"
    if specifier == 'after':
        p_after = f"after {target_year + random.randint(0,R)}"
        unans_after = f"after {target_year - random.randint(1,R)}"
        
        output = [p_after, unans_after]

    #Specifier "since"
    if specifier == 'since':
        p_since = f"since {target_year + random.randint(0,R)}"
        unans_since = f"since {target_year - random.randint(1,R)}"
        
        output = [p_since, unans_since]
    
    #Specifier "before"
    if specifier == 'before':
        p_before = f"before {target_year + random.randint(0,R)}"
        unans_before = f"before {target_year - random.randint(1,R)}"
        
        output = [p_before, unans_before]
    
    #Specifier "until"
    if specifier == 'until':
        p_until = f"until {target_year + random.randint(0,R)}"
        unans_until = f"until {target_year - random.randint(1,R)}"
        
        output = [p_until, unans_until]
    
    output = [f"in {target_year}"]+output

    return output

"""
generate range-form time expression
E.g.) question time: in [early, later] 1990s
      positive time: 
"""
def gen_range_exp(range_type, specifier_type, R=10,y_lower=1800, y_upper=2010):
    #exp_type = 'decade' or 'century'
    if range_type == 'decade':
        range_upper = (y_upper-y_lower)//10
        q_time = y_lower+random.randint(0,range_upper)*10
        q_time_exp = f"in {specifier_type} {q_time}s"
        negative_pool = [i for i in range(-1*R,0)]+ [10+i for i in range(0,R+1)]
        if specifier_type == 'early':
            positive_time = q_time + random.randint(0,4)
            negative_pool+= [i for i in range(5,10)]*4
            negative_time = random.sample(negative_pool,2)
            negative_time = [q_time+rand_t for rand_t in negative_time]
            negative_time_exp = [f"in {time}" for time in negative_time]
        elif specifier_type == 'late':
            positive_time = q_time + random.randint(5,9)
            negative_pool += [i for i in range(0,4)]*4
            negative_time = random.sample(negative_pool,2)
            negative_time = [q_time+rand_t for rand_t in negative_time]
            negative_time_exp = [f"in {time}" for time in negative_time]
        
        else:
            print(f"Unkown specifier type: {specifier_type}")
            return None
        
        
    elif range_type == 'century':
        range_upper = (y_upper-y_lower)//100
        q_time = y_lower+random.randint(0,range_upper)*100
        q_time_exp = f"in {specifier_type} {q_time}s"
        negative_pool = [i for i in range(-1*R*10,0)]+ [100+i for i in range(0,R*10+1)]
        if specifier_type == 'early':
            positive_time = q_time + random.randint(0,25)
            negative_pool+= [i for i in range(26,100)]*3
            negative_time = random.sample(negative_pool,2)
            negative_time = [q_time+rand_t for rand_t in negative_time]
            negative_time_exp = [f"in {time}" for time in negative_time]
        elif specifier_type == 'late':
            positive_time = q_time + random.randint(75,99)
            negative_pool += [i for i in range(0,75)]*3
            negative_time = random.sample(negative_pool,2)
            negative_time = [q_time+rand_t for rand_t in negative_time]
            negative_time_exp = [f"in {time}" for time in negative_time]
        
    
    else:
        print(f"Unkown range type: {range_type}")
        
    positive_time_exp = f"in {positive_time}" 
        
#     print(q_time_exp,'\n', positive_time_exp, '\n', negative_time_exp)
    return q_time_exp, positive_time_exp, negative_time_exp


def add_time_to_template_event(template_pos,pool,qidx, timeq='[TMPQ]', time1='[TMP1]'):
    """
    put time expressions to template
    params:
        template: [dict] {'questions': ..., 'context':, ..., }
        timeq, time1: [str] special token to indicate the location of time 
    return:
        template
    """
    outputs = []

    pool_size = len(pool)
    
    
    time_specifiers = ['in','between','after','before','from','since','until']
    for specifier in time_specifiers:

        rand1 = random.randint(0,pool_size-1)
        random_sent = pool[rand1]['original']
        while True:
            rand2 = random.randint(0,pool_size-1)
            template_neg = pool[rand2]['context']
            if len(sent_tokenize(template_neg)) != 2:
                continue
            template_neg = sent_tokenize(template_neg)[0]
            break

        question = template_pos['question']
        #For test
        # context = template_pos['context'][0].capitalize()+template_pos['context'][1:]
        # answer = template_pos['name'].capitalize()
        context = template_pos['context']
        answer = template_pos['name']
        
        time_context, time_answerable, time_unanswerable = generate_time_for_template(specifier)
        context = context.replace(time1, time_context)
        #same time different context
        template_neg = template_neg.replace(time1, time_context)
        #target context + negative context + negative time
        context_set = [context, template_neg, random_sent]
        random.shuffle(context_set)
        context = ' '.join(context_set)
        if specifier != 'after' and specifier != 'since': 
            #For unanswerable question
            question_unans = question.replace(timeq, time_unanswerable)

            idx_from, idx_end = 0,0

            data = {
                "idx": f"/P#{qidx}/{specifier}/0",
                "question": question_unans,
                "context": context,
                "targets": [''],
                "from": [idx_from],
                "end": [idx_end],
                "answerable": False,
                "time_specifier":specifier,
                #For Fid training
                "paragraphs":[{"title":"", "text":context}]
            }
            outputs.append(data)
        #For answerable question
        question_ans = question.replace(timeq, time_answerable)
        # print(context, answer)
        idx_from = context.index(answer)
        idx_end = idx_from + len(answer)
        data = {
            "idx": f"/P#{qidx}/{specifier}/1",
            "question": question_ans,
            "context": context,
            "targets": [answer],
            "from": [idx_from],
            "end": [idx_end],
            "answerable": True,
            "time_specifier":specifier,
            #For Fid training
            "paragraphs":[{"title":"", "text":context}]
        }
        outputs.append(data)

    return outputs


def generate_time_event_for_tct(specifier,R=10, R_b=5,y_lower=1800, y_upper=2010):
    """
    generate random time for template with considering event duration
    params:
        R: [int] range of year
        R_b: [int] range gor between, from
        y_lower: [int] lower bound of the target year
        y_upper: [int] upper bound of the target year
        specifier: [str] one of time specifier in [in, between~and, after, before, since, until, from~to]
    return:
        generated_time: List[List[(str,str)]] generated temporal expression for 7 time specifier in, between~to, after, before, since, until, from~to
        target year and 2 expressions for each specifier(target year, answerable, unanswerable)
    """
    
    target_year = random.randint(y_lower,y_upper)
    neg_iter = 2
    positive_time = f"in {target_year}"
    #Specifier "in"
    if specifier == 'in':
        q_year = target_year + random.randint(0,R)
        q_time = f"in {q_year}"
        
        negative_time = [f"in {q_year + random.randint(1,R)}" for _ in range(neg_iter)] 
        
    
    #Specifier "between"
    elif specifier == 'between':
        rand_btw = random.randint(0,1)
        if rand_btw == 0:#t1<y<t2
            btw1 = target_year-random.randint(0,R_b)
            btw2 = target_year+random.randint(0,R_b)

        else:#y<t1<t2
            t1 =  random.randint(0,2*R_b)
            btw1 = target_year + t1
            btw2 = target_year + random.randint(t1, 2*R_b)
        
        q_time = f"between {btw1} and {btw2}"

        negative_time = [f"in {btw2 + random.randint(1,R)}" for _ in range(neg_iter)]

    
    # #Specifier "from"
    elif specifier == 'from':
        rand_btw = random.randint(0,1)
        if rand_btw == 0:#t1<y<t2
            btw1 = target_year-random.randint(0,R_b)
            btw2 = target_year+random.randint(0,R_b)

        else:#y<t1<t2
            t1 =  random.randint(0,2*R_b)
            btw1 = target_year + t1
            btw2 = target_year + random.randint(t1, 2*R_b)
        
        q_time = f"from {btw1} to {btw2}"

        negative_time = [f"in {btw2 + random.randint(1,R)}" for _ in range(neg_iter)]
    
    #Specifier "after"
    elif specifier == 'after':
        q_time= f"after {target_year + random.randint(0,R)}"
        negative_time = None
    # #Specifier "since"
    elif specifier == 'since':
        q_time= f"since {target_year + random.randint(0,R)}"
        negative_time = None
    
    #Specifier "before"
    elif specifier == 'before':
        q_year = target_year + random.randint(0,R)
        q_time = f"before {q_year}"

        negative_time = [f"in {q_year + random.randint(1,R)}" for _ in range(neg_iter)]
    
    # #Specifier "until"
    elif specifier == 'until':
        q_year = target_year + random.randint(0,R)
        q_time = f"until {q_year}"

        negative_time = [f"in {q_year + random.randint(1,R)}" for _ in range(neg_iter)]
    
    else:
        print("Unknown specifier: ",specifier)
    

    output = [q_time, positive_time, negative_time]


    return output





def add_time_to_q(question, date):
    question = question[:-1]
    q_words = question.split()
    q_words.append(date)
    q_words.append("?")
    return ' '.join(q_words)