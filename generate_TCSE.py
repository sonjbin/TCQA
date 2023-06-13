import json
import re
import random
from nltk import sent_tokenize
import spacy
from tqdm import tqdm
import nltk
from allennlp_models import pretrained
import argparse
from question_generator.questiongenerator import QuestionGenerator
from utils_tcse import *
import names

#random seed
seed = 77
random.seed(seed)

# nltk.download('stopwords')
def check_temp_sentence(sent):
    """
    Check whether the input sentence include temporal(year) information
    param:
        sent: [str] input sentence
    """
    p_orig = re.compile(r"\b(in|after|before|between|since|until|from)\s(the\s)?\d{4}")
    p_specific = re.compile(r"\b(in|after|before|between|since|until|from)\s(the\s)?\d{4}([^0-9\s])+")
    if len(p_orig.findall(sent.lower()))==1 and len(p_specific.findall(sent.lower()))==0:
        return True
    return False

def write_file(path, data):
    with open(path, 'w') as f:
        for i,d in enumerate(data):
            json.dump(d, f)
            if i<len(data)-1:
                f.write('\n')
"""
Generate syntheticQA example from template
params:
    ...
    <Available only if add_range = True>
    range
"""
def generate_example(i, template, answerable, context_pool, sent_pool, add_range, specifier, range_type=None):
    question = template['question']
    positive_context = template['sent_template']
    random_idxs = random.sample([k for k in range(len(context_pool)) if not k == i],2)
    negative_context = [context_pool[k] for k in random_idxs]
    
    if add_range == True:
        q_time, positive_time, negative_time = gen_range_exp(range_type, specifier, R=10,y_lower=1800, y_upper=2010)
    else:  
        q_time, positive_time, negative_time = generate_time_event_for_tct(specifier, R=10,y_lower=1400, y_upper=1610)

    question = question.replace(token_time_q, q_time)
    target_name = names.get_full_name()
    fake_names = []
    while True:
        fake_name = names.get_full_name()
        if fake_name != target_name:
            fake_names.append(fake_name)
            if len(fake_names) >= 3:
                break
    context_tc = positive_context.replace(token_time_c, positive_time).replace(token_name, target_name)
    if answerable==0:
        #Remove target sentence with after, since classifier
        if negative_time == None:
            context_tc = ""
        else:
            context_tc = positive_context.replace(token_time_c, negative_time[1]).replace(token_name, target_name)
    context_t = negative_context[0].replace(token_time_c, positive_time).replace(token_name, fake_names[0])
    random_sent = sent_pool[random.randint(0,len(sent_pool)-1)]
    context_sum = [context_tc, context_t, random_sent]
    if negative_time != None:
        context_c = positive_context.replace(token_time_c, negative_time[0]).replace(token_name, fake_names[1])
        context_ = negative_context[1].replace(token_time_c, negative_time[0]).replace(token_name, fake_names[2])
        context_sum += [context_c, context_]
    
    random.shuffle(context_sum)
    example = ' '.join(context_sum)

    if answerable == 0:
        idx_from = 0
        idx_end = 0
        target_name = ''
    else:
        idx_from = example.index(target_name)
        idx_end = idx_from+len(target_name)

    if answerable and example[idx_from:idx_end] != target_name:
        print("from - end ERROR")
        print("Error context: ",f"{[example[idx_from:idx_end]]}",'\n',f"{[target_name]}")
        return None
    
    idx_context = [-1,-1,-1]
    
    if answerable == 1:
        idx_context[0] = (example.index(context_tc), example.index(context_tc)+len(context_tc)) #idx_tc

        idx_context[1] = (example.index(context_t), example.index(context_t)+len(context_t)) #idx_t
        if negative_time != None:
            idx_context[2] = (example.index(context_c), example.index(context_c)+len(context_c)) #idx_c



    return example, question, idx_from, idx_end, target_name, idx_context

def generate_crl_example(i, template, context_pool, specifier):
    question = template['question']
    positive_context = template['sent_template']
    random_idxs = random.sample([k for k in range(len(context_pool)) if not k == i],2)
    negative_context = [context_pool[k] for k in random_idxs]
    
    q_time, positive_time, negative_time = generate_time_event_for_tct(specifier, R=10,y_lower=1400, y_upper=1610)

    question = question.replace(token_time_q, q_time)
    target_name = names.get_full_name()
    fake_names = []
    while True:
        fake_name = names.get_full_name()
        if fake_name != target_name:
            fake_names.append(fake_name)
            if len(fake_names) >= 3:
                break
    context_tc = positive_context.replace(token_time_c, positive_time).replace(token_name, target_name)

    context_t = negative_context[0].replace(token_time_c, positive_time).replace(token_name, fake_names[0])

    if negative_time != None:
        context_c = positive_context.replace(token_time_c, negative_time[0]).replace(token_name, fake_names[1])
        context_ = negative_context[1].replace(token_time_c, negative_time[0]).replace(token_name, fake_names[2])

    negative = [context_c, context_t, context_]
    random.shuffle(negative)

    return question, context_tc, negative

p_extract = re.compile(r"(\b(?:in|after|before|between|since|until|from|In|After|Before|Between|Since|Until|From)\s(the\s)?\d{4}\s(?:and\s\d{4}|to\s\d{4})?)")
token_name = '[NAME]'
token_time_c = '[TMPC]'
token_time_q = '[TMPQ]'

def main(args):
    print("Start...")
    root_folder = './dataset/'
    #Dataset type
    folder_name = f'synth_{args.data}'
    #Load and use test set
    if args.loadtc == False:
        train_data = []
        path = root_folder+f'{args.data}.hard.json'
        with open(path,'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                train_data.append(dic)
        
        #Train context set
        train_context = []
        idx_set = []
        
        for i in range(len(train_data)):
            idx = train_data[i]['idx'].split('#')[0]
            #New context
            if not idx in idx_set:
                idx_set.append(idx)
                train_context.append(train_data[i]['context'])

        print("# of contexts: ",len(train_context))
    
        sent_pool = []
        nlp = spacy.load('en_core_web_sm')
        dp_predictor = pretrained.load_predictor('structured-prediction-biaffine-parser')
    #collect all temporal sentences
        temporal_sents = []
        for context in tqdm(train_context):
            doc = nlp(context)
            sents = [sent.text for sent in doc.sents]
            for sent in sents:
                #Collect all sentences to make context pool
                sent_pool.append(sent)
                if check_temp_sentence(sent):
                    temporal_sents.append(sent)

        tcname = root_folder+f'{folder_name}/synthQA_sent_pool.json'
        write_file(tcname, sent_pool)

        start = time.time()
        print("Extracting all temporal sentences...")
        
        result_list = []
        for sent in tqdm(temporal_sents):
            sent_orig = sent
            
            dp_parsed = dp_predictor.predict(
                sentence=sent
            )
            #Ignore when there is no subject
            if not 'nsubj' in dp_parsed['predicted_dependencies']:
                continue
            subj_idx = dp_parsed['predicted_dependencies'].index('nsubj')
            if not dp_parsed['words'][subj_idx].isalpha():
                continue
            dp_parsed['words'][subj_idx] = token_name
            sent_template = ' '.join(dp_parsed['words'])
            name_ = names.get_full_name()
            sent_name = sent_template.replace(token_name, name_)
            #Extract the only one time expression
            time_expression = p_extract.findall(sent)
            if len(time_expression) != 1:
                continue
            time_expression = time_expression[0][0]
            sent_template = sent_template.replace(time_expression, token_time_c+" ")
            result_list.append({'sent_orig':sent_orig, 'sent_name':sent_name, 'name':name_, 'sent_template':sent_template})

        #Save sentences
        tcname = root_folder+f'{folder_name}/synthQA_sent_template.json'
        write_file(tcname, result_list)

        return
    elif args.loadq == False:
        print("Load templates...")
        template_list=[]
        tcname = root_folder+f'{folder_name}/synthQA_sent_template.json'
        with open(tcname,'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                template_list.append(dic)
        # template_list = []
        # template_list = [(tc['sent_orig'],tc['sent_name'],tc['name'],['sent_template']) for tc in template_list]

        print(f"# of Generated templates: {len(template_list)}")

        qg = QuestionGenerator()
        print("Load pretrained question generator...")
        q_template_list = []
        idx = 0

        s = 0
        e = len(template_list)
        print(f"Start: {s}, End: {e}")
        for template in tqdm(template_list[s:e]):
            generated_qs = qg.generate(template['sent_name'], answer_style = "all",num_questions=5)
            target_name = template['name']
            for sample in generated_qs:
                question = sample['question']
                #Do not contain template name
                if question.lower().split()[0] == 'who' and not template['name'] in question:
                    if isinstance(sample['answer'], str):
                        answer = sample['answer']
                    else:
                        answer = [candidate['answer'] for candidate in sample['answer'] if candidate['correct']][0]
                    
                    #Check whether the answer is correct answer
                    if answer == target_name:
                        #Remove time expression if there is
                        #To handle exception case like in 2000?
                        question = question.replace("?"," ?")
                        time_exp = p_extract.findall(question)
                        if len(time_exp)>0:
                            time_exp = time_exp[0][0]
                            question = question.replace(time_exp, '')
                        #Add special token for question time
                        question = question.replace("?",f"{token_time_q} ?")
                        data_idx = f"/C#{idx}"
                        data = {'idx':data_idx, 'question':question, 'sent_template':template['sent_template']}
                        q_template_list.append(data)
                        idx+=1

        print("# of Generated case: ",len(q_template_list))
        tcname = root_folder+f'{folder_name}/synthQA_sent_template_withq.json'
        write_file(tcname, q_template_list)


        return
    
    #Generate full synthetic data
    else:
        print("Load templates with questions...")
        template_list = []
        sent_pool = []
        tcname = root_folder+f'{folder_name}/synthQA_sent_template_withq.json'
        with open(tcname,'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                template_list.append(dic)
        tcname = root_folder+f'{folder_name}/synthQA_sent_pool.json'
        with open(tcname,'r') as f:
                for line in f.readlines():
                    dic = json.loads(line)
                    sent_pool.append(dic)
        
        print(f"Loaded {len(template_list)} of templates")
        context_pool = [template['sent_template'] for template in template_list]

        time_specifiers = ['in','between','after','before','since','until','from']
        #Context contains context_tc, context_t, context_c, context_, random_sentence
        if not args.crl:
            examples = []
            idx = 0
            for i, template in enumerate(tqdm(template_list)):
                if args.add_range == True:
                    range_type = ['decade','century'][random.randint(0,1)]
                    specifier_type = ['early','late'][random.randint(0,1)]
                    for answerable in range(2):
                        example = generate_example(i, template, answerable, context_pool, sent_pool, True, specifier_type, range_type)
                        if example != None:
                            example, question, idx_from, idx_end, target_name,idx_context = example
                            examples.append({'idx': f"/P#{idx}/A{answerable}",'question':question,'context':example, 'targets':[target_name],'from':[idx_from], 'end':[idx_end]})
                    idx+=1

                for specifier in time_specifiers:
                    for answerable in range(2):
                        example = generate_example(i, template, answerable, context_pool, sent_pool, False, specifier)
                        if example != None:
                            example, question, idx_from, idx_end, target_name, idx_context = example
                            examples.append({'idx': f"/P#{idx}/A{answerable}",'question':question,'context':example, 'targets':[target_name],'from':[idx_from], 'end':[idx_end], 'idx_context':idx_context})
                    idx+=1
        else:
            examples = []
            idx = 0
            for i, template in enumerate(tqdm(template_list)):
                for specifier in ['in','between','before','until','from']:
                    example = generate_crl_example(i, template, context_pool, specifier)
                    if example != None:
                        question, positive, negative  = example
                        examples.append({'idx': f"/P#{idx}/",'question':question, 'positive':positive, 'negative':negative})
                    idx+=1


        print("Generated examples: ",len(examples))
        

        fname = root_folder+f'{folder_name}/{args.data}.synthQA.json'
        
        if args.add_range:
            fname = root_folder+f'{folder_name}/{args.data}.synthQA.addrange.json'
        if args.crl:
            fname = root_folder+f'{folder_name}/{args.data}.synthCRL.json'
        print("Stored to ",fname)
        with open(fname, 'w') as f:
            for i,d in enumerate(examples):
                json.dump(d, f)
                if i<len(examples)-1:
                    f.write('\n')

    print("Finish!")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadtc',type=str2bool, default=True)
    parser.add_argument('--loadq',type=str2bool, default=False)
    parser.add_argument('--add_range',type=str2bool, default=False)
    parser.add_argument('--data', default='train')
    parser.add_argument('--crl', type=str2bool,default=False)
    args = parser.parse_args()
    main(args)
