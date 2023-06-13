import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import pandas as pd
from transformers import *
from utils import get_raw_scores
import hydra
from omegaconf import DictConfig
from tqdm import tqdm, trange
from functools import partial
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import logging
from transformers.data.processors.utils import DataProcessor
from multiprocessing import Pool, cpu_count
import numpy as np
from torch.utils.data import TensorDataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.big_bird.modeling_big_bird import BigBirdOutput, BigBirdIntermediate
from transformers import PreTrainedModel
import gzip
from omegaconf import OmegaConf

import random
seed = 77
random.seed(seed)
torch.manual_seed(seed)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score_idx = torch.argmax(scores).item()
    best_score = torch.max(scores).item()

    return best_start_idx[best_score_idx % top_k], best_end_idx[best_score_idx // top_k], best_score

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class TSQAExample(object):
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        
        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    #For RoBERTa
                    # doc_tokens.append(' '+c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

class TSQAProcessor(DataProcessor):
    # Process TSQA dataset
    def __init__(self, data_path: str):
        self.dataset = []
        if data_path.endswith('gzip'):
            with gzip.open(data_path, 'r') as f:
                for line in f:
                    self.dataset.append(json.loads(line))
        else:
            with open(data_path, 'r') as f:
                for line in f:
                    self.dataset.append(json.loads(line))
        logger.info(f'original json file contains {len(self.dataset)} entries')

    # Processing the example
    def _create_examples(self, is_training: bool):
        examples = []
        for entry in tqdm(self.dataset):         
            qas_id = entry["idx"]
            context_text = entry["context"]            
            question_text = entry["question"]
            if is_training:
                for target, start_position in zip(entry['targets'], entry['from']):
                    example = TSQAExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=target,
                        start_position_character=start_position,
                        is_impossible=len(target) == 0,
                    )
                    examples.append(example)
            else:
                for target in entry['targets']:
                    example = TSQAExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=target,
                        start_position_character=None,
                        is_impossible=len(target) == 0,
                    )
                    examples.append(example)
        return examples

    # Get reference mapping
    def _get_reference(self):
        references = {}
        for entry in self.dataset:
            references[entry['idx']] = entry['targets']
        return references

class TSQAFeatures(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        cls_index,
        qas_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.cls_index = cls_index
        self.qas_id = qas_id

        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position

        self.is_impossible = is_impossible

def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    #To perform RoBERTa, change " " to ""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    features = []

    with Pool(cpu_count(), initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert examples to features",
            )
        )

    # Gather the feature outputs
    mapping = {}
    new_features = []
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            
            if example_feature.qas_id not in mapping:
                mapping[example_feature.qas_id] = len(mapping)
            example_feature.qas_id = mapping[example_feature.qas_id]

            new_features.append(example_feature)
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.long)
    
    if not is_training:
        all_qas_ids = torch.tensor([f.qas_id for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_qas_ids
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_start_positions,
            all_end_positions,
            all_is_impossible,
        )

    return mapping, dataset

def convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    assert isinstance(example, TSQAExample), str(type(example))
    
    if is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        #To perform RoBERTa, change " " to ""
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        if actual_text.find(example.answer_text) == -1:
            print(example.doc_tokens)
            print(start_position, end_position)
            logger.warning("Could not find answer: '%s' vs. '%s' in '%s'", actual_text, example.answer_text, example.qas_id)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []
    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, 
            max_length=max_query_length, truncation=True)
    sequence_added_tokens = tokenizer.model_max_length - tokenizer.max_len_single_sentence

    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    span_doc_tokens = all_doc_tokens

    while len(spans) * doc_stride < len(all_doc_tokens):
        encoded_dict = tokenizer.encode_plus(
            truncated_query,
            span_doc_tokens,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            padding='max_length',
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation=True
        )
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or len(encoded_dict["overflowing_tokens"]) <= 0:
            if "overflowing_tokens" in encoded_dict:
                del encoded_dict["overflowing_tokens"]
            break
        
        span_doc_tokens = encoded_dict["overflowing_tokens"]
        del encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)
        assert cls_index == 0, cls_index

        span_is_impossible = example.is_impossible
        
        # Setting start/end position to the last + 1 position to be ignored
        start_position = len(span['input_ids'])
        end_position = len(span['input_ids'])
        
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                span_is_impossible = True
            else:
                doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        #Remove unanswerable case created from answerable question in Bert training
        if max_seq_length==4096 or not is_training or span_is_impossible == example.is_impossible:
            features.append(
                TSQAFeatures(
                    input_ids=span["input_ids"],
                    attention_mask=span["attention_mask"],
                    cls_index=cls_index,
                    qas_id=example.qas_id,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                )
            )
            if is_training and example.is_impossible:
                break
    return features

class BigBirdNullHead(nn.Module):
    """Head for question answering tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, encoder_output):
        hidden_states = self.dropout(encoder_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        logits = self.qa_outputs(hidden_states)
        return logits

class BigBirdForQuestionAnsweringWithNull(PreTrainedModel):
    def __init__(self, config, model_id):
        super().__init__(config)
        self.bertqa = BigBirdForQuestionAnswering.from_pretrained(model_id,
            config=self.config, add_pooling_layer=True)

        self.null_classifier = BigBirdNullHead(self.bertqa.config)



    def forward(self, **kwargs):
        if self.training:
            null_labels = kwargs['is_impossible']
            del kwargs['is_impossible']
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.pooler_output
            null_logits = self.null_classifier(pooler_output)
            loss_fct = CrossEntropyLoss()
            null_loss = loss_fct(null_logits, null_labels)

            outputs.loss = outputs.loss + null_loss

            return outputs.to_tuple()
        else:
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.pooler_output
            null_logits = self.null_classifier(pooler_output)

            return (outputs.start_logits, outputs.end_logits, null_logits)
    
    def forward_once(self, x):

        outputs = self.bertqa(**x)
        pooler_output = outputs.hidden_states[-1][:,0]
        #Need only pooler output for calculating similarity
        return pooler_output
    
    def forward_crl(self, context, question):#inputs: context, positive_q, negative_q1, negative_q2
        output_c = self.forward_once(context)
        output_q = self.forward_once(question)

        return output_c, output_q #[Batch size, 2, vector dim]
        
        # torch.mean(outputs1[0],dim=1)


class BertForQuestionAnsweringWithNull(PreTrainedModel):
    def __init__(self, config, model_id):
        super().__init__(config)
        self.bertqa = AutoModelForQuestionAnswering.from_pretrained(model_id,
                config=self.config)
        self.null_classifier = BigBirdNullHead(self.bertqa.config)
        self.fc = nn.Linear(768, 100)

    def forward(self, **kwargs):
        if self.training:
            null_labels = kwargs['is_impossible']
            del kwargs['is_impossible']
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.hidden_states[-1][:,0]
            null_logits = self.null_classifier(pooler_output)
            loss_fct = CrossEntropyLoss()
            null_loss = loss_fct(null_logits, null_labels)

            outputs.loss = outputs.loss + null_loss

            return outputs.to_tuple()
        else:
            outputs = self.bertqa(**kwargs)
            pooler_output = outputs.hidden_states[-1][:,0]
            null_logits = self.null_classifier(pooler_output)

            return (outputs.start_logits, outputs.end_logits, null_logits)


    def forward_once(self, x):

        outputs = self.bertqa(**x)
        pooler_output = outputs.hidden_states[-1][:,0]
        #Need only pooler output for calculating similarity
        return pooler_output
    
    def forward_crl(self, context, question):#inputs: context, positive_q, negative_q1, negative_q2
        output_c = self.forward_once(context)
        output_q = self.forward_once(question)

        return output_c, output_q #[Batch size, 2, vector dim]
        
        # torch.mean(outputs1[0],dim=1)

class CRL_loss(nn.Module):
    def __init__(self, batch_size, temperature=0.05):
        super().__init__()
        #ToDo
        self.batch_size = batch_size
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = nn.CosineSimilarity(dim=1)

    def forward(self, output_c, output_q, labels):
        """
        params:

            label = [0 or 1]
                0: question is negative sample
                1: question is positive sample
        """

        similarities = self.sim_func(output_c, output_q) #[batch_size]
        labels = labels[0]
        negative = torch.sum(0.5*(1-labels)*torch.exp(similarities))/torch.sum(1-labels)
        positive = torch.sum(0.5*labels*torch.exp(1-similarities))/torch.sum(labels)
        loss = (torch.sum(0.5*(1-labels)*torch.exp(similarities))/torch.sum(1-labels))+(
                            torch.sum(0.5*labels*torch.exp((1-similarities)))/torch.sum(labels))

        return loss,negative, positive

class CRLDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.examples = []
        #load training data
        with open(data_path,'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                self.examples.append(dic)
                # if len(self.examples) >1000:
                #     break

        random.shuffle(self.examples)
        n_num = len(self.examples[0]['negative'])
        text_c = []
        text_q = []
        self.feature_c = []
        self.feature_q = []
        self.labels = []
        #convert examples to contexts and questions
        for example in self.examples:
            question = example['question']
            positive = example['positive']
            negative = example['negative']
            #1 positive q and n_num negative q corresponded with same question
            text_q += [question]*(1+n_num)
            text_c += [positive]+negative
            self.labels += [1]+[0 for _ in range(n_num)]
        
        assert(len(text_q) == len(text_c) and len(text_c)==len(self.labels))
        #convert texts to features
        print("Converting texts to features...")
        self.feature_c = tokenizer(text_c,return_tensors="pt",padding=True)['input_ids']
        self.feature_q = tokenizer(text_q,return_tensors="pt",padding=True)['input_ids']
        self.labels = torch.tensor(self.labels)
        print("Total ",len(self.feature_c), " of pairs.")
            
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.feature_q[idx], self.feature_c[idx], self.labels[idx]

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.cuda:
        cfg.n_gpu = 1
        device = torch.device(f'cuda:{cfg.cuda}')
    else:
        cfg.n_gpu = torch.cuda.device_count()
        device = torch.device('cuda')
    print(cfg)

    OmegaConf.save(config=cfg, f='config.yaml')

    if cfg.model_id == 'triviaqa':
        model_id = "google/bigbird-base-trivia-itc"
    elif cfg.model_id == 'nq':
        model_id = "vasudevgupta/bigbird-roberta-natural-questions"
    elif cfg.model_id == 'bertbase':
        model_id = "bert-base-uncased"
    elif cfg.model_id == 'robertabase':
        model_id = "deepset/roberta-base-squad2"
    elif cfg.model_id == 'albertbase':
        model_id = "albert-base-v2"
    else:
        raise ValueError('Unknown model id!')

    if not cfg.use_bert:
        tokenizer = BigBirdTokenizer.from_pretrained(model_id)
        config = BigBirdConfig.from_pretrained(model_id)
        model = BigBirdForQuestionAnsweringWithNull(config, model_id)
    else:
        if cfg.model_id == 'bertbase':
            config = BertConfig.from_pretrained(model_id)
            tokenizer = BertTokenizer.from_pretrained(model_id)
        elif cfg.model_id == 'robertabase':
            config = RobertaConfig.from_pretrained(model_id)
            tokenizer = RobertaTokenizer.from_pretrained(model_id)
        elif cfg.model_id == 'albertbase':
            config = AlbertConfig.from_pretrained(model_id)
            tokenizer = AlbertTokenizer.from_pretrained(model_id)
        model = BertForQuestionAnsweringWithNull(config, model_id)
    

    model = model.to(device)
    print(config)

    if cfg.model_path:
        logger.info('loading model from {}'.format(cfg.model_path))
        state_dict = torch.load(os.path.join(cfg.model_path, 'pytorch_model.bin'), map_location='cuda:0')
        print("missing keys info ",model.load_state_dict(state_dict, strict=False))


    if cfg.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # print(model.device_ids[0])
        model.to(f'cuda:{model.device_ids[0]}')

    root_folder = os.path.dirname(os.path.dirname(__file__))

    ############## Define evaluation function

    def evaluation(model, tokenizer, cfg, output_dir, epoch, data='dev'):

        model.eval()
        if data=='dev':
            processor = TSQAProcessor(os.path.join(root_folder, cfg.dataset.dev_file))
        elif data=='test':
            processor = TSQAProcessor(os.path.join(root_folder, cfg.dataset.test_file))
        examples = processor._create_examples(is_training=False)
        logger.info('Finished processing the examples')

        references = processor._get_reference()
        mapping, dataset = convert_examples_to_features(
            examples, tokenizer, cfg.max_sequence_length, 
            cfg.doc_stride, cfg.max_query_length, False
        )
        imapping = {v:k for k, v in mapping.items()}

        # validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)

        logger.info('Finished converting the examples')

        batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)

        outputs = {}
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "output_hidden_states" : True
            }

            with torch.no_grad():
                scores = model(**inputs)
                if len(scores) == 3:
                    start_scores, end_scores, null_scores = scores
                elif len(scores) == 2:
                    start_scores, end_scores = scores
                else:
                    raise ValueError(scores)

                for i in range(start_scores.size(0)):
                    is_impossible = null_scores[i].argmax().item()
                    qas_id = batch[2][i].item()

                    if not is_impossible or cfg.TCAS:
                        start_index, end_index, score = get_best_valid_start_end_idx(start_scores[i], end_scores[i], top_k=8, max_size=16)
                        input_ids = inputs["input_ids"][i].tolist()
                        answer_ids = input_ids[start_index: end_index + 1]
                        answer = tokenizer.decode(answer_ids)

                        if imapping[qas_id] not in outputs:
                            outputs[imapping[qas_id]] = (answer, score, start_index.item(), end_index.item())
                        else:
                            if score > outputs[imapping[qas_id]][1]:
                                outputs[imapping[qas_id]] = (answer, score, start_index.item(), end_index.item())
                    else:
                        outputs[imapping[qas_id]] = ('', -10000)

        outputs_with_idx = {k: v for k, v in outputs.items()}
        outputs = {k: v[0] for k, v in outputs.items()}
        scores = get_raw_scores(outputs, references)
        print('evaluation results', scores)
        logger.info('evaluation results', scores)

        if epoch >= 0:
            print(f'evaluation results of epoch {epoch}', scores)
            with open(f'score_{epoch}.json', 'w') as f:
                json.dump(scores,f)
            with open('output.json', 'w') as f:
                json.dump(outputs_with_idx, f, indent=2)
        else:
            with open(f'output.{data}.json', 'w') as f:
                json.dump(outputs_with_idx, f, indent=2)

        return scores

    def run_eval(n_runs, data):
        exact = []
        f1 = []
        scores_list = []
        #average of 3 run
        for i in range(n_runs):
            scores = evaluation(model, tokenizer, cfg, "", -1,data)
            scores_list.append(scores)
            exact.append(scores['exact'])
            f1.append(scores['f1'])
        exact = np.array(exact)
        f1 = np.array(f1)
        mean_em = round(np.mean(exact),2)
        mean_f1 = round(np.mean(f1),2)
        std_em = round(np.std(exact),2)
        std_f1 = round(np.std(f1),2)
        average_scores = {'exact':(mean_em, std_em) , 'f1':(mean_f1,std_f1), 'data':data}
        print(f'evaluation results of {data} set: ',average_scores)
        with open(f'score.{data}.json', 'w') as f:
            json.dump(average_scores,f)
            for scores in scores_list:
                json.dump(scores,f)

    ################

    if cfg.mode == 'eval':
        
        n_runs = 3
        run_eval(n_runs, 'dev')
        run_eval(n_runs, 'test')

    
    if cfg.mode == 'train':
        tb_writer = SummaryWriter(log_dir='')

        processor = TSQAProcessor(os.path.join(root_folder, cfg.dataset.train_file))
        examples = processor._create_examples(is_training=True)
        logger.info('Finished processing the examples')

        _, dataset = convert_examples_to_features(
            examples, tokenizer, cfg.max_sequence_length, 
            cfg.doc_stride, cfg.max_query_length, True
        )
        logger.info('Finished converting the examples')

        batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
        # dataset = torch.utils.data.Subset(dataset, [i for i in range(35000)])
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        print(f"# of TimeQA data: {len(dataloader)}")
        print(f"# of total TimeQA data: {len(dataloader)*batch_size}")


        
        if cfg.TCSE and cfg.CRL:
            processor = TSQAProcessor(os.path.join(root_folder, cfg.dataset.train_synth_file))
            examples_ = processor._create_examples(is_training=True)
            logger.info('Finished processing the synthetic examples')

            _, dataset_ = convert_examples_to_features(
                examples_, tokenizer, cfg.max_sequence_length, 
                cfg.doc_stride, cfg.max_query_length, True
            )
            logger.info('Finished converting the synthetic examples')
            batch_size_synth = cfg.per_gpu_train_batch_size_tcse * max(1, cfg.n_gpu)
            #Get subset of dataset
            dataset_ = torch.utils.data.Subset(dataset_, [i for i in range(len(dataloader)*batch_size_synth)])
            sampler_synth = RandomSampler(dataset_)
            dataloader_synth = DataLoader(dataset_, sampler=sampler_synth, batch_size=batch_size_synth)

            training_data = CRLDataset(os.path.join(root_folder,cfg.dataset.train_crl), tokenizer)
            #If set shuffle to True, sometimes all examples are negative -> divided by 0
            batch_size_crl = cfg.per_gpu_train_batch_size_crl* max(1, cfg.n_gpu)
            dataloader_crl = DataLoader(training_data, batch_size=batch_size_crl, shuffle=False, drop_last=True)
            #smaller one between timeqa and synth
            print(f"# of TimeQA data: {len(dataloader)}, # of synth data: {len(dataloader_synth)}, # of crl data: {len(dataloader_crl)}")
            print(f"# of total TimeQA data: {len(dataloader)*batch_size}, # of total synth data: {len(dataloader_synth)*batch_size_synth}, # of total synth data: {len(dataloader_crl)*batch_size_crl}")
            if len(dataloader) > len(dataloader_synth) or len(dataloader) > len(dataloader_crl):
                print("Warning!!! timeQA data lost")
            len_small = min(len(dataloader), len(dataloader_synth), len(dataloader_crl))
            dataloader = zip(dataloader, dataloader_synth, dataloader_crl)

        elif cfg.TCSE:
            processor = TSQAProcessor(os.path.join(root_folder, cfg.dataset.train_synth_file))
            examples_ = processor._create_examples(is_training=True)
            logger.info('Finished processing the synthetic examples')

            _, dataset_ = convert_examples_to_features(
                examples_, tokenizer, cfg.max_sequence_length, 
                cfg.doc_stride, cfg.max_query_length, True
            )
            logger.info('Finished converting the synthetic examples')
            batch_size_synth = cfg.per_gpu_train_batch_size_tcse * max(1, cfg.n_gpu)
            #Get subset of dataset
            dataset_ = torch.utils.data.Subset(dataset_, [i for i in range(len(dataloader)*batch_size_synth)])
            sampler_synth = RandomSampler(dataset_)
            dataloader_synth = DataLoader(dataset_, sampler=sampler_synth, batch_size=batch_size_synth)
            #smaller one between timeqa and synth
            print(f"# of TimeQA data: {len(dataloader)}, # of synth data: {len(dataloader_synth)}")
            print(f"# of total TimeQA data: {len(dataloader)*batch_size}, # of total synth data: {len(dataloader_synth)*batch_size_synth}")
            if len(dataloader) > len(dataloader_synth):
                print("Warning!!! timeQA data lost")
            len_small = min(len(dataloader), len(dataloader_synth))
            dataloader = zip(dataloader, dataloader_synth)

        elif cfg.CRL:
            training_data = CRLDataset(os.path.join(root_folder,cfg.dataset.train_crl), tokenizer)
            #If set shuffle to True, sometimes all examples are negative -> divided by 0
            batch_size_crl = cfg.per_gpu_train_batch_size_crl* max(1, cfg.n_gpu)
            dataloader_crl = DataLoader(training_data, batch_size=batch_size_crl, shuffle=False, drop_last=True)
            print(f"# of TimeQA data: {len(dataloader)}, # of CRL data: {len(dataloader_crl)}")
            print(f"# of total TimeQA data: {len(dataloader)*batch_size}, # of total crl data: {len(dataloader_crl)*batch_size_crl}")
            if len(dataloader) > len(dataloader_crl):
                print("Warning!!! timeQA data lost")
            len_small = min(len(dataloader), len(dataloader_crl))
            dataloader = zip(dataloader, dataloader_crl)

        

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)

        if cfg.CRL:
            crl_loss = CRL_loss(batch_size_crl)

        if cfg.TCSE  or cfg.CRL:
            t_total = len_small * cfg.num_train_epochs
        else:
            t_total = len(dataloader) * cfg.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Num Epochs = %d", cfg.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)
        
        global_step = 1
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()

        iterator = trange(0, int(cfg.num_train_epochs), desc="Epoch")

        for epoch in iterator:
            #Redefine dataloader
            
            if cfg.TCSE and cfg.CRL and epoch>0:
                sampler = RandomSampler(dataset)
                dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
                sampler_synth = RandomSampler(dataset_)
                dataloader_synth = DataLoader(dataset_, sampler=sampler_synth, batch_size=batch_size_synth)
                dataloader_crl = DataLoader(training_data, batch_size=batch_size_crl, shuffle=False, drop_last=True)
                dataloader = zip(dataloader, dataloader_synth, dataloader_crl)

            elif cfg.TCSE and epoch > 0:
                sampler = RandomSampler(dataset)
                dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
                sampler_synth = RandomSampler(dataset_)
                dataloader_synth = DataLoader(dataset_, sampler=sampler_synth, batch_size=batch_size_synth)
                #smaller one between timeqa and synth
                dataloader = zip(dataloader, dataloader_synth)


            elif cfg.CRL and epoch > 0:
                sampler = RandomSampler(dataset)
                dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
                training_data = CRLDataset(os.path.join(root_folder,cfg.dataset.train_crl), tokenizer)
                #If set shuffle to True, sometimes all examples are negative -> divided by 0
                dataloader_crl = DataLoader(training_data, batch_size=batch_size_crl, shuffle=False, drop_last=True)
                dataloader = zip(dataloader, dataloader_crl)

            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
                model.train()


                if cfg.TCSE and cfg.CRL:
                    batch_timeqa, batch_synth, batch_crl = batch
                    batch_timeqa = tuple(t.to(device) for t in batch_timeqa)
                    batch_synth = tuple(t.to(device) for t in batch_synth)
                    batch_crl = tuple(t.to(device) for t in batch_crl)

                    inputs_timeqa = {
                        "input_ids": batch_timeqa[0],
                        "attention_mask": batch_timeqa[1],
                        "start_positions": batch_timeqa[2],
                        "end_positions": batch_timeqa[3],
                        "is_impossible": batch_timeqa[4],
                        "output_hidden_states" : True
                    }

                    inputs_synth = {
                        "input_ids": batch_synth[0],
                        "attention_mask": batch_synth[1],
                        "start_positions": batch_synth[2],
                        "end_positions": batch_synth[3],
                        "is_impossible": batch_synth[4],
                        "output_hidden_states" : True
                    }

                    feature_q = batch_crl[0],
                    feature_c = batch_crl[1],
                    inputs_q = {
                        "input_ids": feature_q[0],
                        "output_hidden_states" : True
                    }
                    inputs_c = {
                        "input_ids": feature_c[0],
                        "output_hidden_states" : True
                    }
                    labels = batch_crl[2],

                    outputs_timeqa = model(**inputs_timeqa)
                    outputs_synth = model(**inputs_synth)
                    loss_timeqa = outputs_timeqa[0]
                    loss_synth = outputs_synth[0]
                    output_c, output_q = model.forward_crl(inputs_c, inputs_q)

                    loss_crl, n, p = crl_loss(output_c, output_q, labels)
                    #Replace nan to 0
                    loss_timeqa = torch.nan_to_num(loss_timeqa)
                    loss_synth = torch.nan_to_num(loss_synth)
                    loss_crl = torch.nan_to_num(loss_crl)
                    #2:1 weight
                    loss = loss_timeqa+cfg.k*loss_synth+cfg.k_crl*loss_crl
                
                elif cfg.TCSE:
                    batch_timeqa, batch_synth = batch
                    batch_timeqa = tuple(t.to(device) for t in batch_timeqa)
                    batch_synth = tuple(t.to(device) for t in batch_synth)

                    inputs_timeqa = {
                        "input_ids": batch_timeqa[0],
                        "attention_mask": batch_timeqa[1],
                        "start_positions": batch_timeqa[2],
                        "end_positions": batch_timeqa[3],
                        "is_impossible": batch_timeqa[4],
                        "output_hidden_states" : True
                    }

                    inputs_synth = {
                        "input_ids": batch_synth[0],
                        "attention_mask": batch_synth[1],
                        "start_positions": batch_synth[2],
                        "end_positions": batch_synth[3],
                        "is_impossible": batch_synth[4],
                        "output_hidden_states" : True
                    }

                    outputs_timeqa = model(**inputs_timeqa)
                    outputs_synth = model(**inputs_synth)
                    loss_timeqa = outputs_timeqa[0]
                    loss_synth = outputs_synth[0]

                    #Replace nan to 0
                    loss_timeqa = torch.nan_to_num(loss_timeqa)
                    loss_synth = torch.nan_to_num(loss_synth)
                    #2:1 weight
                    loss = loss_timeqa+cfg.k*loss_synth


                elif cfg.CRL:
                    batch_timeqa, batch_crl = batch
                    batch_timeqa = tuple(t.to(device) for t in batch_timeqa)
                    batch_crl = tuple(t.to(device) for t in batch_crl)

                    inputs_timeqa = {
                        "input_ids": batch_timeqa[0],
                        "attention_mask": batch_timeqa[1],
                        "start_positions": batch_timeqa[2],
                        "end_positions": batch_timeqa[3],
                        "is_impossible": batch_timeqa[4],
                        "output_hidden_states" : True
                    }


                    feature_q = batch_crl[0],
                    feature_c = batch_crl[1],
                    inputs_q = {
                        "input_ids": feature_q[0],
                        "output_hidden_states" : True
                    }
                    inputs_c = {
                        "input_ids": feature_c[0],
                        "output_hidden_states" : True
                    }
                    labels = batch_crl[2],

                    outputs_timeqa = model(**inputs_timeqa)
                    loss_timeqa = outputs_timeqa[0]
                    output_c, output_q = model.forward_crl(inputs_c, inputs_q)

                    loss_crl, n, p = crl_loss(output_c, output_q, labels)
                    #Replace nan to 0
                    loss_timeqa = torch.nan_to_num(loss_timeqa)
                    loss_crl = torch.nan_to_num(loss_crl)
                    #2:1 weight
                    loss = loss_timeqa+cfg.k_crl*loss_crl

                # elif cfg.TCSE and cfg.crl:

                else:
                    batch = tuple(t.to(device) for t in batch)

                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "start_positions": batch[2],
                        "end_positions": batch[3],
                        "is_impossible": batch[4],
                        "output_hidden_states" : True
                    }

                    outputs = model(**inputs)

                    loss = outputs[0]
                    #Replace nan to 0
                    loss = torch.nan_to_num(loss)

                if cfg.n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / cfg.logging_steps, global_step)
                    logging_loss = tr_loss
               
            # Save Model to a new Directory
            output_dir = "checkpoint-epoch-{}".format(epoch)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            evaluation(model,tokenizer, cfg, output_dir, epoch, 'dev')
        
         
        tb_writer.close()

if __name__ == "__main__":
    main()
