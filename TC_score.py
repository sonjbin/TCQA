import json
import argparse

def calculate_tc_score(test_data, output):
    time_correct = 0
    context_correct = 0
    correct = 0
    cnt=0
    for i,out in enumerate(output):
        #ignore unanswerable
        if i%2==0:
            continue
        
        if len(out) == 2 or test_data[i]['idx_context'][2] == -1:
            continue
        cnt+=1
        predicted, _, _, _ = out
        if len(predicted.split())!=2:
#             print(predicted)
            continue
        while True:
            if predicted.lower() in test_data[i]['context'].lower():
                s = test_data[i]['context'].lower().index(predicted.lower())
                break
            predicted = predicted[:-1]
        idx_tc, idx_t, idx_c = test_data[i]['idx_context']
        
        if s in [k for k in range(idx_tc[0],idx_tc[1]+1)]:
            time_correct+=1
            context_correct+=1
            correct +=1
        elif s in [k for k in range(idx_t[0],idx_t[1]+1)]:
            time_correct+=1
        elif idx_c != -1 and s in [k for k in range(idx_c[0],idx_c[1]+1)]:
            context_correct+=1
    print(context_correct)
    time_acc = time_correct/cnt
    context_acc = context_correct/cnt
    score = 2*(time_acc*context_acc)/(time_acc+context_acc)
    print(f"Time accuracy: {round(time_acc,4)}, Context accuracy: {round(context_acc,4)}")
    print("***Time-awarness score is in [-1,1]***")
    print("***The closer to 1, the more time-overfitted, and the closer to -1, the more context overfitted***")
    print(f"TC-score: {round((score),4)}")

def main(args):
    path = './dataset/synth_test/test.synthQA.json'
    test_data = []
    with open(path,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            test_data.append(dic)

    test_dict = {}
    for data in test_data:
        test_dict[data['idx']] = data

    output_path = args.predict_path

    with open(output_path,'r') as f:
        output = json.load(f)
    output = [output[key] for key in output]

    calculate_tc_score(test_data, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path',type=str)
    args = parser.parse_args()
    main(args)