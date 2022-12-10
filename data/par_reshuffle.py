import json
import pandas as pd
import random


def make_adv_trainsets(adv_file,output_file,num_ex):
    adv_data = pd.read_json(adv_file,lines=True)
    new_df = pd.DataFrame(columns=adv_data.columns)
    num_rows = len(adv_data)
    samples = random.sample(list(range(num_rows)),num_ex)
    for sample in samples:
        data = adv_data.iloc[sample]
        # separate out the adversarial sentence from the original context
        context = data['context']
        if 'turk' not in data['id']:
            new_df = new_df.append(data,ignore_index=True)
            continue
        adv_sent_idx = context.strip().rindex('. ')+1
        adv_sent = context[adv_sent_idx:]
        og_context = context[:adv_sent_idx]
        sent_ends = [i for i in range(len(og_context)) if og_context[i] == '.']
        # move the adversarial sentence to a random sentence position
        # this is the paragraph reshuffling to avoid the model ignoring the adversarial sentence / always ignoring the last sentence
        new_idx = random.choice(sent_ends) + 1
        answer_idx = data['answers']['answer_start']
        old_answers = []
        for i in range(len(answer_idx)):
            old_answers.append(answer_idx[i])
            if answer_idx[i] > new_idx:
                answer_idx[i] += len(adv_sent)
        new_context = og_context[:new_idx] + adv_sent + og_context[new_idx:]
        data['context'] = new_context
        new_df = new_df.append(data,ignore_index=True)
    with open(output_file,'w') as file:
        for i in range(len(new_df)):
            json.dump(new_df.iloc[i].to_dict(),file,separators=(',',':'))
            file.write('\n')
