import shap
import torch
from LXMERT.utils import Config
import LXMERT.utils
import os, copy, sys
import math, json
import random
import numpy as np
from tqdm import tqdm

from LXMERT.processing_image import Preprocess
from LXMERT.modeling_frcnn import GeneralizedRCNN
from transformers import LxmertTokenizer, LxmertForPreTraining, LxmertForQuestionAnswering

from read_datasets import read_data

import matplotlib.pyplot as plt

num_samples = sys.argv[1] # "all" or number
if num_samples != "all":
    num_samples = int(num_samples)
write_res = sys.argv[2] # "yes" or "no"
task = "image_sentence_alignment"  # image_sentence_alignment, vqa, gqa
# just for info for the script to run. This is not a param.
other_tasks_than_valse = ['mscoco', 'vqa', 'gqa', 'gqa_balanced', 'nlvr2']

DATA = {
    "foil_it": ["data/val2014/", 
                "data/foil-it.json"]
}



def custom_masker(mask, x):
    """
    Shap relevant function.
    It gets a mask from the shap library with truth values about which image and text tokens to mask (False) and which not (True).
    It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the image, but have only defined which image tokens to mask. The image tokens masking happens in get_model_prediction().
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them)
    masked_X[0, 0] = 101
    masked_X[0, nb_text_tokens-1] = 102
    return masked_X


def get_model_prediction(x):
    """
    Shap relevant function.
    1. Mask the image pixel according to the specified patches to mask from the custom masker.
    2. Predict the model output for all combinations of masked image and tokens. This is then further passed to the shap libary.
    """
    # split up the input_ids and the image_token_ids from x (containing both appended)

    # print(x)
    # print(f"x shape: {x.shape}")
    input_ids = torch.tensor(x[:, :inputs.input_ids.shape[1]]).cuda()
    masked_image_token_ids = torch.tensor(x[:, inputs.input_ids.shape[1]:]).cuda()

    # print(input_ids)
    # print(masked_image_token_ids)
    # print(f"input_ids.shape[0]: {input_ids.shape[0]}")
    # print(f"input_ids.shape[1]: {input_ids.shape[1]}")

    # select / mask features and normalized boxes from masked_image_token_ids
    result = np.zeros(input_ids.shape[0])

    # call the model for each "new image" generated with masked features
    for i in range(input_ids.shape[0]):
        # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
        masked_images = copy.deepcopy(images).cuda() # do I need deepcopy?
        
        # pathify the image
        # torch.Size([1, 3, 800, 1067]) LXMERT has variable images sizes, could be 800, 1202 too
        for k in range(masked_image_token_ids[i].shape[0]):
            if masked_image_token_ids[i][k] == 0:  # should be zero
                m = k // p  # 384 (img shape) / 16 (patch size)
                n = k % p
                masked_images[:, :, m * patch_size_row:(m+1)*patch_size_row, n*patch_size_col:(
                    n+1)*patch_size_col] = 0 # torch.rand(3, patch_size_row, patch_size_col) # np.random.rand()
        
        masked_image_output_dict = frcnn(
            masked_images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt"
        )

        # Very important that the boxes are normalized
        masked_image_normalized_boxes = masked_image_output_dict.get("normalized_boxes")
        masked_image_features = masked_image_output_dict.get("roi_features")
        # print(masked_image_features)

        output_lxmert = model(
            input_ids=input_ids[i].unsqueeze(0).cuda(),
            attention_mask=inputs.attention_mask.cuda(),
            visual_feats=masked_image_features.cuda(),
            visual_pos=masked_image_normalized_boxes.cuda(),
            token_type_ids=inputs.token_type_ids.cuda(),
            return_dict=True,
            output_attentions=False,
        )
        m = torch.nn.Softmax(dim=1)
        if task in ['vqa', 'gqa']:
            output = m(output_lxmert['question_answering_score'])[0]
            output = output[model_prediction]
        else:
            output = m(output_lxmert['cross_relationship_score']).detach()[
                :, 1]
        result[i] = output
    return result

def load_models(task):
    """ Load models and model components. """
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn_cfg.MODEL.DEVICE = "cuda"
    frcnn = GeneralizedRCNN.from_pretrained(
        "unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg).cuda()
    image_preprocess = Preprocess(frcnn_cfg)

    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

    if task == "image_sentence_alignment":
        model = LxmertForPreTraining.from_pretrained(
            "unc-nlp/lxmert-base-uncased").cuda()
        answers = None
    elif task == "vqa":
        model = LxmertForQuestionAnswering.from_pretrained(
            "unc-nlp/lxmert-vqa-uncased").cuda()
        VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
        answers = utils.get_data(VQA_URL)    
    elif task == "gqa":
        model = LxmertForQuestionAnswering.from_pretrained(
            "unc-nlp/lxmert-gqa-uncased").cuda()
        GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
        answers = utils.get_data(GQA_URL)
    else:
        raise NotImplementedError
    return frcnn_cfg, frcnn, image_preprocess, tokenizer, model, answers


frcnn_cfg, frcnn, image_preprocess, tokenizer, model, answers = load_models(task)

for instrument, foil_info in DATA.items():
    images_path = foil_info[0]
    foils_path = foil_info[1]

    foils_data = read_data(instrument, foils_path, images_path)
    # subsample the data (for faster estimates), to test code for a few samples
    random.seed(1520)
    if num_samples != "all":
        foils_data = dict(random.sample(foils_data.items(), num_samples))  # 100

    for foil_id, foil in tqdm(foils_data.items()):  # tqdm
        if instrument not in other_tasks_than_valse:
            caption_fits = foil['mturk']['caption']
        else:
            # pretend like the sample was accepted by annotators (for everything other than VALSE)
            caption_fits = 3
        if caption_fits >= 2:  # MTURK filtering! Use only valid set

            test_img_path = os.path.join(images_path, foil["image_file"])

            # work with one sentence at a time to avoid attention mask and image features confusions.
            if instrument not in other_tasks_than_valse:
                if instrument == 'plurals':
                    test_sentences = [foil["caption"][0], foil["foil"][0]]
                else:
                    test_sentences = [foil["caption"], foil["foil"][0]]
            elif instrument == 'mscoco':
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [foil["caption"], confounder["caption"]]
            elif task in ['vqa', 'gqa']:
                test_sentences = [foil["caption"]]
            else:
                confounder = random.sample(foils_data.items(), 1)[0][1]
                test_sentences = [f'{foil["caption"]} {foil["answer"]}.', f'{confounder["caption"]} {confounder["answer"]}.']


            # run frcnn
            # only nlvr2 is allowed exceptions (images downloaded from urls, sometimes bad)
            if instrument != "nlvr2":
                images, sizes, scales_yx = image_preprocess(test_img_path)
            else:
                try:
                    images, sizes, scales_yx = image_preprocess(test_img_path)
                except:
                    continue

            output_dict = frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=frcnn_cfg.max_detections,
                return_tensors="pt"
            )

            # Very important that the boxes are normalized
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")
            
            print("features: ")
            print(features)
            print(f"features' size: {features.size()}")

            print("images:")
            print(images)
            print(f"images' size: {images.size()}")
            # run lxmert
            # shap values need one sentence for transformer
            for k, sentence in enumerate(test_sentences):
                inputs = tokenizer(
                    sentence,
                    padding=False,
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                print(sentence)
                print(inputs)
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                print(tokens)
                # get model prediction
                output_lxmert = model(
                    input_ids=inputs.input_ids.cuda(),
                    attention_mask=inputs.attention_mask.cuda(),
                    visual_feats=features.cuda(),
                    visual_pos=normalized_boxes.cuda(),
                    token_type_ids=inputs.token_type_ids.cuda(),
                    return_dict=True,
                    output_attentions=False,
                )

                m = torch.nn.Softmax(dim=1)
                if task in ['vqa', 'gqa']:
                    output = m(output_lxmert['question_answering_score'])
                    model_prediction = output.argmax(-1)
                else:
                    model_prediction = m(
                        output_lxmert['cross_relationship_score']).cpu().detach()[:, 1].item()

                # determine text length in number of tokens (after tokenization)
                nb_text_tokens = np.count_nonzero(inputs.input_ids) # number of text tokens
                
                p = int(math.ceil(np.sqrt(nb_text_tokens)))
                p = 6
                patch_size_row = images.shape[2] // p # we have 36 image token ids
                patch_size_col = images.shape[3] // p # we have nb_text_tokens-1 image token ids
                print(f"nb_text_tokens={nb_text_tokens}")
                print(f"p={p}")
                print(f"patch_size_row={patch_size_row}")
                print(f"patch_size_col={patch_size_col}")
                # features.shape[1] = 36 as we have 36 image regions
                image_token_ids = torch.tensor(range(1, p**2+1)).unsqueeze(0)
                X = torch.cat((inputs.input_ids, image_token_ids), 1).unsqueeze(1)

                print("X values: " )
                print(X)
                print(X.shape)
                # get_model_prediction(X)
                # create an explainer with model and image masker
                explainer = shap.Explainer(
                    get_model_prediction, custom_masker, silent=True)
                # explainer = shap.Explainer(
                #     get_model_prediction, custom_masker)
                shap_values = explainer(X)
                
                print(shap_values)
                shap_values.values = shap_values.values[0][0]
                shap_values.data = shap_values.data[0][0]
                data = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                for i in range(len(shap_values.data)): 
                    if i < nb_text_tokens:
                        continue
                    else: 
                        data.append(f"Patch {shap_values.data[i]}")
                shap_values.data = data    
                print(shap_values)
                print(f"shap_values shape: {shap_values.values.shape}")
                # shap.plots.bar(shap_values, max_display = 50 ,show=False)
                # shap.plots.waterfall(shap_values[0], max_display = 50, show =False)
                shap.plots.force(shap_values, show=False)
                plt.savefig(f"shap_force_{k}.png")
                # plt.savefig(f"shap_waterfall_{k}.png")
            
    # writing results down to a json file for further analysis of results on VALSE
    if write_res == 'yes':
        path = f"result_jsons/lxmert_{num_samples}/"
        os.makedirs(path, exist_ok=True)
        with open(f'result_jsons/lxmert_{num_samples}/{instrument}.json', 'w') as f:
            json.dump(foils_data, f)
