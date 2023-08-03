import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
from tqdm import tqdm

transform = T.ToPILImage()
import torch.optim as optim
import json
import numpy as np
from llava.model.llava import LlavaLlamaModel


TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
CONTEXT_LEN = 2048
# max_src_len = context_len - max_new_tokens - 8

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_model(MODEL_NAME):
    disable_torch_init()
    model_name = os.path.expanduser(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True
    ).cuda()
    return model, tokenizer


def load_param(MODEL_NAME, model, tokenizer, initial_query):
    model_name = os.path.expanduser(MODEL_NAME)

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.get_model().vision_tower[0]
    vision_tower = CLIPVisionModel.from_pretrained(
        vision_tower.config._name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.get_model().vision_tower[0] = vision_tower

    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    unnorm = UnNormalize(image_processor.image_mean, image_processor.image_std)
    embeds = model.model.embed_tokens.cuda()
    projector = model.model.mm_projector.cuda()

    for param in vision_tower.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    for param in projector.parameters():
        param.requires_grad = False

    for param in embeds.parameters():
        param.requires_grad = False

    for param in model.model.parameters():
        param.requires_grad = False

    qs = initial_query
    if mm_use_im_start_end:
        qs = (
            qs
            + "\n"
            + DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
        )
    else:
        qs = qs + "\n" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    return (
        tokenizer,
        image_processor,
        vision_tower,
        unnorm,
        embeds,
        projector,
        prompt,
        input_ids,
    )


def display_image(image_tensor, unnorm):
    return display(transform(unnorm(image_tensor.data[0].detach().cpu())))


def train_image_entire(
    input_ids, X, y, model, vision_tower, projector, epochs=100, lr=0.01
):
    pbar = tqdm(range(epochs))

    # Loss Function is Cross Entropy Loss
    crit = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([X], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4  # Maximum number of iterations.
    )  # Minimum learning rate.

    for i in pbar:
        loss_acc = []

        for j in range(y.shape[1]):
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()
            cur_image_idx = 0
            # Forward pass: calculate the loss
            new_input_embeds = []

            image_forward_out = vision_tower(X, output_hidden_states=True)
            select_hidden_state = image_forward_out.hidden_states[-2]

            image_features = select_hidden_state[:, 1:]
            image_features = projector(image_features)
            cur_image_features = image_features[0]

            # new input_ids
            cur_input_ids = torch.cat((input_ids, y[:, :j]), dim=1)[0]
            inputs_embeds = model.model.embed_tokens(cur_input_ids.unsqueeze(0))
            cur_input_embeds = inputs_embeds[0]
            num_patches = cur_image_features.shape[0]

            image_start_tokens = torch.where(cur_input_ids == 32001)[0]

            image_start_token_pos = image_start_tokens.item()
            cur_image_features = image_features[0].to(device=cur_input_embeds.device)
            cur_new_input_embeds = torch.cat(
                (
                    cur_input_embeds[: image_start_token_pos + 1],
                    cur_image_features,
                    cur_input_embeds[image_start_token_pos + 256 + 1 :],
                ),
                dim=0,
            )
            inputs_embeds = cur_new_input_embeds.unsqueeze(0)

            res = super(LlavaLlamaModel, model.model).forward(
                inputs_embeds=inputs_embeds,
            )

            res2 = model.lm_head(res.last_hidden_state)
            loss = crit(res2[0][-(j + 1) :], y[0, : j + 1])
            loss_acc.append(loss.item())
            res3 = torch.autograd.grad(outputs=loss, inputs=X)

            X = X - lr * res3[0].sign()

            # Decide how to clamp
            X = torch.clamp(X, min=-1.8, max=2.2)
            del res, res2, res3

        scheduler.step()
        pbar.set_postfix({"loss": np.mean(loss_acc), "lr": scheduler.get_last_lr()[0]})

    return X


def train_image_partial(
    input_ids, X, y, model, vision_tower, projector, epochs=100, lr=0.01, rows=10
):
    pbar = tqdm(range(epochs))

    # Loss Function is Cross Entropy Loss
    crit = torch.nn.CrossEntropyLoss()

    # modify part of X
    part_to_modify = X[0, :, 0:rows].clone().unsqueeze(0).detach().requires_grad_(True)
    remaining_part = X[0, :, rows:].unsqueeze(0)

    optimizer = optim.SGD([part_to_modify], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4  # Maximum number of iterations.
    )  # Minimum learning rate.

    for i in pbar:
        loss_acc = []
        for j in range(y.shape[1]):
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()
            cur_image_idx = 0
            # Forward pass: calculate the loss
            new_input_embeds = []

            modified_X = torch.cat((part_to_modify, remaining_part), dim=2)

            image_forward_out = vision_tower(modified_X, output_hidden_states=True)
            select_hidden_state = image_forward_out.hidden_states[-2]

            image_features = select_hidden_state[:, 1:]
            image_features = projector(image_features)
            cur_image_features = image_features[0]

            # new input_ids
            cur_input_ids = torch.cat((input_ids, y[:, :j]), dim=1)[0]
            inputs_embeds = model.model.embed_tokens(cur_input_ids.unsqueeze(0))
            cur_input_embeds = inputs_embeds[0]
            num_patches = cur_image_features.shape[0]

            image_start_tokens = torch.where(cur_input_ids == 32001)[0]

            image_start_token_pos = image_start_tokens.item()
            cur_image_features = image_features[0].to(device=cur_input_embeds.device)
            cur_new_input_embeds = torch.cat(
                (
                    cur_input_embeds[: image_start_token_pos + 1],
                    cur_image_features,
                    cur_input_embeds[image_start_token_pos + 256 + 1 :],
                ),
                dim=0,
            )
            inputs_embeds = cur_new_input_embeds.unsqueeze(0)

            res = super(LlavaLlamaModel, model.model).forward(
                inputs_embeds=inputs_embeds,
            )

            res2 = model.lm_head(res.last_hidden_state)
            loss = crit(res2[0][-(j + 1) :], y[0, : j + 1])
            loss_acc.append(loss.item())
            res3 = torch.autograd.grad(outputs=loss, inputs=part_to_modify)

            part_to_modify = part_to_modify - lr * res3[0].sign()

            # Decide how to clamp
            part_to_modify = torch.clamp(part_to_modify, min=-1.8, max=2.2)
            del res, res2, res3

            # pbar.set_postfix({'loss': loss.item(), 'word': j})
        scheduler.step()
        pbar.set_postfix({"loss": np.mean(loss_acc), "lr": scheduler.get_last_lr()[0]})

    return modified_X


def save_image(X, unnorm, name="test"):
    # save image to .jpg
    save_img_path = "result_images/llava/" + name + ".png"

    # save the image tensor to .pt
    save_pt_path = "result_images/llava/" + name + ".pt"

    transform(unnorm(X.data[0].detach().cpu())).save(save_img_path)
    torch.save(X, save_pt_path)


@torch.inference_mode()
def generate_stream(model, prompt, tokenizer, input_ids, images=None):
    temperature = TEMPERATURE
    max_new_tokens = MAX_NEW_TOKENS
    context_len = CONTEXT_LEN
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]
    stop_idx = 2

    ori_prompt = prompt
    image_args = {"images": images}

    output_ids = list(input_ids)
    pred_ids = []

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    past_key_values = None

    for i in range(max_new_tokens):
        if i == 0 and past_key_values is None:
            out = model(
                torch.as_tensor([input_ids]).cuda(),
                use_cache=True,
                output_hidden_states=True,
                **image_args,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device="cuda"
            )
            out = model(
                input_ids=torch.as_tensor([[token]], device="cuda"),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values
        # yield out

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)
        pred_ids.append(token)

        if stop_idx is not None and token == stop_idx:
            stopped = True
        elif token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i != 0 and i % 1024 == 0 or i == max_new_tokens - 1 or stopped:
            cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pos = -1  # cur_out.rfind(stop_str)
            if pos != -1:
                cur_out = cur_out[:pos]
                stopped = True
            output = ori_prompt + cur_out

            # print('output', output)

            ret = {
                "text": output,
                "error_code": 0,
            }
            yield cur_out

        if stopped:
            break

    if past_key_values is not None:
        del past_key_values


def run_result(X, prompt, initial_query, query_list, model, tokenizer, unnorm):
    # Display our perturbed image
    print("Image: ")
    display(transform(unnorm(X.data[0].detach().cpu())))

    # Generate the output with initial query
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    res = generate_stream(model, prompt, tokenizer, input_ids[0].tolist(), X)
    for response1 in res:
        outputs1 = response1

    print(f"Query 1:")
    print(initial_query)
    print(f"Response 1:")
    print(outputs1.strip())

    print("********")

    # Generate the outputs with further queries
    for idx, query in enumerate(query_list):
        if idx == 0:
            # Update current prompt with the initial prompt and first output
            new_prompt = prompt + outputs1 + "\n###Human: " + query + "\n###Assistant:"

        else:
            # Update current prompt with the previous prompt and latest output
            new_prompt = (
                new_prompt + outputs + "\n###Human: " + query + "\n###Assistant:"
            )

        input_ids = tokenizer.encode(new_prompt, return_tensors="pt").cuda()

        # Generate the response using the updated prompt
        res = generate_stream(model, new_prompt, tokenizer, input_ids[0].tolist(), X)
        for response in res:
            outputs = response

        # Print the current query and response
        print(f"Query {idx + 2}:")
        print(query)
        print(f"Response {idx + 2}:")
        print(outputs.strip())

        print("********")
