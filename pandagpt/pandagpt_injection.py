import argparse
import torch
import os

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as T_image
from tqdm.notebook import tqdm

transform_image = T_image.ToPILImage()
import torch.optim as optim
import json
import numpy as np
import torchvision

import header
import gc
from pandagpt.code.model.openllama import OpenLLAMAPEFTModel
from pandagpt.code.model.ImageBind.data import (
    load_and_transform_vision_data,
    load_and_transform_audio_data,
)

import torchaudio
import torchaudio.transforms as T_audio
from IPython.display import Audio


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


unnorm = UnNormalize(
    mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
)


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Set parameter
TOP_P = 0.01
TEMPERATURE = 0.7
MAX_LENGTH = 1024


bos_embeds = None
p_before_embeds = None
p_after_embeds = None


# Parameters
num_mel_bins = 128
num_frames = 204
sample_rate = 16000
n_fft = 400
hop_length = n_fft // 4
win_length = n_fft


# Function to create a Mel inversion matrix
def create_mel_inversion_matrix(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    # Create a Mel filter bank using torchaudio
    mel_fb = T_audio.MelScale(
        n_mels, sr, f_min=fmin, f_max=fmax, n_stft=n_fft // 2 + 1, norm=None
    )

    # Convert the filter bank to a tensor
    mel_fb_tensor = torch.tensor(mel_fb.fb, dtype=torch.float)

    # Calculate the pseudo inverse
    inversion_matrix = torch.pinverse(mel_fb_tensor)

    return inversion_matrix


def inverse_it(mel_spectrogram):
    # Create the Mel inversion matrix
    inversion_matrix = create_mel_inversion_matrix(sample_rate, n_fft, num_mel_bins)

    # Invert the Mel spectrogram to a power spectrogram
    power_spectrogram = torch.matmul(mel_spectrogram, inversion_matrix)

    # Create an InverseMelScale transform
    inverse_mel_scale_transform = T_audio.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=num_mel_bins,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=sample_rate // 2,
        norm=None,
    )

    # Apply the InverseMelScale transform to the Mel spectrogram
    spectrogram = inverse_mel_scale_transform(mel_spectrogram.T)

    # Initialize Griffin-Lim transform
    griffin_lim = T_audio.GriffinLim(
        n_fft=n_fft, n_iter=32, win_length=win_length, hop_length=hop_length
    )

    # Recover the waveform from the spectrogram
    recovered_waveform = griffin_lim(spectrogram)

    Audio(recovered_waveform, rate=16000)

    return recovered_waveform


def inverse_normalize(melspec, mean=-4.268, std=9.138):
    return melspec * std + mean


def combine_results(audio):
    results = list()
    for i in range(audio.shape[1]):
        res = inverse_it(
            inverse_normalize(audio.clone().detach().cpu().float()[0][i][0]).T
        )
        results.append(res)
    return [results[0][:-2000], results[1][:-2000], results[2][:-2000]]


def load_model(**args):
    model = OpenLLAMAPEFTModel(**args)
    delta_ckpt = torch.load(
        args["delta_ckpt_path"]
    )  # , map_location=torch.device('cuda'))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.half()
    model.device = "cuda"
    model.to("cuda")

    for param in model.parameters():
        param.requires_grad_ = False
    for param in model.visual_encoder.parameters():
        param.requires_grad_ = False
    for param in model.llama_model.parameters():
        param.requires_grad_ = False

    return model


def load_image(image_path):
    image_tensor = load_and_transform_vision_data([image_path], "cuda").half()
    image_tensor.to("cuda")
    return image_tensor


def load_prompt(init_query, model, image=None, audio=None):
    global p_before_embeds, p_after_embeds, bos_embeds

    if image is not None:
        image_embed = model.visual_encoder({"vision": image})["vision"]
        p_before = "### Human: <Img>"
        text = "</Img> " + init_query + "\n### Assistant: "

    if audio is not None:
        image_embed = model.visual_encoder({"audio": audio})["audio"]
        p_before = "### Human: <Audio>"
        text = "</Audio> " + init_query + "\n### Assistant: "

    image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)

    # p_before = '### Human: <Img>'
    p_before_tokens = model.llama_tokenizer(
        p_before, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    p_before_embeds = model.llama_model.model.model.embed_tokens(
        p_before_tokens.input_ids
    ).expand(1, -1, -1)

    # text = '</Img> ' + init_query + '\n### Assistant: '
    p_after_tokens = model.llama_tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    p_after_embeds = model.llama_model.model.model.embed_tokens(
        p_after_tokens.input_ids
    ).expand(1, -1, -1)

    bos = (
        torch.ones(
            [1, 1],
            dtype=p_before_tokens.input_ids.dtype,
            device=p_before_tokens.input_ids.device,
        )
        * model.llama_tokenizer.bos_token_id
    )
    bos_embeds = model.llama_model.model.model.embed_tokens(bos)


def load_audio(audio_path):
    audio_tensor = load_and_transform_audio_data([audio_path], "cuda").half()
    audio_tensor.to("cuda")
    return audio_tensor


# def load_audio_prompt(init_query, image_tensor, model):
#     image_embed = model.visual_encoder({'vision': image_tensor})['vision']
#     image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)

#     p_before = '### Human: <Img>'
#     p_before_tokens = model.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(model.device)

#     global p_before_embeds, p_after_embeds, bos_embeds

#     p_before_embeds = model.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(1, -1, -1)

#     text = '</Img> ' + init_query + '\n### Assistant: '
#     p_after_tokens = model.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(model.device)
#     p_after_embeds = model.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(1, -1, -1)

#     bos = torch.ones([1, 1],
#                      dtype=p_before_tokens.input_ids.dtype,
#                      device=p_before_tokens.input_ids.device) * model.llama_tokenizer.bos_token_id
#     bos_embeds = model.llama_model.model.model.embed_tokens(bos)


def create_y_embed(model, y_text):
    y_tokens = (
        model.llama_tokenizer(y_text, add_special_tokens=False, return_tensors="pt")
        .to(model.device)
        .input_ids
    )
    y_embeds = model.llama_model.model.model.embed_tokens(y_tokens).expand(1, -1, -1)
    return y_embeds


def train_image_entire(X, y_text, model, epochs=200, lr=0.01):
    pbar = tqdm(range(epochs))

    # Loss Function is Cross Entropy Loss
    crit = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([X], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5  # Maximum number of iterations.
    )  # Minimum learning rate.

    y_tokens = (
        model.llama_tokenizer(y_text, add_special_tokens=False, return_tensors="pt")
        .to(model.device)
        .input_ids
    )
    y_embeds = model.llama_model.model.model.embed_tokens(y_tokens).expand(1, -1, -1)

    for i in pbar:
        loss_acc = []
        for j in range(y_tokens.shape[1]):
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

            image_embed = model.visual_encoder({"vision": X})["vision"]
            image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)
            inputs_embeds = torch.cat(
                [
                    bos_embeds,
                    p_before_embeds,
                    image_llama_embed,
                    p_after_embeds,
                    y_embeds[:, :j],
                ],
                dim=1,
            )
            # attention_mask=torch.ones_like(inputs_embeds[:,:,0]).long()
            # attention_mask[:, -(j+1):] = 0
            # labels = torch.cat([ bos[0], p_before_tokens.input_ids[0], unk_token[0], p_after_tokens.input_ids[0], y_tokens[0][:j+1]]).unsqueeze(0)

            res = model.llama_model.forward(
                inputs_embeds=inputs_embeds,
                # attention_mask=attention_mask,
                attention_mask=None,
                return_dict=True,
                labels=None,
            )

            loss = crit(res.logits[0, -(j + 1) :], y_tokens[0, : j + 1])

            loss_acc.append(loss.item())
            res3 = torch.autograd.grad(outputs=loss, inputs=X)
            X = X - lr * res3[0].sign()
            X = torch.clamp(X, min=-1.8, max=2.2)
            # X = torch.clamp(X, min=-1, max=1)
            del res, res3
            # break
        # break
        scheduler.step()
        pbar.set_postfix({"loss": np.mean(loss_acc), "lr": scheduler.get_last_lr()[0]})

    return X


def train_image_partial(X, y_text, model, epochs=200, lr=0.01, rows=20):
    pbar = tqdm(range(epochs))

    # Loss Function is Cross Entropy Loss
    crit = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([X], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5  # Maximum number of iterations.
    )  # Minimum learning rate.

    # modify part of X
    part_to_modify = X[0, :, 0:rows].clone().unsqueeze(0).detach().requires_grad_(True)
    remaining_part = X[0, :, rows:].unsqueeze(0)

    y_tokens = (
        model.llama_tokenizer(y_text, add_special_tokens=False, return_tensors="pt")
        .to(model.device)
        .input_ids
    )
    y_embeds = model.llama_model.model.model.embed_tokens(y_tokens).expand(1, -1, -1)

    for i in pbar:
        loss_acc = []
        for j in range(y_tokens.shape[1]):
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

            modified_X = torch.cat((part_to_modify, remaining_part), dim=2)

            image_embed = model.visual_encoder({"vision": modified_X})["vision"]
            image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)
            inputs_embeds = torch.cat(
                [
                    bos_embeds,
                    p_before_embeds,
                    image_llama_embed,
                    p_after_embeds,
                    y_embeds[:, :j],
                ],
                dim=1,
            )  # bsz x (1+s1+1+s2) x embed_dim
            # attention_mask=torch.ones_like(inputs_embeds[:,:,0]).long()
            # attention_mask[:, -(j+1):] = 0
            # labels = torch.cat([ bos[0], p_before_tokens.input_ids[0], unk_token[0], p_after_tokens.input_ids[0], y_tokens[0][:j+1]]).unsqueeze(0)

            res = model.llama_model.forward(
                inputs_embeds=inputs_embeds,
                # attention_mask=attention_mask,
                attention_mask=None,
                return_dict=True,
                labels=None,
            )

            loss = crit(res.logits[0, -(j + 1) :], y_tokens[0, : j + 1])

            loss_acc.append(loss.item())
            res3 = torch.autograd.grad(outputs=loss, inputs=part_to_modify)

            part_to_modify = part_to_modify - lr * res3[0].sign()
            part_to_modify = torch.clamp(part_to_modify, min=-1.8, max=2.2)
            # X = torch.clamp(X, min=-1, max=1)
            del res, res3
            # break
        # break
        scheduler.step()
        pbar.set_postfix({"loss": np.mean(loss_acc), "lr": scheduler.get_last_lr()[0]})

    return modified_X


def train_audio_entire(X, y_text, model, epochs=200, lr=0.01):
    pbar = tqdm(range(epochs))

    # Loss Function is Cross Entropy Loss
    crit = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([X], lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5  # Maximum number of iterations.
    )  # Minimum learning rate.

    y_tokens = (
        model.llama_tokenizer(y_text, add_special_tokens=False, return_tensors="pt")
        .to(model.device)
        .input_ids
    )
    y_embeds = model.llama_model.model.model.embed_tokens(y_tokens).expand(1, -1, -1)

    for i in pbar:
        loss_acc = []
        for j in range(y_tokens.shape[1]):
            lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

            image_embed = model.visual_encoder({"audio": X})["audio"]
            image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)
            inputs_embeds = torch.cat(
                [
                    bos_embeds,
                    p_before_embeds,
                    image_llama_embed,
                    p_after_embeds,
                    y_embeds[:, :j],
                ],
                dim=1,
            )
            # attention_mask=torch.ones_like(inputs_embeds[:,:,0]).long()
            # attention_mask[:, -(j+1):] = 0
            # labels = torch.cat([ bos[0], p_before_tokens.input_ids[0], unk_token[0], p_after_tokens.input_ids[0], y_tokens[0][:j+1]]).unsqueeze(0)

            res = model.llama_model.forward(
                inputs_embeds=inputs_embeds,
                # attention_mask=attention_mask,
                attention_mask=None,
                return_dict=True,
                labels=None,
            )

            loss = crit(res.logits[0, -(j + 1) :], y_tokens[0, : j + 1])

            loss_acc.append(loss.item())
            res3 = torch.autograd.grad(outputs=loss, inputs=X)
            X = X - lr * res3[0].sign()
            X = torch.clamp(X, min=-1.8, max=2.2)
            # X = torch.clamp(X, min=-1, max=1)
            del res, res3
            # break
        # break
        scheduler.step()
        pbar.set_postfix({"loss": np.mean(loss_acc), "lr": scheduler.get_last_lr()[0]})

    return X


def save_image(X, name="test"):
    ## save image to .png
    # save_img_path = "result_images/pandagpt/" + name + ".png"
    # torchvision.utils.save_image(unnorm(X.data[0].detach().cpu()), save_img_path)

    # save the image tensor to .pt
    save_pt_path = "result_images/pandagpt/" + name + ".pt"
    torch.save(X, save_pt_path)


def save_audio(X, name="test"):
    # save path of the audio tensor to .wav
    save_wav_path = "result_audios/" + name + ".wav"
    results = combine_results(X)

    # save path of the audio tensor to .pt
    save_pt_path = "result_audios/" + name + ".pt"

    # Save the audio data to the output file as WAV
    torchaudio.save(save_wav_path, torch.cat(results).unsqueeze(0), sample_rate=16000)
    # Save the audio data to the output file as pt
    torch.save(X, save_pt_path)


def display_image(image_tensor):
    return display(transform_image(unnorm(image_tensor.data[0].detach().cpu())))


def display_audio(audio_tensor):
    results = combine_results(audio_tensor)

    return display(Audio(torch.cat(results), rate=16000))


@torch.inference_mode()
def generate_stream(model, input_texts, image=None, audio=None, max_new_tokens=512):
    stop_idx = 2277
    unk_token = torch.as_tensor([[0]], device="cuda")

    temperature = TEMPERATURE
    max_length = MAX_LENGTH

    p_before = "### Human: <Img>"
    if audio is not None:
        p_before = "### Human: <Audio>"
    p_before_tokens = model.llama_tokenizer(
        p_before, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    p_before_embeds = model.llama_model.model.model.embed_tokens(
        p_before_tokens.input_ids
    ).expand(
        1, -1, -1
    )  # bsz x s1 x embed_dim

    p_after_tokens = model.llama_tokenizer(
        input_texts, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    p_after_embeds = model.llama_model.model.model.embed_tokens(
        p_after_tokens.input_ids
    ).expand(
        1, -1, -1
    )  # bsz x s1 x embed_dim

    bos = (
        torch.ones(
            [1, 1],
            dtype=p_before_tokens.input_ids.dtype,
            device=p_before_tokens.input_ids.device,
        )
        * model.llama_tokenizer.bos_token_id
    )  # bsz x 1
    bos_embeds = model.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim

    if image is not None:
        image_embed = model.visual_encoder({"vision": image})["vision"]
    if audio is not None:
        image_embed = model.visual_encoder({"audio": audio})["audio"]
    image_llama_embed = model.llama_proj(image_embed).unsqueeze(1)

    inputs_embeds = torch.cat(
        [bos_embeds, p_before_embeds, image_llama_embed, p_after_embeds], dim=1
    )  # bsz x (1+s1+1+s2) x embed_dim
    # print('inputs_embeds', inputs_embeds.shape)

    past_key_values = None
    output_ids = (
        torch.cat(
            [
                bos[0],
                torch.cat(
                    [
                        p_before_tokens.input_ids[0],
                        unk_token[0],
                        p_after_tokens.input_ids[0],
                    ]
                ),
            ]
        )
        .cpu()
        .tolist()
    )
    pred_ids = []
    ori_prompt = []

    for i in range(max_new_tokens):
        if i == 0 and past_key_values is None:
            out = model.llama_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(inputs_embeds[:, :, 0]).long(),
                return_dict=True,
                output_hidden_states=True,
                labels=None,
            )

            logits = out.logits
            past_key_values = out.past_key_values
            # print(inputs_embeds.shape)
            # yield inputs_embeds, logits, past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device="cuda"
            )

            out = model.llama_model.forward(
                input_ids=torch.as_tensor([[token]], device="cuda"),
                attention_mask=attention_mask,
                return_dict=True,
                labels=None,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
        # print(i, model.llama_tokenizer.decode([token]), token)
        # yield (past_key_values, logits, out.last_hidden_states)
        output_ids.append(token)
        pred_ids.append(token)

        if stop_idx is not None and token == stop_idx:
            stopped = True
        elif token == 2:
            stopped = True
        else:
            stopped = False

        if i != 0 and i % 1024 == 0 or i == max_new_tokens - 1 or stopped:
            cur_out = model.llama_tokenizer.decode(
                pred_ids[:-2], skip_special_tokens=False
            )
            # print(cur_out)
            pos = -1  # cur_out.rfind(stop_str)
            if pos != -1:
                cur_out = cur_out[:pos]
                stopped = True
            output = cur_out

            ret = {
                "text": output,
                "error_code": 0,
            }
            yield cur_out

        if stopped:
            break

    if past_key_values is not None:
        del past_key_values


def run_image_result(X, query_list, model):
    gc.collect()
    torch.cuda.empty_cache()

    # Display our perturbed image
    print("Image: ")
    display_image(X)

    for idx, query in enumerate(query_list):
        if idx == 0:
            text = "</Img> " + query + "\n### Assistant:"
        else:
            text += outputs + "\n### Human: " + query + "\n### Assistant:"

        res = generate_stream(model, text, image=X)
        for response in res:
            outputs = response

        print(f"Query {idx+1}:")
        print(query)
        print(f"Response {idx+1}:")
        print(outputs)

        print("********")


def run_audio_result(X, query_list, model):
    gc.collect()
    torch.cuda.empty_cache()

    # Display our perturbed audio
    print("Audio: ")
    display_audio(X)

    for idx, query in enumerate(query_list):
        if idx == 0:
            text = "</Audio> " + query + "\n### Assistant:"
        else:
            text += outputs + "\n### Human: " + query + "\n### Assistant:"

        res = generate_stream(model, text, audio=X)
        for response in res:
            outputs = response

        print(f"Query {idx+1}:")
        print(query)
        print(f"Response {idx+1}:")
        print(outputs)

        print("********")
