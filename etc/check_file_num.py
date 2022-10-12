import nntplib
import numpy as np
import glob
import scipy
import torch
from torch import nn
from deep_cnn import ConvNet_eval as f_enc
import librosa

DEVICE = torch.device('cuda:{}'.format(2)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

with torch.no_grad():
    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/data4/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward



# 내가 뽑은 single embedding - 무슨 기준으로 했는지 모름. 코드 없어짐..
single_2_embedding = np.load('/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/train_default_nfft/002/0001.npy')
single_2_embedding = torch.tensor(single_2_embedding).to(DEVICE)
single_10_embedding = np.load('/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/train_default_nfft/010/0001.npy')
single_10_embedding = torch.tensor(single_10_embedding).to(DEVICE)

# 1000개 평균낸 lib에서 가져와서 sim를 계산해야할듯.
# lib_2 = np.load('/home/haessun/ai_prod/train_inst_repr_lib_npy/002.npy')
# lib_2 = torch.tensor(lib_2).to(DEVICE).unsqueeze(0)
# lib_1 = np.load('/home/haessun/ai_prod/train_inst_repr_lib_npy/001.npy')
# lib_1 = torch.tensor(lib_1).to(DEVICE).unsqueeze(0)

# path = glob.glob('/data4/aiproducer_inst/rendered_multi_inst_3/train/*')
# for p in path:
#     npy_path = glob.glob(p + '/*.wav')
#     npy_path = [i.split('/')[-1] for i in npy_path]
#     if '2_0.wav' in npy_path:
#         print(p)
#         break

# multi에 저장되어있는 2_0.wav. 이걸 내가 훈련한 f에 통과 시켜보자.
single_2_wav_from_multi_3 = '/data4/aiproducer_inst/rendered_multi_inst_3/train/062237/2_0.wav'
sr, audio = scipy.io.wavfile.read(single_2_wav_from_multi_3)
audio = np.array(audio, dtype=np.float32) / 32768.0
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
log_spec = librosa.power_to_db(mel_spec)
log_spec = torch.tensor(log_spec, dtype=torch.float32).to(DEVICE)
newly_encoded_single_2 = _f_enc(log_spec.unsqueeze(0))



cos_loss = nn.CosineEmbeddingLoss()
target = torch.ones(1).to(DEVICE)

# # single의 10번이랑 multi에 있는 single중 2번이랑 맞는듯
# print(cos_loss(newly_encoded_single_2, single_2_embedding, target))
# print(cos_loss(newly_encoded_single_2, single_10_embedding, target))
# print(cos_loss(newly_encoded_single_2, lib_1, target))
# print(cos_loss(newly_encoded_single_2, lib_2, target))
# # tensor(0.8941, device='cuda:2', grad_fn=<MeanBackward0>)
# # tensor(0.6716, device='cuda:2', grad_fn=<MeanBackward0>)
# # tensor(0.8519, device='cuda:2', grad_fn=<MeanBackward0>)
# # tensor(0.0847, device='cuda:2', grad_fn=<MeanBackward0>)


# 내가 뽑은 single 2 emb이랑 rendered된 2를 다시 emb 뽑은거랑 비교 -> 이게 맞는듯
wav_path = '/data4/aiproducer_inst/rendered_single_inst/train/2/0001.wav'
sr, audio = scipy.io.wavfile.read(wav_path)
audio = np.array(audio, dtype=np.float32) / 32768.0
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
log_spec = librosa.power_to_db(mel_spec)
log_spec = torch.tensor(log_spec, dtype=torch.float32).to(DEVICE)
newly_encoded_single_2_from_rendered_single_inst = _f_enc(log_spec.unsqueeze(0))
print(cos_loss(newly_encoded_single_2_from_rendered_single_inst, single_2_embedding, target))

# # 내가 뽑은 single 2 emb이랑 rendered된 10를 다시 emb 뽑은거랑 비교 -> 이거 아님
# wav_path = '/data4/aiproducer_inst/rendered_single_inst/train/10/0001.wav'
# sr, audio = scipy.io.wavfile.read(wav_path)
# audio = np.array(audio, dtype=np.float32) / 32768.0
# mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
# log_spec = librosa.power_to_db(mel_spec)
# log_spec = torch.tensor(log_spec, dtype=torch.float32).to(DEVICE)
# newly_encoded_single_10_from_rendered_single_inst = _f_enc(log_spec.unsqueeze(0))
# print(cos_loss(newly_encoded_single_10_from_rendered_single_inst, single_2_embedding, target))


# rendered된 2를 다시 emb 뽑은거랑 single_f_emb_npy/train/2 를 비교  -> 같음
single_f_emb_2 = np.load('/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/train/2/0001.npy')
print(cos_loss(newly_encoded_single_2_from_rendered_single_inst, single_2_embedding, target))
