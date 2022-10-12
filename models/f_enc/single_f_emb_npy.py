from glob import glob
import numpy as np
from tqdm.auto import tqdm
import scipy
import librosa
import torch
import os
from deep_cnn import ConvNet_eval as f_enc

if __name__ == "__main__":
    DEVICE = torch.device('cuda:{}'.format(11)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    split = 'train'
    path = '/data4/aiproducer_inst/rendered_single_inst'
    path = f'{path}/{split}'
    path_list = []
    
    if split == 'test':
        for i in range(1, 54):
            path_list.append(f'{path}/{i}')
    elif split == 'train':
        for i in range(1, 954):
            path_list.append(f'{path}/{i}')

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/data4/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    for i, path in tqdm(enumerate(path_list)):
        # path = /data4/aiproducer_inst/rendered_single_inst/train/1
        inst_num = str(i+1).zfill(3)
        file_list = glob(f'{path}/*')

        for fname in file_list:
            sr, audio = scipy.io.wavfile.read(fname)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            # mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128, n_fft=1024)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32).to(DEVICE)
            output = _f_enc(log_spec.unsqueeze(0))

            npy_name = str(fname.split('/')[-1][:-4]) + '.npy'

            # valid 폴더에 있는건 nfft=1024로 한데다가, 폴더 번호 다 틀림; sort를 하니까 1, 10, 11, ... , 9 이런식으로 되어버렸었다는...
            dir_to_save = f'/data4/aiproducer_inst/f_embeddings/f_haessun/single_f_emb_npy/{split}_default_nfft/{inst_num}'
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)

            np.save(f'{dir_to_save}/{npy_name}', output.cpu().detach().numpy())


        