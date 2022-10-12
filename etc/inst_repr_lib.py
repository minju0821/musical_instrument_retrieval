import glob
import torch
import scipy
import random
import librosa
import numpy as np
from tqdm.auto import tqdm
from deep_cnn import ConvNet_eval as f_enc

DEVICE = torch.device('cuda:{}'.format(14)) if torch.cuda.is_available else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

random.seed(0)

with torch.no_grad():

    # load f_enc
    model = f_enc(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/data4/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    path = "/data4/aiproducer_inst/rendered_single_inst/test/"
    
    tqdm_bar = tqdm(enumerate(range(1, 54)))

    for idx, inst_num in tqdm_bar:
        inst_path = path + str(inst_num)
        # random.shuffle(inst_path)
        inst_files = glob.glob(inst_path + "/*.wav")
        out = torch.zeros((1024)).to(DEVICE)
        for i, file in enumerate(inst_files):
            sr, audio = scipy.io.wavfile.read(file)
            audio = np.array(audio, dtype=np.float32) / 32768.0
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=1024, hop_length=512, n_mels=128)
            log_spec = librosa.power_to_db(mel_spec)
            log_spec = torch.tensor(log_spec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out += _f_enc(log_spec).squeeze()

        out = out / 1000.
        out = out.cpu().detach().numpy()

        np.save("/home/haessun/ai_prod/valid_inst_repr_lib_npy/{}.npy".format(inst_num), out)

        