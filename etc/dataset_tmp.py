class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes, audio_len):

        self.audio_len = audio_len
        self.num_classes = num_classes

        if audio_len == 5.:
            path = '/home/haessun/ai_prod/nsynth-inst-valid/*/*/*/'
        elif audio_len ==2.5:
            path = '/home/haessun/ai_prod/nsynth-inst-valid_quarter/*/*/*/'
        
        if num_classes == 53:
            self.inst_list = self.get_inst_list()
        elif num_classes == 11:
            self.inst_list = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

        self.audio_list = glob.glob(path + "*.wav")
        random.shuffle(self.audio_list)

        self.onehot_inst = {}

        for i, inst in enumerate(self.inst_list):
            self.onehot_inst[inst] = i

    def get_inst_list(self):
        f = open("/home/haessun/ai_prod/valid_inst_list.txt", 'r')
        lines = f.readlines()

        inst_list = []
        for l in lines:
            inst_list.append(l[:-1])

        return inst_list
   
    def preprocess(self, wav, sr):

        assert len(wav) == int(22050 * self.audio_len), '{} , {}'.format(len(wav), int(22050 * self.audio_len))

        # 3. normalize by maximum value
        wav = wav / (np.amax(wav) + 1e-9)

        # 4~6. mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=22050, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)

        return log_spec

    def __len__(self):
        return len(self.audio_list)
        
    def __getitem__(self, index):
        wav, sr = librosa.load(self.audio_list[index], sr=22050)
        log_spec = self.preprocess(wav, sr)

        # "/home/haessun/ai_prod/nsynth-inst-valid/flute/synthetic/0/870_0.wav"
        inst = self.audio_list[index].split("/")[-4:-1]
        if self.num_classes == 53:
            inst = "{}/{}/{}".format(inst[0], inst[1], inst[2])
        elif self.num_classes == 11:
            inst = inst[0]

        label = self.onehot_inst[inst]
        
        return log_spec, label

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes, audio_len):

        self.audio_len = audio_len
        self.num_classes = num_classes

        if audio_len == 5.:
            path = '/home/haessun/ai_prod/nsynth-inst-test/*/*/*/'
        elif audio_len ==2.5:
            path = '/home/haessun/ai_prod/nsynth-inst-test_quarter/*/*/*/'
        
        if num_classes == 53:
            self.inst_list = self.get_inst_list()
        elif num_classes == 11:
            self.inst_list = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

        self.audio_list = glob.glob(path + "*.wav")
        random.shuffle(self.audio_list)

        self.onehot_inst = {}

        for i, inst in enumerate(self.inst_list):
            self.onehot_inst[inst] = i

    def get_inst_list(self):
        f = open("/home/haessun/ai_prod/test_inst_list.txt", 'r')
        lines = f.readlines()

        inst_list = []
        for l in lines:
            inst_list.append(l[:-1])

        return inst_list
   
    def preprocess(self, wav, sr):

        assert len(wav) == int(22050 * self.audio_len)

        # 3. normalize by maximum value
        wav = wav / (np.amax(wav) + 1e-9)

        # 4~6. mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=22050, win_length=1024, hop_length=512, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec)

        return log_spec

    def __len__(self):
        return len(self.audio_list)
        
    def __getitem__(self, index):
        wav, sr = librosa.load(self.audio_list[index], sr=22050)
        log_spec = self.preprocess(wav, sr)

        # "/home/haessun/ai_prod/nsynth-inst-test/flute/synthetic/0/870_0.wav"
        inst = self.audio_list[index].split("/")[-4:-1]
        if self.num_classes == 53:
            inst = "{}/{}/{}".format(inst[0], inst[1], inst[2])
        elif self.num_classes == 11:
            inst = inst[0]

        label = self.onehot_inst[inst]
        
        return log_spec, label
