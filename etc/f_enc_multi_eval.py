import wandb
import torch
from tqdm.auto import tqdm
from deep_cnn import ConvNet
from render_data import RenderedMixInstrumentDataset
import torchmetrics.functional as metrics

   
if __name__=='__main__':
    DEVICE = torch.device('cuda:{}'.format(5)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    # load f_enc
    model = ConvNet(out_classes = 953).to(DEVICE)
    loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_00/class953_epoch6_iter22200_trLoss_0.193_trAcc_91.490', map_location = DEVICE)
    loaded_dict = dict(list(loaded_dict.items())[:-2])
    model.load_state_dict(loaded_dict, strict=False)
    model.eval()
    _f_enc = model.forward

    wandb.init(
        project = "evaluation",
        name = "f_enc multi-class eval. : class={}, eer={}, trLoss={}, trAcc={}".format(953, 0.01604, 0.193, 91.490),
    )

    # eval_dataset = InstrumentDataset(split='test')
    eval_dataset = RenderedMixInstrumentDataset(split='test')
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False)

    with torch.no_grad():
        for idx, (inputs, label) in tqdm(enumerate(eval_loader)):
            inputs, label = inputs.to(DEVICE), label.to(DEVICE)

            out = _f_enc(inputs)
            micro_f1 = metrics.f1_score(preds=out, target=label, num_classes=953, threshold=0.5, average='micro')
            macro_f1 = metrics.f1_score(preds=out, target=label, num_classes=953, threshold=0.5, average='macro')

            wandb.log({
                "micro_f1" : micro_f1,
                "macro_f1" : macro_f1
            })
