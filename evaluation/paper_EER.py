from evaluation.EER_method import *
from deep_cnn import ConvNet_eval as f_enc
import torch
from render_data import RenderedInstrumentDataset
from tqdm.auto import tqdm
import wandb
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

def get_f_emb():
    batch_size = 1

    test_dataset = RenderedInstrumentDataset(split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    with torch.no_grad():

        # load f_enc
        model = f_enc(out_classes = 953).to(DEVICE)
        loaded_dict = torch.load('/disk2/aiproducer_inst/haessun_models/f_enc/f_enc_rendered_02/class953_epoch8_iter1200_trLoss_0.300_trAcc_89.031', map_location = DEVICE)
        loaded_dict = dict(list(loaded_dict.items())[:-2])
        model.load_state_dict(loaded_dict, strict=False)
        model.eval()
        _f_enc = model.forward

        output = torch.zeros(53000, 1024).to(DEVICE)
        labels = torch.zeros(53000, 1).to(DEVICE)

        """  test dataset  """ 
        for i, (input, inst_idx) in tqdm(enumerate(test_loader)):
            input = input.to(DEVICE)
            inst_idx = inst_idx.to(DEVICE)

            output[i] = _f_enc(input)
            labels[i] = inst_idx

        print(len(output), output[0].size())
        print(len(labels), labels[0].size())

        torch.save(output, "./valid_output/output.pt")
        torch.save(labels, "./valid_output/labels.pt")

    return output, labels


if __name__ == '__main__':
    DEVICE = torch.device('cuda:{}'.format(5)) if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))

    wandb.init(
        project = 'f_enc',
        name = "EER-code from paper",
    )

    # output, labels = get_f_emb()
    
    output = torch.load("./valid_output/output.pt")
    labels = torch.load("./valid_output/labels.pt")

    f = open("./eer_result.txt", 'w')

    eer_arr = []
    for seed in tqdm(range(30)):
        eer = compute_eer_det(output.cpu().numpy(), labels.cpu(), seed=seed)
        f.write("seed: {}, EER: {}\n".format(seed, eer))
        print(eer)

    f.close()
    print("EER: ", eer_arr)


