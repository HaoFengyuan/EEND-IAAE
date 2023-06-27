import os
import torch
import librosa
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from libs.EEND import EEND
from scipy.ndimage import shift
from scipy.signal import medfilt
from libs.hparam import hparam as hp

EPSILON = 1e-10


class List2Tensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x.copy())
        elif type == 'int':
            return torch.IntTensor(x.copy())


def splice(Y, context_size=0):
    Y_pad = np.pad(Y, [(context_size, context_size), (0, 0)], 'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(Y_pad,
                                                (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
                                                (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format="%Y-%m-%d %H:%M:%S",
               file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler() if not file else logging.FileHandler(name, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    List2Tensor = List2Tensor()
    logger = get_logger(__name__)
    device = torch.device("cuda:{}".format(hp.gpu_id))

    model = EEND(n_speakers=hp.model.n_speakers,
                 in_size=hp.data.dimension,
                 n_heads=hp.model.n_heads,
                 n_units=hp.model.hidden_size,
                 n_encoder_layers=hp.model.n_encoder_layers,
                 dim_feedforward=hp.model.dim_feedforward).eval()
    model = torch.nn.DataParallel(model, (hp.gpu_id,))
    model = model.to(device)
    model.load_state_dict(torch.load(hp.model.model_path))
    params = sum([param.nelement() for param in model.parameters() if param.requires_grad]) / 10.0 ** 6
    logger.info("Loading model from {:s}, Param: {:.2f}M".format(hp.model.model_path, params))

    sys_scps, ref_scps = [], []
    mel_basis = librosa.filters.mel(sr=hp.data.sr,
                                    n_fft=int(hp.data.window * hp.data.sr),
                                    n_mels=hp.data.nmels)

    for i, file in enumerate(os.listdir(os.path.join(hp.infer.test_path, 'audio'))):
        if file.endswith('.wav'):
            wav, _ = sf.read(os.path.join(hp.infer.test_path, 'audio', file))
            feat_wav = librosa.core.stft(y=wav,
                                         n_fft=int(hp.data.window * hp.data.sr),
                                         win_length=int(hp.data.window * hp.data.sr),
                                         hop_length=int(hp.data.hop * hp.data.sr)).T[:-1]
            feat_wav = np.log10(np.dot(np.abs(feat_wav) ** 2, mel_basis.T) + EPSILON)
            feat_wav = splice(feat_wav - np.mean(feat_wav, axis=0), hp.data.context_size)
            feat_wav = feat_wav[::hp.data.subsampling]

            with torch.no_grad():
                y = [List2Tensor(feat_wav, 'float').to(device)]
                output = model(y, activation=torch.sigmoid)
                out_chunks = output[0].cpu().detach().numpy()

            if hp.infer.label_delay != 0:
                out_chunks = shift(out_chunks, (-hp.infer.label_delay, 0))

            # Smooth
            a = np.where(out_chunks[:] > hp.infer.threshold, 1, 0)
            if hp.infer.median > 1:
                a = medfilt(a, (hp.infer.median, 1))

            out_dir = os.path.join(hp.infer.test_path, 'infer',
                                   "thr{0}_median{1}".format(hp.infer.threshold, hp.infer.median))
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            rttm_name = '{:s}_pred.rttm'.format(file.split('.')[0])
            with open(os.path.join(out_dir, rttm_name), 'w') as rttm:
                for speaker_id, frames in enumerate(a.T):
                    frames = np.pad(frames, (1, 1), 'constant')
                    changes, = np.where(np.diff(frames, axis=0) != 0)

                    for start, end in zip(changes[::2], changes[1::2]):
                        lines = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s}" \
                                " <NA> <NA>\n".format(file.split('.')[0],
                                                      start * hp.data.hop * hp.data.subsampling,
                                                      (end - start) * hp.data.hop * hp.data.subsampling,
                                                      str(speaker_id))
                        rttm.write(lines)
            rttm.close()

            # Scps used to evaluate.
            sys_scps.append(os.path.join(out_dir, rttm_name) + '\n')
            ref_scps.append(os.path.join(hp.infer.test_path, 'rttm', '%s.rttm' % (file.split('.')[0])) + '\n')

        logger.info("{:d} sentences: Completed!".format(i + 1))

    with open(os.path.join(hp.infer.test_path, "sys.scp"), 'w') as sys_scp_file:
        sys_scp_file.writelines(sys_scps)

    with open(os.path.join(hp.infer.test_path, "ref.scp"), 'w') as ref_scp_file:
        ref_scp_file.writelines(ref_scps)
