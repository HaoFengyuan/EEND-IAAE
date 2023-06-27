import os
import re
import sys
import shutil
import tempfile
import subprocess
import numpy as np
from tabulate import tabulate
from collections import namedtuple
from scorelib.six import iterkeys
from scorelib.uem import gen_uem, write_uem
from scorelib.utils import error, info, warn
from scorelib.rttm import write_rttm, load_rttm
from scorelib.turn import merge_turns, trim_turns

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
MDEVAL_BIN = os.path.join(SCRIPT_DIR, 'scorelib/md-eval-22.pl')
FILE_REO = re.compile(r'(?<=Speaker Diarization for).+(?=\*\*\*)')
SCORED_SPEAKER_REO = re.compile(r'(?<=SCORED SPEAKER TIME =)[\d.]+')
MISS_SPEAKER_REO = re.compile(r'(?<=MISSED SPEAKER TIME =)[\d.]+')
FA_SPEAKER_REO = re.compile(r'(?<=FALARM SPEAKER TIME =)[\d.]+')
ERROR_SPEAKER_REO = re.compile(r'(?<=SPEAKER ERROR TIME =)[\d.]+')


class Scores(namedtuple('Scores', ['file_id', 'der', 'miss', 'false', 'confusion'])):
    __slots__ = ()


def rectify(arr):
    # Numerator and denominator both 0.
    arr[np.isnan(arr)] = 0

    # Numerator > 0, but denominator = 0.
    arr[np.isinf(arr)] = 1
    arr *= 100.0
    return arr


def load_script_file(fn):
    with open(fn, 'rb') as f:
        return [line.decode('utf-8').strip() for line in f]


def load_rttms(rttm_fns):
    turns = []
    file_ids = set()
    for rttm_fn in rttm_fns:
        if not os.path.exists(rttm_fn):
            error('Unable to open RTTM file: %s' % rttm_fn)
            sys.exit(1)
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns.extend(turns_)
            file_ids.update(file_ids_)
        except IOError as e:
            error('Invalid RTTM file: %s. %s' % (rttm_fn, e))
            sys.exit(1)
    return turns, file_ids


def check_for_empty_files(ref_turns, sys_turns, uem):
    """Warn on files in UEM without reference or speaker turns."""
    ref_file_ids = {turn.file_id for turn in ref_turns}
    sys_file_ids = {turn.file_id for turn in sys_turns}
    for file_id in sorted(iterkeys(uem)):
        if file_id not in ref_file_ids:
            warn('File "%s" missing in reference RTTMs.' % file_id)
        if file_id not in sys_file_ids:
            warn('File "%s" missing in system RTTMs.' % file_id)
    # TODO: Clarify below warnings; this indicates that there are no
    #       ELIGIBLE reference/system turns.
    if not ref_turns:
        warn('No reference speaker turns found within UEM scoring regions.')
    if not sys_turns:
        warn('No system speaker turns found within UEM scoring regions.')


def print_table(file_scores, global_scores, n_digits=2, table_format='simple'):
    col_names = ['File',
                 'DER',  # Diarization error rate (DER)
                 'MISS',  # Miss speech error rate
                 'FALSE',  # False alarm speech error rate
                 'CONFUSION',  # Speaker confusion error rate
                 ]
    rows = sorted(file_scores, key=lambda x: x.file_id)
    rows.append(global_scores._replace(file_id='*** OVERALL ***'))
    floatfmt = '.%df' % n_digits
    tbl = tabulate(
        rows, headers=col_names, floatfmt=floatfmt, tablefmt=table_format)

    print(tbl)


def DER(ref_turns, sys_turns, uem, collar=0.0, ignore_overlaps=False):
    tmp_dir = tempfile.mkdtemp()

    # Write RTTMs.
    ref_rttm_fn = os.path.join(tmp_dir, 'ref.rttm')
    write_rttm(ref_rttm_fn, ref_turns)
    sys_rttm_fn = os.path.join(tmp_dir, 'sys.rttm')
    write_rttm(sys_rttm_fn, sys_turns)

    # Write UEM.
    if uem is None:
        uem = gen_uem(ref_turns, sys_turns)
    uemf = os.path.join(tmp_dir, 'all.uem')
    write_uem(uemf, uem)

    # Actually score.
    try:
        cmd = ['perl',
               MDEVAL_BIN,
               '-af',
               '-r', ref_rttm_fn,
               '-s', sys_rttm_fn,
               '-c', str(collar),
               '-u', uemf]
        if ignore_overlaps:
            cmd.append('-1')
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        stdout = e.output
    finally:
        shutil.rmtree(tmp_dir)

    # Parse md-eval output to extract by-file and total scores.
    stdout = stdout.decode('gbk')
    file_ids = [m.strip() for m in FILE_REO.findall(stdout)]
    file_ids = [file_id[2:] if file_id.startswith('f=') else file_id for file_id in file_ids]
    scored_speaker_times = np.array([float(m) for m in SCORED_SPEAKER_REO.findall(stdout)])
    miss_speaker_times = np.array([float(m) for m in MISS_SPEAKER_REO.findall(stdout)])
    fa_speaker_times = np.array([float(m) for m in FA_SPEAKER_REO.findall(stdout)])
    error_speaker_times = np.array([float(m) for m in ERROR_SPEAKER_REO.findall(stdout)])
    with np.errstate(invalid='ignore', divide='ignore'):
        error_times = miss_speaker_times + fa_speaker_times + error_speaker_times
        ders = error_times / scored_speaker_times
        miss_erros = miss_speaker_times / scored_speaker_times
        fa_errors = fa_speaker_times / scored_speaker_times
        error_errors = error_speaker_times / scored_speaker_times

    ders = rectify(ders)
    miss_erros = rectify(miss_erros)
    fa_errors = rectify(fa_errors)
    error_errors = rectify(error_errors)

    # Reconcile with UEM, keeping in mind that in the edge case where no
    # reference turns are observed for a file, md-eval doesn't report results
    # for said file.
    file_to_der = dict(zip(file_ids, ders))
    file_to_miss = dict(zip(file_ids, miss_erros))
    file_to_fa = dict(zip(file_ids, fa_errors))
    file_to_error = dict(zip(file_ids, error_errors))

    return file_to_der, file_to_miss, file_to_fa, file_to_error


def score(ref_turns, sys_turns, uem, **kwargs):
    file_to_der, file_to_miss, file_to_fa, file_to_error = DER(ref_turns, sys_turns, uem, **kwargs)

    # Compute clustering metrics.
    def compute_metrics(fid, der, miss, false, confusion):
        return Scores(fid, der, miss, false, confusion)

    file_scores = []
    for file_id, der in file_to_der.items():
        if file_id != 'ALL':
            file_scores.append(
                compute_metrics(file_id, der, file_to_miss[file_id], file_to_fa[file_id], file_to_error[file_id]))
    global_scores = compute_metrics(
        '*** OVERALL ***', file_to_der['ALL'], file_to_miss['ALL'], file_to_fa['ALL'], file_to_error['ALL'])

    return file_scores, global_scores


if __name__ == "__main__":
    ref_rttm_fns = load_script_file('test_audio/ref.scp')
    sys_rttm_fns = load_script_file('test_audio/sys.scp')

    if not ref_rttm_fns:
        error('No reference RTTMs specified.')
        sys.exit(1)
    if not sys_rttm_fns:
        error('No system RTTMs specified.')
        sys.exit(1)

    # Load speaker/reference speaker turns and UEM. If no UEM specified,
    # determine it automatically.
    info('Loading speaker turns from reference RTTMs...', file=sys.stderr)
    ref_turns, _ = load_rttms(ref_rttm_fns)
    info('Loading speaker turns from system RTTMs...', file=sys.stderr)
    sys_turns, _ = load_rttms(sys_rttm_fns)
    warn('No universal evaluation map specified. Approximating from reference and speaker turn extents...')
    uem = gen_uem(ref_turns, sys_turns)

    # Trim turns to UEM scoring regions and merge any that overlap.
    info('Trimming reference speaker turns to UEM scoring regions...', file=sys.stderr)
    ref_turns = trim_turns(ref_turns, uem)
    info('Trimming system speaker turns to UEM scoring regions...', file=sys.stderr)
    sys_turns = trim_turns(sys_turns, uem)
    info('Checking for overlapping reference speaker turns...', file=sys.stderr)
    ref_turns = merge_turns(ref_turns)
    info('Checking for overlapping system speaker turns...', file=sys.stderr)
    sys_turns = merge_turns(sys_turns)

    # Score.
    info('Scoring...', file=sys.stderr)
    check_for_empty_files(ref_turns, sys_turns, uem)
    file_scores, global_scores = score(ref_turns, sys_turns, uem, collar=0.25, ignore_overlaps=False)
    print_table(file_scores, global_scores, 3)
