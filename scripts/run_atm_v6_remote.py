"""Frozen, text-only candidate scoring on the existing private GPU host.

Importing this file uses only the standard library. GPU imports/model loading
require --execute. Input text never appears in output logs or error messages.
Scores are an ordered, fsynced prefix of one ordinal-modulo-four shard.
"""
import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


REMOTE_ROOT = Path('/software/cenzihan/agent-memory-multimodal')
MODEL_PATH = Path('/software/cenzihan/models/Qwen3-VL-Reranker-2B')
DEPENDENCIES = ('/software/cenzihan/agent-memory-multimodal/reranker-deps',
                '/user/cenzihan/projects/zindi-underwater-noise/env/lib/python3.10/site-packages')
SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".')
SPEC = dict(model='Qwen/Qwen3-VL-Reranker-2B', revision='4bd860ac4f15ad1897a214615cccc700f8f71818',
    weights_sha256='466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9',
    input_mode='source_text_only', instruction="Retrieve text relevant to the user's query.",
    max_length=4096, truncation='document_prefix_only_preserve_query_and_template',
    score='yes_minus_no_logit_float32', score_direction='higher_is_better',
    batch_size=8, padded_token_budget=8192, dtype='bfloat16', attention='sdpa', workers=4,
    max_seconds=7200, generation=False, new_embedding_calls=0)
FILES = {'queries.jsonl', 'sources.jsonl', 'candidates.jsonl', 'pairs.jsonl'}
# Metadata controlling native architecture/tokenization is pinned as well as weights.
MODEL_FILES = {
    'config.json': '82d38a8f803e38e13986fdd622114a6fec12a834adbd3cee9253d757a257d23d',
    'chat_template.jinja': '3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4',
    'additional_chat_templates/reranker.jinja': '47c758cb74d7f1e20e22483949a5ba4c8c1f4515126ad173da1c63211f472aa7',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'preprocessor_config.json': 'fd32af55c2d3846adb0bc46df8eb07c92c332b31b34c338ae85259f3f3951f24',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': '81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'model.safetensors': SPEC['weights_sha256'],
}


class WorkerBlocked(RuntimeError):
    pass


class PartialScoreTail(WorkerBlocked):
    pass


class WorkerDeadline(WorkerBlocked):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WorkerBlocked('duplicate JSON key')
            result[key] = value
        return result
    def bad(_value):
        raise WorkerBlocked('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def pair_identity(query_id, source_id):
    return digest(json.dumps([query_id, source_id], ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def json_lines(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024 * 1024:
        raise WorkerBlocked('bounded ordinary input file required')
    with path.open('rb') as stream:
        for line in stream:
            if not line.strip() or len(line) > 4 * 1024 * 1024:
                raise WorkerBlocked('empty or oversized JSONL row')
            yield strict_json(line)


def load_inputs(work, runner_path=None):
    work = Path(work)
    frozen_path = work / 'FROZEN_CONFIG.json'
    if frozen_path.is_symlink() or frozen_path.stat().st_size > 2 * 1024 * 1024:
        raise WorkerBlocked('bounded ordinary freeze required')
    frozen = strict_json(frozen_path.read_bytes())
    if frozen.get('reranker_spec') != SPEC or frozen.get('workers', 4) != 4:
        raise WorkerBlocked('fixed reranker specification differs')
    runner_path = Path(runner_path or __file__)
    if frozen.get('remote_runner_sha256') != file_hash(runner_path):
        raise WorkerBlocked('remote runner differs from freeze')
    hashes = frozen.get('artifact_hashes')
    if not isinstance(hashes, dict):
        raise WorkerBlocked('frozen input hashes required')
    names = {}
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or '\\' in relative or relative.startswith('/') or '..' in relative.split('/'):
            raise WorkerBlocked('relative frozen artifact path required')
        name = relative.rsplit('/', 1)[-1]
        if name not in FILES or name in names or not valid_digest(expected):
            raise WorkerBlocked('exact distinct frozen input files required')
        path = work / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise WorkerBlocked('frozen input bytes differ')
        names[name] = expected
    if set(names) != FILES:
        raise WorkerBlocked('all four frozen inputs required')
    queries, sources, pairs = {}, {}, []
    for row in json_lines(work / 'queries.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'query_id', 'dataset', 'id', 'qtype', 'question'}
                or not all(nonempty(row[key]) for key in row)
                or row['dataset'] not in ('regular', 'hard')
                or row['qtype'] not in ('number', 'open_end', 'list_recall')
                or row['query_id'] != json.dumps([row['dataset'], row['id']], ensure_ascii=False, separators=(',', ':'))
                or row['query_id'] in queries):
            raise WorkerBlocked('exact unique question-only records required')
        queries[row['query_id']] = row['question']
    for row in json_lines(work / 'sources.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'source_id', 'text'}
                or not nonempty(row['source_id']) or not nonempty(row['text']) or row['source_id'] in sources):
            raise WorkerBlocked('exact unique source-only records required')
        sources[row['source_id']] = row['text']
    seen = set()
    for row in json_lines(work / 'pairs.jsonl'):
        if (not isinstance(row, dict) or set(row) != {'pair_id', 'query_id', 'source_id'}
                or not all(isinstance(row[key], str) for key in row)
                or row['query_id'] not in queries or row['source_id'] not in sources
                or row['pair_id'] != pair_identity(row['query_id'], row['source_id']) or row['pair_id'] in seen):
            raise WorkerBlocked('unique frozen query/source pair required')
        seen.add(row['pair_id'])
        pairs.append(row)
    for field, actual in (('query_count', len(queries)), ('source_count', len(sources)), ('pair_count', len(pairs))):
        if type(frozen.get(field)) is not int or frozen[field] != actual or not actual:
            raise WorkerBlocked('frozen input count differs')
    return dict(frozen=frozen, frozen_sha256=file_hash(frozen_path), hashes=names,
                queries=queries, sources=sources, pairs=pairs)


def messages(query, document):
    return [dict(role='system', content=[dict(type='text', text=SYSTEM)]),
            dict(role='user', content=[dict(type='text', text='<Instruct>: ' + SPEC['instruction']),
                dict(type='text', text='<Query>:'), dict(type='text', text=query),
                dict(type='text', text='\n<Document>:'), dict(type='text', text=document)])]


def fit_document(query, document, render, encode, max_length=4096):
    """Render the real template each time; shorten only an exact document prefix.

    Character-prefix search avoids slicing UTF-8/BPE tokens into replacement
    characters, and retains the original full query and assistant suffix. Token
    counts need not be perfectly monotonic: every chosen result is rechecked.
    """
    if not nonempty(query) or not nonempty(document) or type(max_length) is not int or max_length < 1:
        raise WorkerBlocked('nonempty original text and positive token bound required')
    marker = '__AML_DOCUMENT_' + digest((query + '\0' + document).encode()) + '__'
    if marker in query or marker in document:
        raise WorkerBlocked('template marker collision')
    rendered = render(messages(query, marker))
    if not isinstance(rendered, str) or rendered.count(marker) != 1:
        raise WorkerBlocked('unsupported native chat template')
    prefix, suffix = rendered.split(marker)
    if not suffix or render(messages(query, document)) != prefix + document + suffix:
        raise WorkerBlocked('template must preserve source text and assistant suffix')
    def tokens(characters):
        result = encode(prefix + document[:characters] + suffix)
        if not isinstance(result, list) or not result or any(type(value) is not int for value in result):
            raise WorkerBlocked('native integer token sequence required')
        return result
    full = tokens(len(document))
    if len(full) <= max_length:
        return dict(input_ids=full, original_tokens=len(full), document_characters=len(document), truncated=False)
    if len(tokens(0)) >= max_length:
        raise WorkerBlocked('query and template exceed fixed bound; query is never truncated')
    low, high, best = 0, len(document), None
    while low <= high:
        middle = (low + high) // 2
        encoded = tokens(middle)
        if len(encoded) <= max_length:
            best = (middle, encoded)
            low = middle + 1
        else:
            high = middle - 1
    if best is None or not document[:best[0]].strip() or len(best[1]) > max_length:
        raise WorkerBlocked('no nonempty document prefix fits fixed bound')
    return dict(input_ids=best[1], original_tokens=len(full), document_characters=best[0], truncated=True)


def batches(items, batch_size=8, padded_budget=8192):
    """Stable order and a bound on batch_size * longest padded sequence."""
    pending, longest = [], 0
    for item in items:
        length = len(item['input_ids'])
        if not 1 <= length <= SPEC['max_length'] or length > padded_budget:
            raise WorkerBlocked('encoded input exceeds frozen capacity')
        candidate = max(longest, length)
        if pending and (len(pending) >= batch_size or (len(pending) + 1) * candidate > padded_budget):
            yield pending
            pending, longest = [], 0
        pending.append(item)
        longest = max(longest, length)
    if pending:
        yield pending


def shard(pairs, worker, workers=4):
    if workers != 4 or type(worker) is not int or not 0 <= worker < workers:
        raise WorkerBlocked('fixed four-worker shard required')
    return [pair for ordinal, pair in enumerate(pairs) if ordinal % workers == worker]


def atomic_json(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def committed_prefix(path, expected, *, repair_partial_tail=False):
    path = Path(path)
    if path.is_symlink():
        raise WorkerBlocked('score symlink forbidden')
    if not path.exists():
        return 0
    def validate(raw):
        rows = raw.splitlines()
        if len(rows) > len(expected):
            raise WorkerBlocked('too many committed shard results')
        for number, line in enumerate(rows):
            row = strict_json(line)
            if (not isinstance(row, dict) or set(row) != {'pair_id', 'score'}
                    or row['pair_id'] != expected[number]['pair_id']
                    or type(row['score']) not in (int, float) or not math.isfinite(row['score'])):
                raise WorkerBlocked('committed results are not the exact finite shard prefix')
        return len(rows)
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        if not repair_partial_tail:
            raise PartialScoreTail('explicit local-inference tail repair required')
        cutoff = raw.rfind(b'\n') + 1
        validate(raw[:cutoff])  # Never repair a tail when the preceding journal is corrupt.
        tail = raw[cutoff:]
        stamp = str(time.time_ns())
        backup = path.with_name(path.name + '.partial-tail-' + stamp + '.bin')
        with backup.open('xb') as stream:
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(path.with_name(path.name + '.repair-' + stamp + '.json'),
            dict(original_sha256=digest(raw), kept_bytes=cutoff, removed_bytes=len(tail),
                 tail_sha256=digest(tail), tail_backup=backup.name,
                 reason='Explicit repair of incomplete local inference output; no paid request is replayed'))
        with path.open('r+b') as stream:
            stream.truncate(cutoff)
            stream.flush()
            os.fsync(stream.fileno())
        raw = raw[:cutoff]
    return validate(raw)


def append_scores(path, batch, scores):
    if len(batch) != len(scores) or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores):
        raise WorkerBlocked('one finite score per pair required')
    raw = ''.join(canonical(dict(pair_id=item['pair_id'], score=float(value))) + '\n'
                  for item, value in zip(batch, scores)).encode('utf-8')
    with Path(path).open('ab') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_lock(path):
    import fcntl
    if path.is_symlink():
        raise WorkerBlocked('lock symlink forbidden')
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_gpu_idle(uuid):
    result = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, timeout=15, check=True)
    rows = [line.strip().split(', ') for line in result.stdout.decode('ascii').splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != uuid or int(rows[0][1]) > 128 or int(rows[0][2]) < 12000:
        raise WorkerBlocked('assigned GPU is not idle with sufficient memory')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                            capture_output=True, timeout=15, check=True)
    if any(line.split(',', 1)[0].strip() == uuid for line in result.stdout.decode('ascii').splitlines()):
        raise WorkerBlocked('assigned GPU already has a compute process')


def verify_model():
    plan = strict_json((MODEL_PATH / 'DOWNLOAD_PLAN.json').read_bytes())
    if plan.get('model_id') != SPEC['model'] or plan.get('revision') != SPEC['revision']:
        raise WorkerBlocked('local model revision differs')
    for relative, expected in MODEL_FILES.items():
        path = MODEL_PATH / relative
        if path.is_symlink() or file_hash(path) != expected:
            raise WorkerBlocked('local pinned model artifact differs')


def run_worker(args):
    work = Path(args.work).resolve()
    if not work.is_dir() or work == REMOTE_ROOT or not work.is_relative_to(REMOTE_ROOT):
        raise WorkerBlocked('existing private remote work directory required')
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu_uuid):
        raise WorkerBlocked('explicit GPU UUID required')
    if args.workers != 4 or not 0 <= args.worker_index < 4 or not 1 <= args.max_seconds <= 7200:
        raise WorkerBlocked('fixed workers and bounded deadline required')
    if args.max_pairs is not None and args.max_pairs < 1:
        raise WorkerBlocked('positive newly committed pair limit required')
    with exclusive_lock(work / ('worker-' + str(args.worker_index) + '.lock')):
        data = load_inputs(work)
        expected = shard(data['pairs'], args.worker_index)
        score_path = work / ('scores-' + str(args.worker_index) + '.jsonl')
        completed = committed_prefix(score_path, expected, repair_partial_tail=args.repair_partial_tail)
        initial = completed
        started = time.monotonic()
        stats = dict(state='preflight', worker_index=args.worker_index, workers=4, gpu_uuid=args.gpu_uuid,
            total_pairs=len(expected), existing_pairs=initial, completed_pairs=completed, new_pairs=0,
            input_tokens_this_invocation=0, truncated_pairs_this_invocation=0, batches_this_invocation=0,
            frozen_sha256=data['frozen_sha256'], remote_runner_sha256=file_hash(Path(__file__)),
            spec=SPEC, provider_requests=0, generation_called=False)
        def report(state, error_type=None):
            stats.update(state=state, elapsed_seconds=time.monotonic()-started, completed_pairs=completed,
                         new_pairs=completed-initial)
            if error_type is not None:
                stats['error_type'] = error_type
            if score_path.exists() and state != 'running':
                stats['scores_sha256'] = file_hash(score_path)
            atomic_json(work / ('progress-' + str(args.worker_index) + '.json'), stats)
            atomic_json(work / ('report-' + str(args.worker_index) + '.json'), stats)
            print(canonical(stats), flush=True)
        if completed == len(expected):
            report('complete')
            return
        def deadline(_signum, _frame):
            raise WorkerDeadline('local worker deadline reached')
        old_handler = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
        try:
            with exclusive_lock(REMOTE_ROOT / ('v6-' + args.gpu_uuid + '.lock')):
                check_gpu_idle(args.gpu_uuid)
                report('verifying_model')
                verify_model()
                os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
                sys.path[:0] = list(DEPENDENCIES)
                import torch
                import transformers
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
                if torch.__version__ != '2.2.2+cu121' or transformers.__version__ != '4.57.1':
                    raise WorkerBlocked('reviewed runtime versions required')
                transformers.utils.logging.set_verbosity_error()
                transformers.utils.logging.disable_progress_bar()
                torch.set_num_threads(2)
                torch.backends.cuda.matmul.allow_tf32 = False
                if not torch.cuda.is_bf16_supported():
                    raise WorkerBlocked('fixed bfloat16 requires GPU support')
                torch.cuda.set_per_process_memory_fraction(.85, 0)
                torch.cuda.reset_peak_memory_stats(0)
                report('loading_model')
                model = Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_PATH), local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                    device_map={'': 0}, attn_implementation='sdpa').eval()
                processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True,
                    trust_remote_code=False, use_fast=False, padding_side='left')
                tokenizer = processor.tokenizer
                tokenizer.padding_side = 'left'
                vocab = tokenizer.get_vocab()
                if vocab.get('yes') != 9693 or vocab.get('no') != 2152:
                    raise WorkerBlocked('pinned yes/no vocabulary differs')
                head = model.get_output_embeddings().weight
                direction = head[vocab['yes']].detach().float() - head[vocab['no']].detach().float()
                limit = len(expected) if args.max_pairs is None else min(len(expected), initial + args.max_pairs)
                def encoded():
                    for pair in expected[initial:limit]:
                        if time.monotonic() - started >= args.max_seconds:
                            raise WorkerDeadline('local worker deadline reached')
                        item = fit_document(data['queries'][pair['query_id']], data['sources'][pair['source_id']],
                            lambda value: processor.apply_chat_template(value, tokenize=False, add_generation_prompt=True),
                            lambda value: tokenizer.encode(value, add_special_tokens=False))
                        item['pair_id'] = pair['pair_id']
                        yield item
                for batch in batches(encoded()):
                    inputs = tokenizer.pad({'input_ids': [row['input_ids'] for row in batch]},
                                           padding=True, return_tensors='pt')
                    if inputs['input_ids'].numel() > 8192 or not bool(inputs['attention_mask'][:, -1].all()):
                        raise WorkerBlocked('padded token bound or left padding violated')
                    inputs = inputs.to(model.device)
                    with torch.inference_mode():
                        hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1]
                        values = torch.mv(hidden.float(), direction).cpu().tolist()
                    torch.cuda.synchronize()
                    append_scores(score_path, batch, values)
                    completed += len(batch)
                    stats['input_tokens_this_invocation'] += sum(len(row['input_ids']) for row in batch)
                    stats['truncated_pairs_this_invocation'] += sum(row['truncated'] for row in batch)
                    stats['batches_this_invocation'] += 1
                    stats['peak_allocated_mib'] = torch.cuda.max_memory_allocated(0)/1024**2
                    stats['peak_reserved_mib'] = torch.cuda.max_memory_reserved(0)/1024**2
                    report('running')
                if committed_prefix(score_path, expected) != completed:
                    raise WorkerBlocked('committed score prefix differs after execution')
                for name, expected_hash in data['hashes'].items():
                    if file_hash(work / name) != expected_hash:
                        raise WorkerBlocked('input changed during execution')
                if file_hash(work / 'FROZEN_CONFIG.json') != data['frozen_sha256']:
                    raise WorkerBlocked('freeze changed during execution')
                report('complete' if completed == len(expected) else 'pair_limit_reached')
        except BaseException as exc:
            report('stopped', type(exc).__name__)
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--worker-index', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-pairs', type=int)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--execute', '--live', dest='execute', action='store_true')
    parser.add_argument('--repair-partial-tail', action='store_true',
                        help='Explicitly retain and repair an incomplete local-inference output tail before recomputing it.')
    args = parser.parse_args(argv)
    if not args.execute:
        parser.exit(2, 'Explicit --execute required; no model loading or scoring performed.\n')
    try:
        run_worker(args)
    except BaseException as exc:
        print(canonical(dict(state='stopped', error_type=type(exc).__name__, provider_requests=0)), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
