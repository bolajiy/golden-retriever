import argparse
import json
import os
import string
import itertools
import functools
import concurrent.futures

import pandas as pd


def chk_mkdir(dirname):
    if isinstance(dirname, str):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
    else:
        try:
            dirnames = iter(dirname)
            for d in dirnames:
                chk_mkdir(d)
        except TypeError:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)


def remove_punctuation(string_x: str):
    return ''.join([_ for _ in string_x if _ not in string.punctuation])


def to_lower(string_x: str, tr: bool = False, special=None, sep='', use_spaces=True) -> str:
    if special is None:
        special = {}
    string_x = remove_punctuation(string_x)
    y = [x.lower() if x not in special.keys() else special[x]
         for x in string_x]
    if not use_spaces:
        y = [x for x in y if not(x.isspace())]
    return sep.join(y).strip()


def kst_normalization(pre_scores, alpha=0, beta=999.9, duration=36000,):
    if len(pre_scores) <= 1 or alpha <= 0:
        return 1
    else:
        threshold = (beta * alpha * sum(pre_scores)) / (duration + (beta-1) * alpha * sum(pre_scores))
        a = 1./(2 * threshold)
        return a

def load_segments(segments_file):
    segments = dict()
    with open(segments_file) as _seg:
        for line in _seg:
            ln = line.split()
            segments[ln[0]] = ln[1:]
    return segments



def read_rttm(rttm_file: str):
    'LEXEME 16257_A_20120709_025101_001451 1 0.13 0.44 alo <NA> <NA> <NA>'
    'islex utterance channel begin duration word e1 e2 e3'
    names = 'islex utterance channel begin duration word e1 e2 e3'.split()
    rttm = pd.read_csv(rttm_file, sep=' ', header=None,
                       names=names, na_filter=False)
    rttm = rttm[rttm['islex'] == 'LEXEME']
    rttm["end"] = rttm["begin"] + rttm["duration"]
    locations = rttm[['utterance', 'begin', 'end', 'word']]
    return locations


def load_results(results_file, kwid, keywords, segments=None, frame_length=0.04,
                 beta=999.9, duration=36000, kst_alpha=0):
    lines = [line.strip().split() for line in open(results_file, encoding='UTF-8')]
    kwtext = keywords[kwid]
    all_scores = []
    results = {}
    for kwid_r, utt, beg, end, score in lines:
        assert kwid == kwid_r
        if segments is None:
            doc = utt
            segment_beg = 0
        else:
            segment = segments[utt]
            doc, segment_beg, segment_end = segment
        beg = float(beg) * frame_length + float(segment_beg)
        end = float(end) * frame_length + float(segment_beg)
        score = float(score)
        if doc == '_kwtext_':
            raise ValueError('utterance id should not be _kwtext_ as it is reserved')

        if doc not in results:
            results[doc] = []
        all_scores.append(score)
        results[doc].append([kwid, kwtext, doc, beg, end, score])

    normalization_scale = kst_normalization(
        all_scores,
        kst_alpha,
        beta,
        duration
    )
    for doc, val in results.items():
        for v in val:
            v[-1] *= normalization_scale

    results['_kwtext_'] = kwtext
    max_score = max(all_scores) * normalization_scale if all_scores else None
    min_score = min(all_scores) * normalization_scale if all_scores else None
    return results, min_score, max_score


def align(hypothesis_dict, reference_dict, score_tolerance=0.5):
    list_hits = []
    list_fa = []
    list_misses = []

    all_docs = list(reference_dict.keys()) + list(hypothesis_dict.keys())
    all_docs = [doc for doc in all_docs if doc != '_kwtext_']
    all_docs = sorted(list(set(all_docs)))

    for doc in all_docs:
        if doc not in reference_dict:
            list_fa += hypothesis_dict[doc]
        elif doc not in hypothesis_dict:
            list_misses += reference_dict[doc]
        else:
            hlocs = hypothesis_dict[doc]
            rlocs = reference_dict[doc]
            possible_hits_by_r = {r: [] for r in range(len(rlocs))}
            possible_hits_by_h = {h: [] for h in range(len(hlocs))}
            all_hyps_refs = []
            for h, hloc in enumerate(hlocs):
                for r, rloc in enumerate(rlocs):
                    hmid = (hloc[3] + hloc[4])/2
                    rmid = (rloc[2] + rloc[3])/2
                    if abs(hmid - rmid) <= score_tolerance:
                        possible_hits_by_r[r].append(h)
                        possible_hits_by_h[h].append(r)
                        all_hyps_refs.append((h, r))

            max_num_hits = 0
            max_total_score = 0
            best_alignment = []

            all_alignment_combinations = itertools.combinations(
                all_hyps_refs,
                min(len(possible_hits_by_h), len(possible_hits_by_r))
            )
            
            all_alignment_combinations = list(all_alignment_combinations)
            
            for combination in all_alignment_combinations:
                total_score = 0
                hits = []
                for h, r in combination:
                    if r in possible_hits_by_h[h]:
                        hits.append((h, r))
                        total_score += hlocs[h][5]

                if len(hits) > max_num_hits:
                    best_alignment = hits
                    max_num_hits = len(hits)
                    max_total_score = total_score
                elif len(hits) == max_num_hits and total_score > max_total_score:
                    best_alignment = hits
                    max_num_hits = len(hits)
                    max_total_score = total_score

            list_hits += [(hlocs[h], rlocs[r]) for h, r in best_alignment]
            hit_rs = [r for _, r in best_alignment]
            hit_hs = [h for h, _ in best_alignment]
            list_fa += [hloc for h, hloc in enumerate(hlocs) if h not in hit_hs]
            list_misses += [rloc for r, rloc in enumerate(rlocs) if r not in hit_rs]

    return list_hits, list_fa, list_misses

def get_hits_fas_misses(single_kw_results, rttm, rttm_tolerance=0.1, score_tolerance=0.5, eps=0.01):
    kwtext = single_kw_results['_kwtext_']
    words = kwtext.split()
    num_words = len(words)

    all_candidates = [rttm[rttm['word'] == word] for word in words]
    target_locations = [[location['word'], location['utterance'], location['begin'], location['end']]
                        for _, location in all_candidates[0].iterrows()]
    if num_words > 1:
        for candidates_word_i in all_candidates[1:]:
            target_locations_word_i = [[location['word'], location['utterance'], location['begin'], location['end']]
                                       for _, location in candidates_word_i.iterrows()]
            tentative_target_locations = []
            for orig_location in target_locations:
                for location_to_maybe_append in target_locations_word_i:
                    if (orig_location[1] == location_to_maybe_append[1]  # Same utterance
                        and location_to_maybe_append[2] >= (orig_location[3] - eps)  # Occurs later
                        and location_to_maybe_append[2] - orig_location[3] <= rttm_tolerance):  # But close enough
                        tentative_target_locations.append([
                            ' '.join([orig_location[0], location_to_maybe_append[0]]),
                            orig_location[1],
                            orig_location[2],
                            location_to_maybe_append[3],
                        ]
                        )
                        break
            target_locations = tentative_target_locations
    target_locations_dict = {}
    for location in target_locations:
        if location[1] not in target_locations_dict:
            target_locations_dict[location[1]] = []
        target_locations_dict[location[1]].append(location)
    
    return align(single_kw_results, target_locations_dict, score_tolerance=score_tolerance)

def compute_single_term_atwv(alignment, threshold, duration, beta):
    hits, false_alarms, misses = alignment
    hit_scores = [hit[0][5] for hit in hits]
    fa_scores = [fa[5] for fa in false_alarms]

    high_hits = [s for s in hit_scores if s >= threshold]
    high_fas = [s for s in fa_scores if s >= threshold]

    num_hits = len(high_hits)
    num_fas = len(high_fas)
    num_misses = len(misses) + (len(hit_scores) - len(high_hits))

    if (num_misses + num_hits) == 0:
        atwv_single = 0
        weight = 0
    else:
        pmiss = num_misses / (num_misses + num_hits)
        pfa = num_fas / (duration - (num_misses + num_hits))
        atwv_single = 1 - (pmiss + beta * pfa)
        weight = 1

    return atwv_single, weight, (num_hits, num_fas, num_misses)


def compute_atwv(alignment, atwv_threshold, duration, beta):
    atwvs = [compute_single_term_atwv(metrics, atwv_threshold, duration, beta)
             for metrics in alignment]

    vs = [_v[0] for _v in atwvs]
    ws = [_v[1] for _v in atwvs]

    atwv = sum([a * b for a, b in zip(vs, ws)]) / max(sum([b for b in ws]), 1)
    return atwv, atwv_threshold, ws


def compute_mtwv(alignment, duration, beta, min_score, max_score, mtwv_granularity):
    assert max_score > min_score
    atwv_threshold = min_score
    best_atwv, best_threshold, ws = compute_atwv(alignment, atwv_threshold, duration, beta)

    while atwv_threshold <= max_score:
        atwv_threshold += mtwv_granularity
        atwv, _, _ = compute_atwv(alignment, atwv_threshold, duration, beta)
        if atwv >= best_atwv:
            best_threshold = atwv_threshold
            best_atwv = atwv

    return best_atwv, best_threshold, ws


def compute_otwv(alignment, duration, beta, min_score, max_score, mtwv_granularity):
    mtwvs = []
    weights = []
    for metric in alignment:
        mtwv, _, ws = compute_mtwv([metric], duration, beta, min_score, max_score, mtwv_granularity)
        mtwvs.append(mtwv)
        weights += ws
    otwv = sum(mtwvs) / max(sum(weights), 1)
    return otwv, None, weights

def compute_stwv(alignment, duration, min_score):
    atwv_threshold = min_score - 0.1
    best_atwv, _, ws = compute_atwv(alignment, atwv_threshold, duration, 0)
    return best_atwv, None, ws


def compute_metrics(alignment, atwv_threshold, duration, beta, min_score, max_score, mtwv_granularity):
    atwv = compute_atwv(alignment, atwv_threshold, duration, beta)
    mtwv = compute_mtwv(alignment, duration, beta, min_score, max_score, mtwv_granularity)
    otwv = compute_otwv(alignment, duration, beta, min_score, max_score, mtwv_granularity)
    stwv = compute_stwv(alignment, duration, min_score)
    return atwv, mtwv, otwv, stwv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers', '--nw', type=int, default=1)
    parser.add_argument('--atwv-threshold', type=float, default=0.5)
    parser.add_argument('--kst-alpha', type=float, default=0)
    parser.add_argument('--frame-length', '--fl', type=float, default=0.04,
                        help='length of each frame in the results files')

    parser.add_argument('--mtwv-granularity', type=float, default=1e-3)
    parser.add_argument('--char-map',
                        help='optional mapping of upper case to lower case')
    parser.add_argument('--categories',
                        help='optional mapping from kwlist to categories')
    parser.add_argument('--score-tolerance', type=float, default=0.5,
                        help='maximum distance allowed between the midpoints '
                             'of a hypothesis and reference to be considered a hit')
    parser.add_argument('--beta', type=float, default=999.9)

    parser.add_argument('--segments-file', '--segments',
                        help='path to segments file containing mapping from utterances to recordings')
    parser.add_argument('keywords_file', help='list of keyword ids and keyword text')
    parser.add_argument('duration', type=float)
    parser.add_argument('rttm_file')
    parser.add_argument('results_directory')

    parser.add_argument('output_directory')

    args = parser.parse_args()

    rttm = read_rttm(rttm_file=args.rttm_file)
    if args.segments_file:
        segments = load_segments(args.segments_file)
    else:
        segments = None

    if args.char_map is not None and os.path.isfile(args.char_map):
        special = {line.strip().split(':')[0]: line.strip().split(':')[1]
                   for line in open(args.char_map, encoding='UTF-8')}
    else:
        special = None
    keywords = {line.strip().split(maxsplit=1)[0]: to_lower(' '.join(line.strip().split(maxsplit=1)[1:]), special=special)
                for line in open(args.keywords_file, encoding='UTF-8')}

    results = [load_results(os.path.join(args.results_directory, fname),
                            fname, keywords, segments, args.frame_length,
                            beta=args.beta, duration=args.duration, kst_alpha=args.kst_alpha)
               for fname in os.listdir(args.results_directory)]

    mins = [r[1] for r in results]
    maxs = [r[2] for r in results]
    results = [r[0] for r in results]

    min_score = min([_ for _ in mins if _ is not None])
    max_score = max([_ for _ in maxs if _ is not None])

    _get_hits_fas_misses = functools.partial(get_hits_fas_misses, rttm=rttm, eps=0.01, score_tolerance=args.score_tolerance)
    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            alignment = pool.map(_get_hits_fas_misses, results)
            alignment = list(alignment)
    else:
        alignment = [_get_hits_fas_misses(result) for result in results]

    chk_mkdir(args.output_directory)
    with open(os.path.join(args.output_directory, 'alignment.csv'), 'w', encoding='UTF-8') as _out:
        _out.write('file,termid,term,ref_bt,ref_et,sys_bt,sys_et,sys_score,sys_decision,alignment\n')
        for kwid, ali in zip(keywords.keys(), alignment):
            hits, fas, misses = ali
            kwtext = keywords[kwid]

            for hyp, ref in hits:
                # hyp: [kwid, kwtext, doc, beg, end, score]
                # ref: [location['word'], location['utterance'], location['begin'], location['end']]
                decision = 'YES_CORR' if hyp[5] > args.atwv_threshold else 'NO,MISS'
                to_write = f'{hyp[2]},{hyp[0]},{hyp[1]},{ref[2]:.2f},{ref[3]:.2f},{hyp[3]:.2f},{hyp[4]:.2f},{hyp[5]:.4f},{decision}\n'
                _out.write(to_write)

            for ref in misses:
                decision = ',MISS'
                to_write = f'{ref[1]},{kwid},{kwtext},{ref[2]:.2f},{ref[3]:.2f},,,,{decision}\n'
                _out.write(to_write)

            for hyp in fas:
                decision = 'YES,FA' if hyp[5] > args.atwv_threshold else 'NO,CORR'
                to_write = f'{hyp[2]},{hyp[0]},{hyp[1]},,,{hyp[3]:.2f},{hyp[4]:.2f},{hyp[5]:.4f},{decision}\n'
                _out.write(to_write)

    atwv, mtwv, otwv, stwv = compute_metrics(
        alignment,
        args.atwv_threshold,
        args.duration,
        args.beta,
        min_score,
        max_score,
        args.mtwv_granularity,
    )

    summary = {
        'Header': ['TWV', 'THRESHOLD'],
        'ATWV': atwv[:2],
        'MTWV': mtwv[:2],
        'OTWV': otwv[:2],
        'STWV': stwv[:2],
        }

    with open(os.path.join(args.output_directory, 'summary.json'), 'w', encoding='UTF-8') as _out:
        json.dump(summary, _out, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_directory, 'summary.tsv'), 'w', encoding='UTF-8') as _out:
        _out.write(f'Category\tATWV\tThr.\tMTWV\tThr.\tOTWV\tThr.\tSTWV\tThr.\n')
        for cat, cat_val in {'All': summary}.items():
            _out.write(f'{cat}\t')
            for k, v in cat_val.items():
                if k == 'Header':
                    continue
                m = v[0]
                t = v[1]
                if t is not None:
                    t = f'{t:.4f}'
                _out.write(f'{m:.4f}\t{t}\t')
            _out.write('\n')

    per_query_summary = {}

    for key, per_query_ali in zip(keywords.keys(), alignment):
        atwv, mtwv, otwv, stwv = compute_metrics(
            [per_query_ali],
            args.atwv_threshold,
            args.duration,
            args.beta,
            min_score,
            max_score,
            args.mtwv_granularity,
            )
        per_query_summary[key] = {
            'Header': ['TWV', 'THRESHOLD'],
            'ATWV': atwv[:2],
            'MTWV': mtwv[:2],
            'OTWV': otwv[:2],
            'STWV': stwv[:2],
            }

    with open(os.path.join(args.output_directory, 'per_query_summary.json'), 'w', encoding='UTF-8') as _out:
        json.dump(per_query_summary, _out, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_directory, 'per_query_summary.tsv'), 'w', encoding='UTF-8') as _out:
        _out.write(f'Category\tATWV\tThr.\tMTWV\tThr.\tOTWV\tThr.\tSTWV\tThr.\n')
        for cat, cat_val in per_query_summary.items():
            _out.write(f'{cat}\t')
            for k, v in cat_val.items():
                if k == 'Header':
                    continue
                m = v[0]
                t = v[1]
                if t is not None:
                    t = f'{t:.4f}'
                _out.write(f'{m:.4f}\t{t}\t')
            _out.write('\n')

    if args.categories:
        categories_dict = {}
        lines = [line.strip().split() for line in open(args.categories, encoding='UTF-8')]
        assert len(lines) == len(keywords), 'number of queries in keyword list and categories file must be identical'
        for i, (line, key) in enumerate(zip(lines, keywords.keys())):
            assert line[0] == key, f'mis-match at {(line, key)}'
            for category in line[1:]:
                if category == key:
                    continue
                if category not in categories_dict:
                    categories_dict[category] = []
                categories_dict[category].append(i)

        categories_dict = {k: categories_dict[k] for k in sorted(categories_dict.keys())}
        categorical_summary = {}
        for category, kwids in categories_dict.items():
            categorical_alignment = [alignment[i] for i in kwids]
            atwv, mtwv, otwv, stwv = compute_metrics(
                categorical_alignment,
                args.atwv_threshold,
                args.duration,
                args.beta,
                min_score,
                max_score,
                args.mtwv_granularity,
                )
            categorical_summary[category] = {
                'Header': ['TWV', 'THRESHOLD'],
                'ATWV': atwv[:2],
                'MTWV': mtwv[:2],
                'OTWV': otwv[:2],
                'STWV': stwv[:2],
                }

        with open(os.path.join(args.output_directory, 'categorical_summary.json'), 'w', encoding='UTF-8') as _out:
            json.dump(categorical_summary, _out, indent=2, ensure_ascii=False)

        with open(os.path.join(args.output_directory, 'categorical_summary.tsv'), 'w', encoding='UTF-8') as _out:
            _out.write(f'Category\tATWV\tThr.\tMTWV\tThr.\tOTWV\tThr.\tSTWV\tThr.\n')
            for cat, cat_val in categorical_summary.items():
                _out.write(f'{cat}\t')
                for k, v in cat_val.items():
                    if k == 'Header':
                        continue
                    m = v[0]
                    t = v[1]
                    if t is not None:
                        t = f'{t:.4f}'
                    _out.write(f'{m:.4f}\t{t}\t')
                _out.write('\n')
            


if __name__ == '__main__':
    main()
