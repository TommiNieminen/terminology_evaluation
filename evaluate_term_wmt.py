import re
import TER
import stanza
import string
import argparse
import sacrebleu
import TER_modified
from bs4 import BeautifulSoup
import numpy as np

def tokenize_and_truecase(sentence):
	doc_f = l2_stanza(sentence)
	all_sentence_surfaces = []
	for sentence in doc_f.sentences:
		sentence_words = [w for w in sentence.words]
		non_punct_indices = [i for i, word in enumerate(sentence_words) if word.upos != "PUNCT"]
		sentence_surfaces = [w.text for w in sentence_words]
		if non_punct_indices:
			if sentence_words[non_punct_indices[0]].lemma[0].islower():
				sentence_surfaces[non_punct_indices[0]] = sentence_surfaces[non_punct_indices[0]][0].lower() + sentence_surfaces[non_punct_indices[0]][1:]
		else:
			print("no non-punct found: " + " ".join(sentence_surfaces))
		all_sentence_surfaces = all_sentence_surfaces + sentence_surfaces

	return " ".join(all_sentence_surfaces)


def read_reference_data_wmt(lt, ls):
	with open(lt, encoding="utf-8") as inp:
		linest = inp.readlines()
	with open(ls, encoding="utf-8") as inp:
		liness = inp.readlines()

	refs = {}
	sources = []
	outputs = []
	for idx in range(len(liness)):
		linet = linest[idx]
		lines = liness[idx]
		if "</seg>" in lines:
			soups = BeautifulSoup(lines, "lxml")
			soupt = BeautifulSoup(linet, "lxml")

			id = soups.seg['id']
			source_tokens = soups.text.split()
			target_tokens = soupt.text.split()

			source = " " + " ".join(source_tokens) + " "
			target = " " + " ".join(target_tokens) + " "
            
            # if stanza supported, tokenize and truecase target with stanza
			if SUPPORTED:
				target = " " + tokenize_and_truecase(target) + " "

			sources.append(source)
			outputs.append(target)

			terms = []
			terms_l = []
			mod_terms = []
			src_terms = soups.find_all('term')
			tgt_terms = soupt.find_all('term')
			for ids, item in enumerate(tgt_terms):
				src_start = source_tokens.index(src_terms[ids].text.split()[0])
				src_end = source_tokens.index(src_terms[ids].text.split()[-1])
				src_ids = ""
				for ind in range(src_start, src_end - 1):
					src_ids += (str)(ind) + ","
				src_ids += (str)(src_end)

				tgt_start = target_tokens.index(item.text.split()[0])
				tgt_end = target_tokens.index(item.text.split()[-1])
				tgt_ids = ""
				for ind in range(tgt_start, tgt_end - 1):
					tgt_ids += (str)(ind) + ","
				tgt_ids += (str)(tgt_end)

				tgt_term = item['tgt']
				if item.text.strip() not in tgt_term:
					tgt_term = item['tgt'] + "|" + item.text.strip()
				mod_terms.append(f"{src_terms[ids].text} ||| {src_ids} --> {tgt_term} ||| {tgt_ids}")

				if "tgt_original" in tgt_terms[ids]['type']:
					terms.append(f"{src_terms[ids].text} ||| {src_ids} --> {tgt_terms[ids].text} ||| {tgt_ids}")
				if "tgt_lemma" in tgt_terms[ids]['type']:
					if SUPPORTED:
						doc_f = l2_stanza(tgt_terms[ids].text)
						tgt_lemma = ' ' + ' '.join([w.lemma for w in doc_f.sentences[0].words]) + ' '
						terms_l.append(f"{src_terms[ids].text} ||| {src_ids} --> {tgt_lemma} ||| {tgt_ids}")
                        

			refs[id] = (source, target, terms, terms_l, mod_terms)

	return sources, outputs, refs

def read_outputs_wmt(f):
	with open(f) as inp:
		lines = inp.readlines()
	outputs = []
	ids = []
	for line in lines:
		if "</seg>" in line:
			soup = BeautifulSoup(line, "lxml")
			ids.append(soup.seg['id'])
			output = ' ' + ' '.join(soup.seg.text.strip().split()) + ' '
			if SUPPORTED:
				output = " " + tokenize_and_truecase(output) + " "
			outputs.append(output)
	return ids, outputs



def compare_EXACT(hyp, ref):
	_, _, _, _, terms = ref

	if SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' '
		for sentence in doc_f.sentences:
			try:
				# The hash sign is a used as compound marker in stanza lemma output, remove it
				hyp_l = hyp_l  + ' '.join([w.lemma.replace("#","") for w in sentence.words]) + ' '
			except:
				hyp_l = hyp_l

	count_correct = 0
	count_wrong = 0
	count_correct_l = 0
	count_wrong_l = 0
	terms_regexes = []
	starts = []
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired_list = []
		for item in t[0].split("|"):
			desired_list.append("(?= " + item + " )")
		desireds = "|".join(desired_list)
		terms_regexes.append(desireds)
		flag = False
		for desired in desireds.split("|"):
			desired_starts = [m.start() for m in re.finditer(desired, hyp)]
			for desired_start in desired_starts:
				if desired_start not in starts:
					starts.append(desired_start)
					flag = True
					break
		if not flag and SUPPORTED:
			flag_l = False
			for desired in desireds.split("|"):
				desired_starts = [m.start() for m in re.finditer(desired, hyp_l)]
				for desired_start in desired_starts:
					if desired_start not in starts:
						starts.append(desired_start)
						flag_l = True
						break
			if flag_l:
				count_correct_l += 1
			else:
				count_wrong += 1
		else:
			count_correct += 1

	return count_correct, count_wrong, count_correct_l, count_wrong_l, hyp_l, terms_regexes

def compare_TER_w(hyp, ref, lc):
	source, reference, terms, terms_l, _ = ref
	ter = 0
	term_ids = []
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		term_ids.extend(t[1].split(','))
	ter += TER_modified.ter(hyp.split(), reference.split(), lc, term_ids)
	term_l_ids = []
	if terms_l and SUPPORTED:
		doc_f = l2_stanza(hyp)
		hyp_l = ' '
		for sentence in doc_f.sentences:
			hyp_l = hyp_l + ' '.join([w.lemma for w in sentence.words]) + ' '
		doc_f = l2_stanza(reference)
		reference_l = ' '
		for sentence in doc_f.sentences:
			reference_l = reference_l + ' '.join([w.lemma for w in sentence.words]) + ' '
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			term_l_ids.extend(t[1].split(','))
		ter += TER_modified.ter(hyp_l.split(), reference_l.split(), lc, term_l_ids)
		ter /= 2.0
					
	return ter

def compare_exact_window_overlap(hyp, ref, window):
	source, reference, terms, terms_l, _ = ref
	accuracy = 0.0
	matched = 0

	hyp_tokens = hyp.strip().split()
	if not hyp_tokens:
		return 0.0
	reference_tokens = reference.strip().split()
	desireds = {}
	for t in terms:
		t = t.split(' --> ')
		t = t[1].split(' ||| ')
		desired  = " " + t[0].strip() + " "
		desiredindxs = [(int)(item)for item in t[1].split(",")]
		if desired in desireds:
			desireds[desired].append(desiredindxs)
		else:
			desireds[desired] = [desiredindxs]
	for desired in desireds:
		fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
		accs = {}
		for j, listfs in enumerate(desireds[desired]):
			ref_words = []
			for win in range(min(listfs) - 1, -1, -1):
				if 0 <= win < len(reference_tokens) and reference_tokens[win] not in string.punctuation:
					ref_words.append(reference_tokens[win])
					if len(ref_words) == window:
						break
			ref_wordsr = []
			for win in range(max(listfs) + 1, len(reference_tokens), 1):
				if 0 <= win < len(reference_tokens) and reference_tokens[win] not in string.punctuation:
					ref_wordsr.append(reference_tokens[win])
					if len(ref_wordsr) == window:
						break
			ref_words.extend(ref_wordsr)
			for k, ft in enumerate(fts):
				cntfts = hyp.count(" ", 0, ft)
				listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
				hyp_words = []
				for win in range(min(listft) - 1, -1, -1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_words.append(hyp_tokens[win])
						if len(hyp_words) == window:
							break
				hyp_wordsr = []
				for win in range(max(listft) + 1, len(hyp_tokens), 1):
					if hyp_tokens[win] not in string.punctuation:
						hyp_wordsr.append(hyp_tokens[win])
						if len(hyp_wordsr) == window:
							break
				hyp_words.extend(hyp_wordsr)
				cnt = 0
				for ref_word in ref_words:
					if ref_word in hyp_words:
						cnt += 1
						hyp_words.remove(ref_word)
				accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
		accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
		mapped_ref = []
		mapped_hyp = []
		for acc in accs:
			if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
				mapped_ref.append(acc.split("-")[0])
				mapped_hyp.append(acc.split("-")[1])
				matched += 1
				accuracy += accs[acc]
	
	if SUPPORTED and terms_l:
		doc_f = l2_stanza(hyp)
		hyp_l = ' '
		for sentence in doc_f.sentences:
			hyp_l = hyp_l + ' '.join([w.lemma for w in sentence.words]) + ' '
		doc_f = l2_stanza(reference)
		reference_l = ' '
		for sentence in doc_f.sentences:
			reference_l = reference_l + ' '.join([w.lemma for w in sentence.words]) + ' '
		hyp_tokens = hyp.strip().split()
		reference_tokens = reference.strip().split()
		desireds = {}
		for t in terms_l:
			t = t.split(' --> ')
			t = t[1].split(' ||| ')
			desired  = " " + t[0].strip() + " "
			desiredindxs = [(int)(item)for item in t[1].split(",")]
			if desired in desireds:
				desireds[desired].append(desiredindxs)
			else:
				desireds[desired] = [desiredindxs]
		for desired in desireds:
			fts = [m.start() for m in re.finditer(f"(?={desired})", hyp)]
			accs = {}
			for j, listfs in enumerate(desireds[desired]):
				ref_words = []
				for win in range(min(listfs) - 1, -1, -1):
					if reference_tokens[win] not in string.punctuation:
						ref_words.append(reference_tokens[win])
						if len(ref_words) == window:
							break
				ref_wordsr = []
				for win in range(max(listfs) + 1, len(reference_tokens), 1):
					if reference_tokens[win] not in string.punctuation:
						ref_wordsr.append(reference_tokens[win])
						if len(ref_wordsr) == window:
							break
				ref_words.extend(ref_wordsr)
				for k, ft in enumerate(fts):
					cntfts = hyp.count(" ", 0, ft)
					listft = [item for item in range(cntfts, cntfts + len(desired.strip().split()))]
					hyp_words = []
					for win in range(min(listft) - 1, -1, -1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_words.append(hyp_tokens[win])
							if len(hyp_words) == window:
								break
					hyp_wordsr = []
					for win in range(max(listft) + 1, len(hyp_tokens), 1):
						if hyp_tokens[win] not in string.punctuation:
							hyp_wordsr.append(hyp_tokens[win])
							if len(hyp_wordsr) == window:
								break
					hyp_words.extend(hyp_wordsr)
					cnt = 0
					for ref_word in ref_words:
						if ref_word in hyp_words:
							cnt += 1
							hyp_words.remove(ref_word)
					accs[f"{j}-{k}"] = cnt / len(ref_words) if len(ref_words) != 0 else + 0
			accs = dict(sorted(accs.items(), key=lambda item: item[1], reverse=True))
			mapped_ref = []
			mapped_hyp = []
			for acc in accs:
				if acc.split("-")[0] not in mapped_ref and acc.split("-")[1] not in mapped_hyp:
					mapped_ref.append(acc.split("-")[0])
					mapped_hyp.append(acc.split("-")[1])
					matched += 1
					accuracy += accs[acc]
	
	return accuracy / matched if matched > 0 else 0



def exact_match(l2, references, outputs, ids, LOG, match_counts_path):
	correct = 0
	wrong = 0
	correctl = 0
	wrongl = 0
	with open(match_counts_path,'w') as match_counts_file:
		for i, id in enumerate(ids):
			if id in references:
				c, w, cl, wl, hyp_l, terms_regexes = compare_EXACT(outputs[i], references[id])
				match_counts_file.write(f'{id}\t{outputs[i]}\t{hyp_l}\t{",".join(terms_regexes)}\tC{c}\tW{w}\tCL{cl}\n')
				correct += c
				wrong += w
				correctl += cl
				wrongl += wl 

	print(f"Exact-Match Statistics")
	print(f"\tTotal correct: {correct}")
	print(f"\tTotal wrong: {wrong}")
	print(f"\tTotal correct (lemma): {correctl}")
	print(f"\tTotal wrong (lemma): {wrongl}")
	print(f"Exact-Match Accuracy: {(correct + correctl) / (correct + correctl + wrong + wrongl)}")
	with open(LOG, 'a') as op:
		op.write(f"Exact-Match Statistics\n")
		op.write(f"\tTotal correct: {correct}\n")
		op.write(f"\tTotal wrong: {wrong}\n")
		op.write(f"\tTotal correct (lemma): {correctl}\n")
		op.write(f"\tTotal wrong (lemma): {wrongl}\n")
		op.write(f"Exact-Match Accuracy: {(correct + correctl) / (correct + correctl + wrong + wrongl)}\n")

def comet(l2, sources, outputs, references, comet_model_path, LOG):
	from comet.models import download_model, load_from_checkpoint
	#model_path = download_model("Unbabel/wmt22-comet-da",saving_directory="../comet_models")
	model = load_from_checkpoint(comet_model_path)

	data = {"src": sources, "mt": outputs, "ref": references}
	data = [dict(zip(data, t)) for t in zip(*data.values())]

	model_output = model.predict(data, batch_size=8, gpus=1)
                                
	print(f"COMET score: {model_output.system_score}\n")
	with open(LOG, 'a') as op:
		op.write(f"COMET score: {model_output.system_score}\n")


def bleu(l2, references, outputs, LOG):
	references = [references]
	bleu_score = sacrebleu.corpus_bleu(outputs, references)
	print(f"BLEU score: {bleu_score.score}")
	with open(LOG, 'a') as op:
		op.write(f"BLEU score: {bleu_score.score}\n")


def ter_w_shift(l2, references, outputs, IDS_to_exclude=[], LOG=None):
	ter = 0
	cnt = 0
	for i in range(len(outputs)):
		if i in IDS_to_exclude:
			continue
		ter += TER.ter(outputs[i].split(), references[i].split())
		cnt += 1
	print(f"1 - TER Score: {1 - (ter / cnt)}")
	with open(LOG, 'a') as op:
		op.write(f"1 - TER Score: {1 - (ter / cnt)}\n")


def mod_ter_w_shift(l2, references, outputs, nonreferences, ids, lc, IDS_to_exclude=[], LOG=''):
	ter = 0.0
	for i, sid in enumerate(ids):
		if i in IDS_to_exclude:
			continue
		if sid in references:
			ter += compare_TER_w(outputs[i], references[sid], lc)
		else:
			ter += TER_modified.ter(outputs[i].split(), nonreferences[i].split(), lc)
		
	print(f"1 - TERm Score: {1 - (ter / len(ids))}")
	with open(LOG, 'a') as op:
		op.write(f"1 - TERm Score: {1 - (ter / len(ids))}\n")


def exact_window_overlap_match(l2, references, outputs, ids, window, LOG):
	acc = 0.0
	for i, id in enumerate(ids):
		if id in references:
			if outputs[i] != "":
				acc1 = compare_exact_window_overlap(outputs[i], references[id], window)
			acc += acc1
	
	print(f"\tExact Window Overlap Accuracy: {acc / (len(references))}")
	with open(LOG, 'a') as op:
		op.write(f"\tExact Window Overlap Accuracy: {acc / (len(references))}")



parser = argparse.ArgumentParser()
parser.add_argument("--language", help="target language", type=str, default="")
parser.add_argument("--hypothesis", help="hypothesis file", type=str, default="")
parser.add_argument("--source", help="directory where source side sgm file is located", type=str, default="")
parser.add_argument("--target_reference", help="directory where target side sgm file is located", type=str, default="")
parser.add_argument("--log", help="to write all outputs", type=str, default="")
parser.add_argument("--match_counts_path", help="Output file for all match counts per line", type=str, default="")
parser.add_argument("--comet_model_path", help="Path to COMET model to use", type=str, default="")
parser.add_argument("--BLEU", help="", type=str, default="True")
parser.add_argument("--COMET", help="", type=str, default="False")
parser.add_argument("--EXACT_MATCH", help="", type=str, default="True")
parser.add_argument("--WINDOW_OVERLAP", help="", type=str, default="True")
parser.add_argument("--MOD_TER", help="", type=str, default="True")
parser.add_argument("--TER", help="", type=str, default="False")
args = parser.parse_args()

l2 = args.language
windows = [2, 3]

LOG = args.log

SUPPORTED = False
try:
	stanza.download(l2, processors='tokenize,pos,lemma,depparse')
	# switched pretokenized to False, since using SentencePiece.
	l2_stanza = stanza.Pipeline(processors='tokenize,pos,lemma,depparse', lang=l2, use_gpu=True, tokenize_pretokenized=False)
	SUPPORTED = True
except:
	print(f"Language {l2} does not seem to be supported by Stanza -- or something went wrong with downloading the models.")
	print(f"Will continue without searching over lemmatized versions of the data.")
	SUPPORTED = False

ids, outputs = read_outputs_wmt(args.hypothesis)
# Why not en?
#if l2 != "en":
sources, sentreferences, exactreferences = read_reference_data_wmt(args.target_reference, args.source)
if args.BLEU == "True":
    bleu(l2, sentreferences, outputs, LOG)
if args.COMET == "True":
    comet(l2, sources, outputs, sentreferences, args.comet_model_path, LOG)
if args.EXACT_MATCH == "True":
    exact_match(l2, exactreferences, outputs, ids, LOG, args.match_counts_path)
if args.WINDOW_OVERLAP == "True":
    print("Window Overlap Accuracy :")
    with open(LOG, 'a') as op:
        op.write("Window Overlap Accuracy:\n")
    for window in windows:
        print(f"\tWindow {window}:")
        with open(LOG, 'a') as op:
            op.write("Window Overlap Accuracy {window}:\n")
        exact_window_overlap_match(l2, exactreferences, outputs, ids, window, LOG)
if args.MOD_TER == "True":
    mod_ter_w_shift(l2, exactreferences, outputs, sentreferences, ids, 2, [], LOG)
if args.TER == "True":
    ter_w_shift(l2, sentreferences, outputs, [], LOG)
