# Datasets

I found no public EEG dataset that labels trials by how hard the word was
to find (tip-of-the-tongue vs instant recall). That gap is the reason to record new data. The
datasets below each cover one piece: checking the pipeline, measuring speech artefacts, or
borrowing a prior for where the effect should show up.

Run `stuckword datasets` to list them, and `stuckword download <key> --include "sub-01/*"` to
fetch part of an OpenNeuro dataset.

| Key | Dataset | What it is | Why it matters here |
|---|---|---|---|
| `eegbci` | [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/) | 109 subjects, 64 ch; run 1 eyes open, run 2 eyes closed | **Days 1–2.** Known-good alpha reactivity for testing `sanity.py` before the Octopus arrives (`stuckword sanity-eegbci`). Fetched with MNE. |
| `erpcore-n400` | [ERP CORE N400](https://osf.io/29xpq/) (Kappenman et al., 2021, *NeuroImage* 225:117465) | 40 subjects, related vs unrelated word pairs | A word-level effect that every pipeline should reproduce. Good for checking epoching and event handling on language data. |
| `pavlov-digit-span` | [OpenNeuro ds003838](https://openneuro.org/datasets/ds003838) (Pavlov et al., 2022, *Sci Data*) | 64 ch EEG and pupillometry, digit span with loads 5/9/13 | Graded memory load, which is the best public stand-in for "effort". Use it to check whether frontal-midline theta scales with load in your hands, and on which electrodes. The paper reports 86 participants; some have no EEG. |
| `inner-speech` | [OpenNeuro ds003626](https://openneuro.org/datasets/ds003626) "Thinking out loud" (Nieto et al., 2022, *Sci Data*) | 10 subjects, 128 ch; inner, pronounced and visualised speech for the same words | Pronounced vs inner speech shows how far articulation EMG spreads across channels and frequencies. That tells you how much margin the pre-speech window needs. |
| `rennes-naming` | [HD-EEG naming/spelling](https://doi.org/10.1038/s41597-021-00821-1) (Mheich et al., 2021, *Sci Data*) | 23 subjects, 256 ch, 148 pictures named overtly, speech onsets marked | Closest public analogue of the retrieval window: stimulus → word found → speech. The paper says the data are on OpenNeuro, but I couldn't confirm the accession number. Find it in the paper's Data Availability section. |
| `kara-one` | [KaraOne](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) (Zhao & Rudzicz, 2015) | 14 subjects, 64 ch, imagined and spoken phonemes/words | Probably one of the imagined-speech sets you've already used. A benchmark for the existing decoding code. |
| `anomia-tacs` | [OpenNeuro ds007315](https://openneuro.org/datasets/ds007315) (Martzoukou et al., [arXiv 2603.20476](https://arxiv.org/abs/2603.20476)) | 2 chronic post-stroke anomia patients, 32 ch, picture naming across an 8-week tACS programme | **Rehab direction.** The only open naming EEG from people with anomia I found. Too small to train on, but it shows what the signal and artefacts look like in the target population. Check the licence on OpenNeuro. |

## Literature without open data (useful for hypotheses)

- An MEG study of tip-of-the-tongue states in older adults (celebrity naming) found reduced beta
  power in perisylvian areas on TOT trials ([UCL Discovery](https://discovery-pp.ucl.ac.uk/id/eprint/1426016)).
  So beta should go into the features alongside frontal theta and alpha.
- An EEG study linked alpha suppression to the TOT state before feedback on verbal queries
  ([NSF PAR](https://par.nsf.gov/biblio/10486439)).
- CamCAN (MEG plus behavioural TOT measures in a large lifespan cohort) has been used for TOT
  research, but it's MEG and has no trial-level TOT labels.

## What to publish from this project

Release the word list, the recording script, raw XDF files (or BIDS via `mne-bids`), and the
notebooks. Write a BIDS `events.tsv` that uses the same `cue/<label>` and `retrieved/<label>`
names as `stuckword.markers`. As far as I can tell, this would be the first open dataset with
trial-level tip-of-the-tongue labels.
